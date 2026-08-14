from __future__ import annotations

import csv
from dataclasses import replace
from io import BytesIO
from pathlib import Path

import pytest

from ab_harness import (
    BenchmarkConfig,
    DataFrame,
    DoneFrame,
    EngineConfig,
    EngineRequestError,
    PerfFrame,
    ProtocolError,
    ProtocolReader,
    RequestOptions,
    RequestRecord,
    ResetStrategy,
    SteadyStateConfig,
    SubmitFrame,
    encode_submit,
    run_benchmark,
    summarize_steady_state,
)


def test_data_payload_uses_byte_count_when_token_contains_newline() -> None:
    # Given
    payload = "alpha\nβeta".encode()
    stream = BytesIO(f"DATA 7 {len(payload)}\n".encode() + payload + b"\n")

    # When
    frame = ProtocolReader(stream).next_event()

    # Then
    assert frame == DataFrame(request_id=7, text="alpha\nβeta")


def test_reader_skips_unknown_and_advisory_lines() -> None:
    # Given
    stream = BytesIO(
        b"FUTURE_METRIC 1 2 3\n"
        b"HITS 1 1 ff\n"
        b"PERF 9 1.0 2.0 3.0 4.0 5.0 6.0 7.0\n"
    )

    # When
    frame = ProtocolReader(stream).next_event()

    # Then
    assert frame == PerfFrame(9, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)


def test_reader_recognizes_framed_ready_after_startup_telemetry() -> None:
    # Given
    stream = BytesIO(b"STAT 0 0.00 0.0 1.2\n\x01\x01READY\x01\x01\n")

    # When
    ready = ProtocolReader(stream).wait_for_ready()

    # Then
    assert ready is True


def test_done_parses_every_stat_field() -> None:
    # Given
    stream = BytesIO(b"DONE 11 STAT 60 1.25 87.5 42.75 123 1\n")

    # When
    frame = ProtocolReader(stream).next_event()

    # Then
    assert frame == DoneFrame(11, 60, 1.25, 87.5, 42.75, 123, True)


def test_perf_rejects_non_finite_stage_time() -> None:
    # Given
    reader = ProtocolReader(BytesIO(b"PERF 9 1.0 2.0 inf 4.0 5.0 6.0 7.0\n"))

    # When / Then
    with pytest.raises(ProtocolError, match="malformed PERF"):
        reader.next_event()


@pytest.mark.parametrize(
    "line",
    [
        b"DONE 11 STATS 60 1.25 87.5 42.75 123 1\n",
        b"DONE 11 STAT 60 1.25 87.5 42.75 123\n",
        b"DONE 11 STAT sixty 1.25 87.5 42.75 123 1\n",
        b"DONE 11 STAT 60 1.25 87.5 42.75 123 maybe\n",
        b"DONE 11 STAT 60 nan 87.5 42.75 123 0\n",
    ],
)
def test_malformed_done_fails_loudly(line: bytes) -> None:
    # Given
    reader = ProtocolReader(BytesIO(line))

    # When / Then
    with pytest.raises(ProtocolError, match="malformed DONE"):
        reader.next_event()


def test_error_frame_is_hard_request_failure() -> None:
    # Given
    reader = ProtocolReader(BytesIO(b"ERROR 17 SLOT_BUSY\n"))

    # When / Then
    with pytest.raises(EngineRequestError, match="request 17: SLOT_BUSY"):
        reader.next_event()


def test_data_requires_trailing_newline_after_exact_payload() -> None:
    # Given
    reader = ProtocolReader(BytesIO(b"DATA 3 5\nhelloX"))

    # When / Then
    with pytest.raises(ProtocolError, match="DATA trailing newline"):
        reader.next_event()


def test_submit_frame_uses_utf8_byte_length() -> None:
    # Given
    prompt = "line one\nβ"

    # When
    frame = encode_submit(SubmitFrame(5, 2, prompt, 16, 0.0, 1.0))

    # Then
    assert frame == b"SUBMIT 5 2 11 16 0.0 1.0\nline one\n\xce\xb2\n"


def test_summary_uses_first_stable_window_after_discard() -> None:
    # Given
    records = [
        _record(1, 20.0, 1.0, 50.0),
        _record(2, 12.0, 2.0, 70.0),
        _record(3, 10.0, 4.0, 70.4),
        _record(4, 8.0, 6.0, 70.2),
        _record(5, 2.0, 20.0, 90.0),
    ]

    # When
    summary = summarize_steady_state(records, SteadyStateConfig(1, 3, 0.5))

    # Then
    assert summary is not None
    assert summary.start_request_index == 2
    assert summary.end_request_index == 4
    assert summary.median_wall_s == 10.0
    assert summary.median_tok_s == 4.0


def test_summary_returns_none_when_no_stable_window_exists() -> None:
    # Given
    records = [
        _record(1, 3.0, 1.0, 10.0),
        _record(2, 2.0, 2.0, 20.0),
        _record(3, 1.0, 3.0, 30.0),
    ]

    # When
    summary = summarize_steady_state(records, SteadyStateConfig(0, 2, 1.0))

    # Then
    assert summary is None


def test_benchmark_writes_each_fake_engine_turn_to_csv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    binary, closed_marker = _fake_engine(tmp_path)
    monkeypatch.setenv("FAKE_ENGINE_CLOSED", str(closed_marker))
    output_csv = tmp_path / "results.csv"
    config = _benchmark_config(binary, output_csv)

    # When
    run = run_benchmark(config, ["first distinct prompt", "second distinct prompt"])

    # Then
    with output_csv.open(newline="", encoding="utf-8") as output:
        rows = list(csv.DictReader(output))
    assert len(run.ready_seconds) == 1
    assert [record.emitted for record in run.records] == [2, 2]
    assert [row["request_index"] for row in rows] == ["1", "2"]
    assert rows[0]["t_head"] == "0.7"
    assert closed_marker.read_text(encoding="utf-8") == "closed"


def test_malformed_done_still_closes_fake_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    binary, closed_marker = _fake_engine(tmp_path)
    monkeypatch.setenv("FAKE_ENGINE_CLOSED", str(closed_marker))
    monkeypatch.setenv("FAKE_ENGINE_MALFORMED", "1")
    config = _benchmark_config(binary, tmp_path / "failed.csv")

    # When / Then
    with pytest.raises(ProtocolError, match="malformed DONE"):
        run_benchmark(config, ["one prompt"])
    assert closed_marker.read_text(encoding="utf-8") == "closed"


def _record(index: int, wall_s: float, tok_s: float, hit_pct: float) -> RequestRecord:
    return RequestRecord(
        index=index,
        wall_s=wall_s,
        emitted=1,
        tok_s=tok_s,
        hit_pct=hit_pct,
        rss_gb=1.0,
        prompt_tokens=1,
        length_limited=False,
        perf=PerfFrame(index, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )


def _benchmark_config(binary: Path, output_csv: Path) -> BenchmarkConfig:
    return BenchmarkConfig(
        engine=EngineConfig(binary, Path("unused-model"), 2, 1, 96.0, 2.0, 2.0, 2.0),
        request=RequestOptions(2, 0.0, 1.0),
        output_csv=output_csv,
        strategy=ResetStrategy.NONE,
        reset_frame=None,
    )


def _fake_engine(tmp_path: Path) -> tuple[Path, Path]:
    binary = tmp_path / "fake_mux_engine.py"
    closed_marker = tmp_path / "closed.txt"
    binary.write_text(
        """#!/usr/bin/env python3
import os
import sys
from pathlib import Path

output = sys.stdout.buffer
source = sys.stdin.buffer
output.write(b"\\x01\\x01READY\\x01\\x01\\nSTAT 0 0.00 0.0 1.0\\n")
output.flush()
while True:
    header = source.readline()
    if not header:
        break
    parts = header.split()
    request_id = int(parts[1])
    byte_count = int(parts[3])
    source.read(byte_count)
    source.read(1)
    payload = b"x\\ny"
    output.write(f"DATA {request_id} {len(payload)}\\n".encode() + payload + b"\\n")
    output.write(f"PERF {request_id} 0.1 0.2 0.3 0.4 0.5 0.6 0.7\\n".encode())
    if os.environ.get("FAKE_ENGINE_MALFORMED"):
        output.write(f"DONE {request_id} STAT broken\\n".encode())
    else:
        output.write(f"DONE {request_id} STAT 2 4.5 70.0 8.0 12 0\\n".encode())
    output.flush()
Path(os.environ["FAKE_ENGINE_CLOSED"]).write_text("closed", encoding="utf-8")
""",
        encoding="utf-8",
    )
    binary.chmod(binary.stat().st_mode | 0o111)
    return binary, closed_marker


def test_adjacent_duplicate_prompts_rejected(tmp_path: Path) -> None:
    """Adjacency rejection is CONSERVATIVE, not a V4 correctness requirement.

    V4 cannot reuse here (the fed record is prompt+generation, strictly longer than a
    repeated prompt, and kv_prefix.h:116-127 returns 0 for shorter/equal). The guard
    exists because identical neighbours indicate a caller mistake, and would matter on
    an engine with true longest-common-prefix reuse.
    """
    from ab_harness import _BenchmarkRunner

    config = _benchmark_config(tmp_path / "fake", tmp_path / "out.csv")
    runner = _BenchmarkRunner(config, lambda _s: None)
    with pytest.raises(ProtocolError, match="adjacent"):
        runner._validate(["same prompt", "same prompt"])


def test_non_adjacent_repeat_allowed(tmp_path: Path) -> None:
    """The cycled-prompt protocol REQUIRES non-adjacent repeats.

    After turn N the fed record is prompt+generation, which is LONGER than the repeated
    prompt, so kv_prefix.h:116-127 returns 0 and the engine performs a full attention reset.
    """
    from ab_harness import _BenchmarkRunner

    config = _benchmark_config(tmp_path / "fake", tmp_path / "out.csv")
    runner = _BenchmarkRunner(config, lambda _s: None)
    runner._validate(["alpha", "beta", "alpha", "beta"])  # must not raise


def test_done_accepts_engine_tenth_field_prefix_reused() -> None:
    """The real engine emits a 10th field, session->prefix_reused (c/deepseek_v4.c:9213-9217).

    Its own comment states readers must accept len(fields) >= 7 so older readers ignore
    additions, so the documented 9-field form must not be treated as exact. The harness
    originally hard-failed on the real engine because of this.
    """
    stream = BytesIO(b"DONE 7 STAT 60 1.474 75.2 66.43 16 1 0\n")
    frame = ProtocolReader(stream).next_event()
    assert isinstance(frame, DoneFrame)
    assert frame.prompt_tokens == 16
    assert frame.prefix_reused == 0


def test_done_reports_prefix_reuse_when_engine_sets_it() -> None:
    """prefix_reused=1 proves the engine SKIPPED prefill via KV reuse - a benchmark invalidator."""
    stream = BytesIO(b"DONE 7 STAT 60 1.474 75.2 66.43 16 1 1\n")
    frame = ProtocolReader(stream).next_event()
    assert isinstance(frame, DoneFrame)
    assert frame.prefix_reused == 1


def test_v4_invalid_reset_strategies_hard_fail(tmp_path: Path) -> None:
    """slot_rotate and reset_frame cannot work on V4; offering them would silently mismeasure."""
    from ab_harness import _BenchmarkRunner

    for strategy, needle in ((ResetStrategy.SLOT_ROTATE, "slot"), (ResetStrategy.RESET_FRAME, "reset_frame")):
        config = replace(_benchmark_config(tmp_path / "fake", tmp_path / "o.csv"), strategy=strategy)
        runner = _BenchmarkRunner(config, lambda _s: None)
        with pytest.raises(ProtocolError, match=needle):
            runner._validate(["alpha", "beta"])
