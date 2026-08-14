#!/usr/bin/env python3
"""Benchmark a persistent colibri V4 mux-serve engine."""
from __future__ import annotations

import argparse
import csv
import statistics
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Final, Sequence, TextIO

from ab_protocol import (
    DataFrame,
    DoneFrame,
    EngineRequestError,
    PerfFrame,
    ProtocolError,
    ProtocolReader,
    SubmitFrame,
    assert_never,
    encode_submit,
)
from ab_runtime import EngineConfig, EngineSession, RequestOptions, TurnRequest, launch_engine


CSV_HEADER: Final = (
    "request_index", "wall_s", "emitted_tokens", "tok_s", "hit_pct",
    "rss_gb", "prompt_tokens", "length_limited", "prefix_reused", "dt", "t_edisk",
    "t_ewait", "t_emm", "t_attn", "t_kvb", "t_head",
)


@dataclass(frozen=True, slots=True)
class RequestRecord:
    index: int
    wall_s: float
    emitted: int
    tok_s: float
    hit_pct: float
    rss_gb: float
    prompt_tokens: int
    length_limited: bool
    perf: PerfFrame
    prefix_reused: int = 0


@dataclass(frozen=True, slots=True)
class SteadyStateConfig:
    discard_first_n: int
    window_m: int
    hit_pct_band: float


@dataclass(frozen=True, slots=True)
class SteadyStateSummary:
    start_request_index: int
    end_request_index: int
    median_wall_s: float
    median_tok_s: float


def summarize_steady_state(
    records: Sequence[RequestRecord], config: SteadyStateConfig,
) -> SteadyStateSummary | None:
    if config.discard_first_n < 0 or config.window_m < 1 or config.hit_pct_band < 0:
        raise ProtocolError("steady-state arguments must be non-negative and window_m positive")
    candidates = records[config.discard_first_n:]
    for start in range(len(candidates) - config.window_m + 1):
        window = candidates[start:start + config.window_m]
        hit_values = [record.hit_pct for record in window]
        if max(hit_values) - min(hit_values) <= config.hit_pct_band:
            return SteadyStateSummary(
                start_request_index=window[0].index,
                end_request_index=window[-1].index,
                median_wall_s=statistics.median(record.wall_s for record in window),
                median_tok_s=statistics.median(record.tok_s for record in window),
            )
    return None


class ResetStrategy(str, Enum):
    SLOT_ROTATE = "slot_rotate"
    RESET_FRAME = "reset_frame"
    RESTART = "restart"
    NONE = "none"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    engine: EngineConfig
    request: RequestOptions
    output_csv: Path
    strategy: ResetStrategy
    reset_frame: bytes | None


@dataclass(frozen=True, slots=True)
class BenchmarkRun:
    records: tuple[RequestRecord, ...]
    ready_seconds: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class PromptCase:
    index: int
    prompt: str


ResetHook = Callable[[EngineSession, int], int]


class _CsvSink:
    def __init__(self, output: TextIO) -> None:
        self._output = output
        self._writer = csv.writer(output)
        self._writer.writerow(CSV_HEADER)

    def append(self, record: RequestRecord) -> None:
        perf = record.perf
        self._writer.writerow((
            record.index, record.wall_s, record.emitted, record.tok_s,
            record.hit_pct, record.rss_gb, record.prompt_tokens,
            int(record.length_limited), record.prefix_reused,
            perf.dt, perf.t_edisk, perf.t_ewait,
            perf.t_emm, perf.t_attn, perf.t_kvb, perf.t_head,
        ))
        self._output.flush()


class _BenchmarkRunner:
    """Accumulate benchmark records while owning each required engine session."""

    def __init__(self, config: BenchmarkConfig, reset_hook: ResetHook | None) -> None:
        self._config = config
        self._reset_hook = reset_hook
        self._records: list[RequestRecord] = []
        self._ready_seconds: list[float] = []

    def run(self, prompts: Sequence[str]) -> BenchmarkRun:
        self._validate(prompts)
        self._config.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with self._config.output_csv.open("w", newline="", encoding="utf-8") as output:
            sink = _CsvSink(output)
            if self._config.strategy is ResetStrategy.RESTART:
                for case in self._cases(prompts):
                    with launch_engine(self._config.engine) as session:
                        self._ready_seconds.append(session.ready_seconds)
                        self._measure(session, case, sink)
            else:
                with launch_engine(self._config.engine) as session:
                    self._ready_seconds.append(session.ready_seconds)
                    for case in self._cases(prompts):
                        self._measure(session, case, sink)
        return BenchmarkRun(tuple(self._records), tuple(self._ready_seconds))

    def _measure(self, session: EngineSession, case: PromptCase, sink: _CsvSink) -> None:
        slot = self._slot(session, case.index)
        turn = session.submit(TurnRequest(case.index, slot, case.prompt, self._config.request))
        done = turn.done
        record = RequestRecord(
            case.index + 1, turn.wall_s, done.emitted, done.tok_s, done.hit_pct,
            done.rss_gb, done.prompt_tokens, done.length_limited, turn.perf,
            done.prefix_reused,
        )
        self._records.append(record)
        sink.append(record)

    def _slot(self, session: EngineSession, index: int) -> int:
        if self._reset_hook is not None:
            return self._reset_hook(session, index)
        match self._config.strategy:
            case ResetStrategy.SLOT_ROTATE:
                return index % self._config.engine.kv_slots
            case ResetStrategy.RESET_FRAME:
                if self._config.reset_frame is None:
                    raise ProtocolError("reset frame bytes missing")
                session.send_control(self._config.reset_frame)
                return 0
            case ResetStrategy.RESTART | ResetStrategy.NONE:
                return 0
            case unreachable:
                assert_never(unreachable)

    def _validate(self, prompts: Sequence[str]) -> None:
        if not prompts or any(not prompt for prompt in prompts):
            raise ProtocolError("at least one non-empty prompt is required")
        # Adjacent duplicates would let kv_prefix_reuse() extend the previous turn
        # (prior fed sequence is a strict prefix of the new prompt) and SKIP prefill.
        # Non-adjacent repeats are safe and are REQUIRED by the cycled-prompt protocol:
        # after turn N the fed record is prompt+generation, which is LONGER than the
        # repeated prompt, so kv_prefix.h:116-127 returns 0 and forces a full reset.
        for a, b in zip(prompts, prompts[1:]):
            if a == b:
                raise ProtocolError("adjacent prompts must differ (would skip prefill via KV reuse)")
        if self._config.strategy is ResetStrategy.SLOT_ROTATE and self._config.engine.kv_slots < 2:
            raise ProtocolError("slot_rotate requires KV_SLOTS > 1")
        if self._config.strategy is ResetStrategy.RESET_FRAME and self._config.reset_frame is None:
            raise ProtocolError("reset_frame strategy requires reset frame bytes")

    @staticmethod
    def _cases(prompts: Sequence[str]) -> tuple[PromptCase, ...]:
        return tuple(PromptCase(index, prompt) for index, prompt in enumerate(prompts))


def run_benchmark(
    config: BenchmarkConfig,
    prompts: Sequence[str],
    reset_hook: ResetHook | None = None,
) -> BenchmarkRun:
    return _BenchmarkRunner(config, reset_hook).run(prompts)


def _load_prompts(values: list[str], files: list[Path], prompt_list: Path | None) -> list[str]:
    prompts = list(values)
    prompts.extend(path.read_text(encoding="utf-8") for path in files)
    if prompt_list is not None:
        prompts.extend(line for line in prompt_list.read_text(encoding="utf-8").splitlines() if line)
    return prompts


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--prompt-file", type=Path, action="append", default=[])
    parser.add_argument("--prompt-list", type=Path)
    parser.add_argument("--ngen", type=int, required=True)
    parser.add_argument("--kv-slots", type=int, required=True)
    parser.add_argument("--ram-gb", type=float, default=96.0)
    parser.add_argument("--reset-strategy", type=ResetStrategy, choices=list(ResetStrategy), required=True)
    parser.add_argument("--reset-frame-hex", help="exact reset control frame bytes as hex")
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--discard-first-n", type=int, default=1)
    parser.add_argument("--window-m", type=int, default=3)
    parser.add_argument("--hit-pct-band", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--ready-timeout", type=float, default=1200.0)
    parser.add_argument("--request-timeout", type=float, default=3600.0)
    parser.add_argument("--shutdown-timeout", type=float, default=60.0)
    return parser


def main() -> int:
    args = _parser().parse_args()
    reset_frame = bytes.fromhex(args.reset_frame_hex) if args.reset_frame_hex else None
    engine = EngineConfig(
        args.binary, args.model_dir, args.ngen, args.kv_slots, args.ram_gb,
        args.ready_timeout, args.request_timeout, args.shutdown_timeout,
    )
    config = BenchmarkConfig(
        engine=engine,
        request=RequestOptions(args.ngen, args.temperature, args.top_p),
        output_csv=args.output_csv,
        strategy=args.reset_strategy,
        reset_frame=reset_frame,
    )
    prompts = _load_prompts(args.prompt, args.prompt_file, args.prompt_list)
    run = run_benchmark(config, prompts)
    print(f"time-to-READY seconds: {', '.join(f'{value:.3f}' for value in run.ready_seconds)}")
    summary = summarize_steady_state(
        run.records, SteadyStateConfig(args.discard_first_n, args.window_m, args.hit_pct_band),
    )
    if summary is None:
        print("steady state: no qualifying hit_pct-stable window")
    else:
        print(
            f"steady state: requests {summary.start_request_index}-{summary.end_request_index}, "
            f"median wall={summary.median_wall_s:.3f}s, "
            f"median tok/s={summary.median_tok_s:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
