"""Owned subprocess lifecycle for colibri mux-serve benchmarks."""
from __future__ import annotations

import os
import queue
import subprocess
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from ab_protocol import (
    DataFrame,
    DoneFrame,
    EngineRequestError,
    PerfFrame,
    ProtocolError,
    ProtocolReader,
    ReadyFrame,
    SubmitFrame,
    WireEvent,
    assert_never,
    encode_submit,
)


@dataclass(frozen=True, slots=True)
class EngineConfig:
    binary: Path
    model_dir: Path
    ngen: int
    kv_slots: int
    ram_gb: float
    ready_timeout_s: float
    request_timeout_s: float
    shutdown_timeout_s: float


@dataclass(frozen=True, slots=True)
class RequestOptions:
    max_tokens: int
    temperature: float
    top_p: float


@dataclass(frozen=True, slots=True)
class TurnRequest:
    index: int
    slot: int
    prompt: str
    options: RequestOptions


@dataclass(frozen=True, slots=True)
class TurnResult:
    wall_s: float
    perf: PerfFrame
    done: DoneFrame


@dataclass(frozen=True, slots=True)
class _PumpFailure:
    error: ProtocolError | EngineRequestError


_PumpItem = WireEvent | _PumpFailure


class EngineSession:
    """Own one engine process and serialize request/response turns."""

    def __init__(self, process: subprocess.Popen[bytes], config: EngineConfig) -> None:
        if process.stdin is None or process.stdout is None:
            process.kill()
            process.wait()
            raise ProtocolError("engine pipes were not created")
        self._process = process
        self._stdin = process.stdin
        self._reader = ProtocolReader(process.stdout)
        self._config = config
        self._events: queue.Queue[_PumpItem] = queue.Queue()
        self.ready_seconds = 0.0

    def start(self, launched_at: float) -> None:
        threading.Thread(target=self._pump, daemon=True).start()
        match self._next_event(self._config.ready_timeout_s):
            case ReadyFrame():
                self.ready_seconds = time.perf_counter() - launched_at
            case DataFrame() | PerfFrame() | DoneFrame() as event:
                raise ProtocolError(f"received {type(event).__name__} before READY")
            case unreachable:
                assert_never(unreachable)

    def submit(self, request: TurnRequest) -> TurnResult:
        request_id = request.index + 1
        started = time.perf_counter()
        self._stdin.write(encode_submit(SubmitFrame(
            request_id=request_id,
            slot=request.slot,
            prompt=request.prompt,
            max_tokens=request.options.max_tokens,
            temperature=request.options.temperature,
            top_p=request.options.top_p,
        )))
        self._stdin.flush()
        perf: PerfFrame | None = None
        while True:
            match self._next_event(self._config.request_timeout_s):
                case DataFrame(request_id=event_id):
                    self._require_request_id(request_id, event_id)
                case PerfFrame(request_id=event_id) as frame:
                    self._require_request_id(request_id, event_id)
                    perf = frame
                case DoneFrame(request_id=event_id) as done:
                    self._require_request_id(request_id, event_id)
                    if perf is None:
                        # V4 emits NO PERF line (0 PERF printfs in c/deepseek_v4.c); PERF is
                        # GLM/advisory telemetry. The protocol requires readers to tolerate
                        # absent optional lines, so a missing PERF is normal, not an error.
                        # V4 stage data comes from COLI_V4_PROFILE=1 on stderr instead.
                        perf = PerfFrame(request_id, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                    return TurnResult(time.perf_counter() - started, perf, done)
                case ReadyFrame():
                    raise ProtocolError("duplicate READY from engine")
                case unreachable:
                    assert_never(unreachable)

    def send_control(self, frame: bytes) -> None:
        self._stdin.write(frame if frame.endswith(b"\n") else frame + b"\n")
        self._stdin.flush()

    def shutdown(self) -> None:
        try:
            self._stdin.close()
        except OSError:
            self._process.kill()
            self._process.wait()
            return
        try:
            self._process.wait(timeout=self._config.shutdown_timeout_s)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()

    def _pump(self) -> None:
        try:
            while True:
                self._events.put(self._reader.next_event())
        except (ProtocolError, EngineRequestError) as exc:
            self._events.put(_PumpFailure(exc))
        except OSError as exc:
            self._events.put(_PumpFailure(ProtocolError(f"engine stdout failed: {exc}")))

    def _next_event(self, timeout_s: float) -> WireEvent:
        try:
            item = self._events.get(timeout=timeout_s)
        except queue.Empty as exc:
            raise ProtocolError(f"engine response timed out after {timeout_s}s") from exc
        match item:
            case ReadyFrame() | DataFrame() | PerfFrame() | DoneFrame():
                return item
            case _PumpFailure(error=error):
                raise error
            case unreachable:
                assert_never(unreachable)

    @staticmethod
    def _require_request_id(expected: int, actual: int) -> None:
        if actual != expected:
            raise ProtocolError(f"expected response id {expected}, got {actual}")


@contextmanager
def launch_engine(config: EngineConfig) -> Iterator[EngineSession]:
    env = os.environ.copy()
    env.update(
        SNAP=str(config.model_dir),
        SERVE="1",
        SERVE_BATCH="1",
        NGEN=str(config.ngen),
        KV_SLOTS=str(config.kv_slots),
        RAM_GB=str(config.ram_gb),
        COLI_V4_SAVE_USAGE="0",
    )
    launched_at = time.perf_counter()
    process = subprocess.Popen(
        [str(config.binary)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=None, env=env, bufsize=0,
    )
    session = EngineSession(process, config)
    try:
        session.start(launched_at)
        yield session
    finally:
        session.shutdown()
