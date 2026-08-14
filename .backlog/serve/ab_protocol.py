"""Typed parser for colibri mux-serve wire frames."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import BinaryIO, NoReturn


def assert_never(value: NoReturn) -> NoReturn:
    raise AssertionError(f"unreachable protocol variant: {value!r}")


@dataclass(frozen=True, slots=True)
class ProtocolError(Exception):
    detail: str

    def __str__(self) -> str:
        return self.detail


@dataclass(frozen=True, slots=True)
class EngineRequestError(Exception):
    request_id: int
    code: str

    def __str__(self) -> str:
        return f"request {self.request_id}: {self.code}"


@dataclass(frozen=True, slots=True)
class ReadyFrame:
    pass


@dataclass(frozen=True, slots=True)
class DataFrame:
    request_id: int
    text: str


@dataclass(frozen=True, slots=True)
class PerfFrame:
    request_id: int
    dt: float
    t_edisk: float
    t_ewait: float
    t_emm: float
    t_attn: float
    t_kvb: float
    t_head: float


@dataclass(frozen=True, slots=True)
class DoneFrame:
    request_id: int
    emitted: int
    tok_s: float
    hit_pct: float
    rss_gb: float
    prompt_tokens: int
    length_limited: bool
    prefix_reused: int = 0


@dataclass(frozen=True, slots=True)
class SubmitFrame:
    request_id: int
    slot: int
    prompt: str
    max_tokens: int
    temperature: float
    top_p: float


WireEvent = ReadyFrame | DataFrame | PerfFrame | DoneFrame


class ProtocolReader:
    """Parse line headers and byte-counted DATA payloads from mux stdout."""

    def __init__(self, stream: BinaryIO) -> None:
        self._stream = stream

    def wait_for_ready(self) -> bool:
        while True:
            match self.next_event():
                case ReadyFrame():
                    return True
                case DataFrame() | PerfFrame() | DoneFrame() as frame:
                    raise ProtocolError(f"received {type(frame).__name__} before READY")
                case unreachable:
                    assert_never(unreachable)

    def next_event(self) -> WireEvent:
        while True:
            raw_line = self._read_line()
            if b"READY" in raw_line:
                return ReadyFrame()
            try:
                parts = raw_line.decode("utf-8").split()
            except UnicodeDecodeError as exc:
                raise ProtocolError("protocol header is not UTF-8") from exc
            if not parts:
                continue
            match parts[0]:  # noqa: MATCH_OK - unknown kinds must be skipped by protocol.
                case "DATA":
                    return self._parse_data(parts)
                case "PERF":
                    return self._parse_perf(parts)
                case "DONE":
                    return self._parse_done(parts)
                case "ERROR":
                    self._raise_request_error(parts)
                case _:
                    continue

    def _parse_data(self, parts: list[str]) -> DataFrame:
        if len(parts) != 3:
            raise ProtocolError("malformed DATA header")
        try:
            request_id, byte_count = int(parts[1]), int(parts[2])
        except ValueError as exc:
            raise ProtocolError("malformed DATA header") from exc
        if request_id <= 0 or byte_count < 0:
            raise ProtocolError("malformed DATA header")
        payload = self._read_exact(byte_count)
        if self._read_exact(1) != b"\n":
            raise ProtocolError("malformed DATA trailing newline")
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ProtocolError("DATA payload is not UTF-8") from exc
        return DataFrame(request_id=request_id, text=text)

    @staticmethod
    def _parse_perf(parts: list[str]) -> PerfFrame:
        if len(parts) != 9:
            raise ProtocolError("malformed PERF line")
        try:
            request_id = int(parts[1])
            values = tuple(float(value) for value in parts[2:])
        except ValueError as exc:
            raise ProtocolError("malformed PERF line") from exc
        if request_id <= 0 or any(not math.isfinite(value) or value < 0 for value in values):
            raise ProtocolError("malformed PERF line")
        return PerfFrame(
            request_id, values[0], values[1], values[2], values[3], values[4],
            values[5], values[6],
        )

    @staticmethod
    def _parse_done(parts: list[str]) -> DoneFrame:
        # The engine emits a 10th field (session->prefix_reused) and its own comment at
        # c/deepseek_v4.c:9211-9212 states readers must accept len(fields) >= 7 so older
        # readers ignore additions. Accept >= 9 and read the optional 10th.
        if len(parts) < 9 or parts[2] != "STAT":
            raise ProtocolError("malformed DONE line")
        try:
            request_id = int(parts[1])
            emitted = int(parts[3])
            tok_s, hit_pct, rss_gb = (float(value) for value in parts[4:7])
            prompt_tokens, limited = int(parts[7]), int(parts[8])
            prefix_reused = int(parts[9]) if len(parts) > 9 else 0
        except ValueError as exc:
            raise ProtocolError("malformed DONE line") from exc
        metrics = (tok_s, hit_pct, rss_gb)
        if (
            request_id <= 0 or emitted < 0 or prompt_tokens < 0 or limited not in (0, 1)
            or any(not math.isfinite(value) or value < 0 for value in metrics)
        ):
            raise ProtocolError("malformed DONE line")
        return DoneFrame(
            request_id, emitted, tok_s, hit_pct, rss_gb, prompt_tokens, bool(limited),
            prefix_reused,
        )

    @staticmethod
    def _raise_request_error(parts: list[str]) -> None:
        if len(parts) < 3:
            raise ProtocolError("malformed ERROR line")
        try:
            request_id = int(parts[1])
        except ValueError as exc:
            raise ProtocolError("malformed ERROR line") from exc
        raise EngineRequestError(request_id, " ".join(parts[2:]))

    def _read_line(self) -> bytes:
        data = bytearray()
        while True:
            byte = self._stream.read(1)
            if byte == b"":
                raise ProtocolError("unexpected EOF from engine")
            if byte == b"\n":
                return bytes(data)
            data.extend(byte)

    def _read_exact(self, count: int) -> bytes:
        data = bytearray()
        while len(data) < count:
            chunk = self._stream.read(count - len(data))
            if chunk == b"":
                raise ProtocolError("unexpected EOF in DATA frame")
            data.extend(chunk)
        return bytes(data)


def encode_submit(frame: SubmitFrame) -> bytes:
    payload = frame.prompt.encode("utf-8")
    header = (
        f"SUBMIT {frame.request_id} {frame.slot} {len(payload)} "
        f"{frame.max_tokens} {frame.temperature} {frame.top_p}\n"
    ).encode()
    return header + payload + b"\n"
