from __future__ import annotations

import importlib.util
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace


def load_bridge():
    path = Path(__file__).with_name("serve_web.py")
    spec = importlib.util.spec_from_file_location("serve_web_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_engine_stream_routes_post_done_telemetry() -> None:
    # Given
    bridge = load_bridge()
    bridge.proc = SimpleNamespace(
        stdin=BytesIO(),
        stdout=BytesIO(
            b"ACCEPT request 3\n"
            b"DATA request 2\nok\n"
            b"DONE request STAT 2 4.5 75.0 8.0 3 0 0\n"
            b"PROF 1.250 3 2 0.100 0.000 0.700 0.000 0.000 0\n"
            b"HITS 1 8 81\n"
            b"EMAP 1 8 0001020304050647\n"
            b"TIERS 0 1 7 0.00 0.25\n"
        ),
    )

    # When
    events = list(bridge.engine_stream("hi", 2))

    # Then
    assert events == [
        ("text", "ok"),
        (
            "done",
            {
                "completion_tokens": 2,
                "tok_s": 4.5,
                "expert_hit_pct": 75.0,
                "rss_gb": 8.0,
                "prompt_tokens": 3,
            },
        ),
    ]
    assert bridge.experts_snapshot() == {
        "rows": 1,
        "cols": 8,
        "map": "0001020304050647",
        "hits": "81",
        "seq": 1,
    }
    assert bridge.profile_snapshot() == {
        "seq": 1,
        "turns": [
            {
                "wall_s": 1.25,
                "prompt_tokens": 3,
                "completion_tokens": 2,
                "expert_disk_s": 0.1,
                "expert_wait_s": 0.0,
                "expert_matmul_s": 0.7,
                "attention_s": 0.0,
                "lm_head_s": 0.0,
                "forwards": 0,
            }
        ],
    }


def test_startup_telemetry_populates_health() -> None:
    # Given
    bridge = load_bridge()

    # When
    bridge.consume_telemetry(
        "HWINFO 16 128.0 64.0 1 40.0 Apple M3 Max|Apple GPU".split()
    )
    bridge.consume_telemetry("TIERS 0 32 224 0.00 4.25".split())

    # Then
    assert bridge.health_telemetry_snapshot() == {
        "tiers": {
            "vram": 0,
            "ram": 32,
            "disk": 224,
            "vram_gb": 0.0,
            "ram_gb": 4.25,
        },
        "hwinfo": {
            "cores": 16,
            "ram_total_gb": 128.0,
            "ram_avail_gb": 64.0,
            "gpus": 1,
            "vram_total_gb": 40.0,
            "cpu": "Apple M3 Max",
            "gpu": "Apple GPU",
        },
    }
