#!/usr/bin/env python3
"""Serve the colibri web UI + an OpenAI-compatible API backed by the engine.

The engine's SERVE=1 mode speaks a line-framed stdio protocol, not HTTP, and the repo
has no bridge. This is that bridge. UI and API share one origin, which is what
web/src/App.tsx expects: any port != 5173 makes it use window.location.origin + "/v1",
so the browser needs no configuration.

One resident engine (model stays loaded), requests serialised by a lock.
"""
import collections, json, os, subprocess, threading, time, uuid, signal, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT  = "/Users/cptn/workbench/ai/colibri"
BIN   = os.path.join(ROOT, "c", "deepseek_v4")
MODEL = os.path.join(ROOT, "models", "deepseek-v4-flash")
DIST  = os.path.join(ROOT, "web", "dist")
PORT  = int(os.environ.get("PORT", "8000"))
CTX   = int(os.environ.get("CTX", "8192"))
MODEL_ID = "deepseek-v4-flash"
PROFILE_TURNS = 120

FLAGS = dict(kv.split("=", 1) for kv in os.environ.get(
    "FLAGS",
    "COLI_V4_MOE_GROUPED=1 COLI_V4_METAL_ATTN=1 COLI_V4_MOE_BATCHED=1"
).split() if "=" in kv)

lock = threading.Lock()
state_lock = threading.Lock()
proc = None
tiers = None
hwinfo = None
emap = {"rows": 0, "cols": 0, "map": ""}
hits = ""
hits_seq = 0
profile = collections.deque(maxlen=PROFILE_TURNS)
profile_seq = 0

def consume_telemetry(fields):
    """Route one line-framed telemetry record into dashboard state."""
    global tiers, hwinfo, emap, hits, hits_seq, profile_seq
    if not fields:
        return False
    kind = fields[0]
    with state_lock:
        if kind == "HWINFO" and len(fields) >= 7:
            parts = " ".join(fields[6:]).split("|")
            hwinfo = {"cores": int(fields[1]), "ram_total_gb": float(fields[2]),
                      "ram_avail_gb": float(fields[3]), "gpus": int(fields[4]),
                      "vram_total_gb": float(fields[5]),
                      "cpu": parts[0].strip() if parts else "",
                      "gpu": parts[1].strip() if len(parts) > 1 else ""}
        elif kind == "TIERS" and len(fields) >= 6:
            tiers = {"vram": int(fields[1]), "ram": int(fields[2]),
                     "disk": int(fields[3]), "vram_gb": float(fields[4]),
                     "ram_gb": float(fields[5])}
        elif kind == "EMAP" and len(fields) == 4:
            emap = {"rows": int(fields[1]), "cols": int(fields[2]), "map": fields[3]}
        elif kind == "HITS" and len(fields) == 4:
            hits = fields[3]
            hits_seq += 1
        elif kind == "PROF" and len(fields) >= 10:
            profile.append({
                "wall_s": float(fields[1]),
                "prompt_tokens": int(fields[2]),
                "completion_tokens": int(fields[3]),
                "expert_disk_s": float(fields[4]),
                "expert_wait_s": float(fields[5]),
                "expert_matmul_s": float(fields[6]),
                "attention_s": float(fields[7]),
                "lm_head_s": float(fields[8]),
                "forwards": int(fields[9]),
            })
            profile_seq += 1
        else:
            return False
    return True

def experts_snapshot():
    with state_lock:
        return {**emap, "hits": hits, "seq": hits_seq}

def profile_snapshot():
    with state_lock:
        return {"seq": profile_seq, "turns": list(profile)}

def health_telemetry_snapshot():
    with state_lock:
        result = {}
        if tiers is not None:
            result["tiers"] = dict(tiers)
        if hwinfo is not None:
            result["hwinfo"] = dict(hwinfo)
        return result

def start_engine():
    global proc
    env = dict(os.environ, SERVE="1", SNAP=MODEL, CTX=str(CTX),
               COLI_V4_SAVE_USAGE="0", **FLAGS)
    print(f"[bridge] starting engine, flags: {' '.join(f'{k}={v}' for k,v in FLAGS.items())}", flush=True)
    t0 = time.time()
    proc = subprocess.Popen([BIN], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, env=env, bufsize=0)
    while True:
        line = proc.stdout.readline()
        if not line:
            raise SystemExit("[bridge] engine died during startup")
        if b"READY" in line:
            break
    while True:
        line = proc.stdout.readline()
        if not line:
            raise SystemExit("[bridge] engine died during telemetry startup")
        fields = line.decode("utf-8", "replace").split()
        consume_telemetry(fields)
        if fields and fields[0] == "EMAP":
            break
    print(f"[bridge] engine READY in {time.time()-t0:.1f}s -> http://127.0.0.1:{PORT}/", flush=True)

def engine_stream(prompt, max_tokens):
    """Yield ('text', chunk) then ('done', stats)."""
    rid = "w%d" % (int(time.time() * 1000) % 10_000_000)
    payload = prompt.encode("utf-8")
    done_stats = None
    with lock:
        proc.stdin.write(
            f"SUBMIT {rid} 0 {len(payload)} {max_tokens} 0.0 1.0 0\n".encode("ascii")
            + payload + b"\n")
        proc.stdin.flush()
        while True:
            line = proc.stdout.readline()
            if not line:
                raise RuntimeError("engine closed stdout")
            f = line.decode("utf-8", "replace").split()
            if not f:
                continue
            if f[0] == "ERROR":
                raise RuntimeError(line.decode("utf-8", "replace").strip())
            if f[0] == "DATA" and len(f) == 3:
                n = int(f[2]); chunk = proc.stdout.read(n); proc.stdout.read(1)
                yield ("text", chunk.decode("utf-8", "replace"))
            elif f[0] == "DONE":
                # DONE <id> STAT <completion> <tok/s> <hit%> <rss> <prompt> <cap> [<reuse>]
                done_stats = {}
                if len(f) >= 8:
                    done_stats = {"completion_tokens": int(float(f[3])), "tok_s": float(f[4]),
                                  "expert_hit_pct": float(f[5]), "rss_gb": float(f[6]),
                                  "prompt_tokens": int(float(f[7]))}
            elif consume_telemetry(f) and f[0] == "TIERS" and done_stats is not None:
                yield ("done", done_stats)
                return

class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a): pass

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, str): body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self): self._send(204, b"", "text/plain")

    def do_GET(self):
        p = self.path.split("?")[0]
        if p in ("/v1/models", "/models"):
            return self._send(200, json.dumps({"object": "list",
                "data": [{"id": MODEL_ID, "object": "model", "owned_by": "colibri"}]}))
        if p in ("/health", "/v1/health"):
            return self._send(200, json.dumps({
                "status": "ok", "kv_slots": 1,
                "scheduler": {"active": 1 if lock.locked() else 0, "capacity": 1, "queued": 0,
                              "max_queue": 1, "queue_timeout_seconds": 0, "admitted": 0,
                              "completed": 0, "rejected": 0, "timed_out": 0, "cancelled": 0},
                **health_telemetry_snapshot()}))
        if p in ("/profile", "/v1/profile"):
            return self._send(200, json.dumps(profile_snapshot()))
        if p in ("/v1/experts", "/experts"):
            return self._send(200, json.dumps(experts_snapshot()))
        rel = "index.html" if p == "/" else p.lstrip("/")
        full = os.path.normpath(os.path.join(DIST, rel))
        if not full.startswith(DIST) or not os.path.isfile(full):
            return self._send(404, json.dumps({"error": {"message": "not found"}}))
        ext = os.path.splitext(full)[1]
        ctype = {".html": "text/html", ".js": "text/javascript", ".css": "text/css",
                 ".json": "application/json", ".svg": "image/svg+xml",
                 ".woff2": "font/woff2", ".png": "image/png"}.get(ext, "application/octet-stream")
        with open(full, "rb") as fh: return self._send(200, fh.read(), ctype)

    def do_POST(self):
        p = self.path.split("?")[0]
        if p not in ("/v1/chat/completions", "/chat/completions"):
            return self._send(404, json.dumps({"error": {"message": "not found"}}))
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"{}")
        msgs = body.get("messages", [])
        max_tokens = int(body.get("max_completion_tokens") or body.get("max_tokens") or 256)

        # SERVE mode does NOT apply the V4 chat template (v4_serve_one tok_encodes the raw
        # bytes) while the CLI does via coli_v4_prompt_build. Without this the model just
        # continues text instead of answering. Mirrors deepseek_v4.c:8923-8975.
        BOS, USER, ASSISTANT = "<\uff5cbegin\u2581of\u2581sentence\uff5c>", "<\uff5cUser\uff5c>", "<\uff5cAssistant\uff5c>"
        thinking = "<think>" if body.get("enable_thinking") else "</think>"
        sys_msg = next((m["content"] for m in msgs if m.get("role") == "system"), "")
        parts = [BOS, sys_msg]
        for m in [m for m in msgs if m.get("role") in ("user", "assistant")]:
            parts += [USER if m["role"] == "user" else ASSISTANT, m.get("content", "")]
        parts += [ASSISTANT, thinking]
        prompt = "".join(parts)

        cid = "chatcmpl-" + uuid.uuid4().hex[:12]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        def frame(obj):
            self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n"); self.wfile.flush()

        # No client-side clamp: upstream 8c40fbd makes max_tokens a CEILING in the engine,
        # which clamps to (context - prompt_count) using the REAL tokenised prompt length and
        # emits "[V4] max_tokens N clamped to M". Our old estimate-based clamp guessed at that
        # length and could only ever be more conservative than the engine's own answer.
        # CONTEXT_EXCEEDED now means the only honest thing: the prompt itself does not fit.

        try:
            began = time.time(); first = None; stats = {}
            frame({"id": cid, "object": "chat.completion.chunk", "model": MODEL_ID,
                   "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})
            for kind, val in engine_stream(prompt, max_tokens):
                if kind == "text":
                    if first is None:
                        first = time.time()
                        print(f"[bridge] TTFT {first-began:.2f}s", flush=True)
                    frame({"id": cid, "object": "chat.completion.chunk", "model": MODEL_ID,
                           "choices": [{"index": 0, "delta": {"content": val}, "finish_reason": None}]})
                else:
                    stats = val
            frame({"id": cid, "object": "chat.completion.chunk", "model": MODEL_ID,
                   "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                   "usage": {"prompt_tokens": stats.get("prompt_tokens", 0),
                             "completion_tokens": stats.get("completion_tokens", 0),
                             "total_tokens": stats.get("prompt_tokens", 0) + stats.get("completion_tokens", 0)}})
            self.wfile.write(b"data: [DONE]\n\n"); self.wfile.flush()
            print(f"[bridge] done: TTFT {(first or began)-began:.2f}s  "
                  f"{stats.get('completion_tokens',0)} tok @ {stats.get('tok_s',0):.2f} tok/s  "
                  f"wall {time.time()-began:.2f}s", flush=True)
        except Exception as e:
            try:
                frame({"id": cid, "choices": [{"index": 0, "delta": {"content": f"\n[bridge error: {e}]"},
                                               "finish_reason": "stop"}]})
                self.wfile.write(b"data: [DONE]\n\n"); self.wfile.flush()
            except Exception: pass

if __name__ == "__main__":
    if not os.path.isdir(DIST): raise SystemExit(f"[bridge] missing {DIST} (run: cd web && npm run build)")
    start_engine()
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), H)
    print(f"[bridge] serving UI + API on http://127.0.0.1:{PORT}/", flush=True)
    try: srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[bridge] stopping")
        try: proc.terminate()
        except Exception: pass
