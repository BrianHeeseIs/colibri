#!/usr/bin/env python3
"""Drive the colibri V4 engine in mux serve mode over stdin/stdout.
Measures per-request wall time + engine-reported stats, to test whether a
persistent process preserves the heated expert cache across requests."""
import os, subprocess, sys, time, threading, queue

MODEL = sys.argv[1] if len(sys.argv) > 1 else "models/deepseek-v4-flash"
NGEN  = os.environ.get("NGEN", "60")
RAM   = os.environ.get("RAM_GB", "96")

env = dict(os.environ,
           SNAP=MODEL, SERVE="1", SERVE_BATCH="1", NGEN=NGEN,
           KV_SLOTS="1", RAM_GB=RAM, COLI_V4_SAVE_USAGE="0")

p = subprocess.Popen(["./c/deepseek_v4"], stdin=subprocess.PIPE,
                     stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                     env=env, bufsize=0)

outq = queue.Queue()
def pump(stream, tag):
    for raw in iter(stream.readline, b""):
        outq.put((tag, raw))
    outq.put((tag, None))
threading.Thread(target=pump, args=(p.stdout, "o"), daemon=True).start()
threading.Thread(target=pump, args=(p.stderr, "e"), daemon=True).start()

t0 = time.time()
ready = False
while not ready:
    tag, raw = outq.get(timeout=1200)
    if raw is None: print("ENGINE DIED before READY"); sys.exit(1)
    if tag == "o" and b"READY" in raw:
        ready = True
        print(f"  READY after {time.time()-t0:.2f}s")

def submit(rid, prompt, maxtok):
    payload = prompt.encode()
    hdr = f"SUBMIT {rid} 0 {len(payload)} {maxtok} 0.0 1.0\n".encode()
    t = time.time()
    p.stdin.write(hdr + payload + b"\n"); p.stdin.flush()
    perf = done = None; ntok = 0
    while True:
        tag, raw = outq.get(timeout=3600)
        if raw is None: return None
        if tag != "o": continue
        s = raw.decode("utf-8", "replace").rstrip("\n")
        if s.startswith("DATA "): ntok += 1
        elif s.startswith("PERF "): perf = s
        elif s.startswith(f"DONE {rid}"): done = s; break
    return time.time()-t, ntok, perf, done

PROMPT = ("Write a detailed technical explanation of how a mixture-of-experts "
          "transformer routes tokens.")
print(f"  {'req':>4} {'wall_s':>9} {'tok':>5}  DONE/PERF")
for i in range(1, 4):
    r = submit(i, PROMPT, int(NGEN))
    if r is None: print("  engine died"); break
    wall, ntok, perf, done = r
    print(f"  {i:>4} {wall:>9.3f} {ntok:>5}")
    if done: print(f"       {done}")
    if perf: print(f"       {perf}")
p.stdin.close()
try: p.wait(timeout=60)
except Exception: p.kill()
