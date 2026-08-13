#!/usr/bin/env bash
# run_experiment.sh <label> <csvname> [ENV=VAL ...]
# Restarts the server with the given env, runs the fixed 10-question set (15 min cap), alerts.
LABEL="$1"; CSV="$2"; shift 2
BASE=/Users/cptn/workbench/ai/colibri
MODEL=$BASE/models/deepseek-v4-flash
cd "$BASE/validation/dsv4"


# ENGINE PROVENANCE GUARD (added 2026-08-12, RESULTS.md S10).
# bin/coli and c/coli resolve to DIFFERENT binaries: engine_for() (c/coli:249) prefers
# HERE/deepseek_v4 over libexec/. Running a feature test against a stale libexec build
# produces a false negative that looks like a real result. Print what we are about to run.
_ENG="$BASE/libexec/colibri/deepseek_v4"
[ -x "$BASE/bin/deepseek_v4" ] && _ENG="$BASE/bin/deepseek_v4"
echo "  ENGINE: $_ENG  (built $(stat -f '%Sm' -t '%m-%d %H:%M' "$_ENG" 2>/dev/null || echo '?'))"
echo "  NOTE  : c/deepseek_v4 is $(stat -f '%Sm' -t '%m-%d %H:%M' "$BASE/c/deepseek_v4" 2>/dev/null || echo absent) -- if newer, this harness is testing OLD code"

echo "=== [$LABEL] restarting server with: $* (--ram ${V4RAM:-48}) ==="
pkill -f 'coli serve' 2>/dev/null; pkill -f openai_server 2>/dev/null
pkill -f 'libexec/colibri/deepseek_v4' 2>/dev/null
tmux send-keys -t colibri-lab:0.0 C-c 2>/dev/null
for i in $(seq 1 30); do sleep 2; [ -z "$(lsof -ti:8090 2>/dev/null)" ] && break; done
sleep 4
tmux send-keys -t colibri-lab:0.0 "clear && env $* COLI_MODEL=$MODEL $BASE/bin/coli serve --ram ${V4RAM:-48} --port 8090" Enter
MID=""
for i in $(seq 1 90); do
  sleep 5
  MID=$(curl -s --max-time 5 http://127.0.0.1:8090/v1/models 2>/dev/null | python3 -c "import json,sys;print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null)
  [ -n "$MID" ] || continue
  PROBE=$(curl -s --max-time 600 -X POST http://127.0.0.1:8090/v1/chat/completions -H 'Content-Type: application/json' \
     -d "{\"model\":\"$MID\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1,\"temperature\":0}" \
     | python3 -c "import json,sys;print(json.load(sys.stdin).get('usage',{}).get('completion_tokens',0))" 2>/dev/null||echo 0)
  [ "${PROBE:-0}" -ge 1 ] && { echo "server READY (probe generated $PROBE token)"; break; }
  MID=""
done
[ -n "$MID" ] || { echo "SERVER FAILED TO BECOME READY"; ./notify.sh "$LABEL failed to start"; exit 1; }
echo "server ready, model=$MID"

DEADLINE=$(( $(date +%s) + 900 ))
echo "n,ts,wall_s,tokens,tok_s,question" > "$CSV"
Q=("What is a mixture-of-experts model?"
"How does the router choose which experts to activate?"
"Why do MoE models activate only a few experts per token?"
"What is expert capacity and why does it matter in MoE?"
"How does MoE compare to a dense transformer of equal size?"
"What causes load imbalance between experts during MoE training?"
"How is an auxiliary loss used to balance MoE expert routing?"
"What are the memory trade-offs of serving an MoE model?"
"How does top-k routing affect MoE inference latency?"
"What is a mixture-of-experts model?")
n=0
printf "  n  time      wall  tok/s   question\n"
for q in "${Q[@]}"; do
  [ "$(date +%s)" -ge "$DEADLINE" ] && { echo "*** 15 MIN CAP ***"; break; }
  n=$((n+1)); S=$(date +%s)
  curl -s -X POST http://127.0.0.1:8090/v1/chat/completions -H 'Content-Type: application/json' \
    -d "$(python3 -c "import json,sys;print(json.dumps({'model':'$MID','messages':[{'role':'user','content':sys.argv[1]}],'max_tokens':24,'temperature':0}))" "$q")" \
    --max-time 880 -o /tmp/${CSV%.csv}_$n.json
  E=$(date +%s); W=$((E-S))
  T=$(python3 -c "import json;print(json.load(open('/tmp/${CSV%.csv}_$n.json')).get('usage',{}).get('completion_tokens',0))" 2>/dev/null||echo 0)
  R=$(python3 -c "print(f'{$T/max(1,$W):.4f}')")
  echo "$n,$(date +%H:%M:%S),$W,$T,$R,\"$q\"" >> "$CSV"
  printf "%3d  %s %5ss %7s   %.44s\n" "$n" "$(date +%H:%M:%S)" "$W" "$R" "$q"
done
echo "=== [$LABEL] DONE $(date +%H:%M:%S) -> $CSV ==="
./notify.sh "$LABEL complete"
