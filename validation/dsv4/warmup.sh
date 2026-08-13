#!/usr/bin/env bash
# Charts expert-cache warming: same topic, different questions, fixed token budget.
MID=$(cat /tmp/dsv4_model_id); OUT=warmup.csv; LOG=warmup.log
DEADLINE=$(( $(date +%s) + 900 ))   # 15 minutes
echo "n,ts,wall_s,tokens,tok_s,question" > "$OUT"
Q=(
"What is a mixture-of-experts model?"
"How does the router choose which experts to activate?"
"Why do MoE models activate only a few experts per token?"
"What is expert capacity and why does it matter in MoE?"
"How does MoE compare to a dense transformer of equal size?"
"What causes load imbalance between experts during MoE training?"
"How is an auxiliary loss used to balance MoE expert routing?"
"What are the memory trade-offs of serving an MoE model?"
"How does top-k routing affect MoE inference latency?"
"Why is expert parallelism used when scaling MoE models?"
)
n=0
for q in "${Q[@]}"; do
  [ "$(date +%s)" -ge "$DEADLINE" ] && { echo "deadline reached" >> "$LOG"; break; }
  n=$((n+1)); S=$(date +%s)
  curl -s -X POST http://127.0.0.1:8090/v1/chat/completions -H 'Content-Type: application/json' \
    -d "$(python3 -c "import json,sys;print(json.dumps({'model':'$MID','messages':[{'role':'user','content':sys.argv[1]}],'max_tokens':24,'temperature':0}))" "$q")" \
    --max-time 890 -o /tmp/w_$n.json
  E=$(date +%s); W=$((E-S))
  T=$(python3 -c "import json;print(json.load(open('/tmp/w_$n.json')).get('usage',{}).get('completion_tokens',0))" 2>/dev/null||echo 0)
  R=$(python3 -c "print(f'{$T/max(1,$W):.4f}')")
  echo "$n,$(date +%H:%M:%S),$W,$T,$R,\"$q\"" >> "$OUT"
  echo "[$(date +%H:%M:%S)] #$n wall=${W}s tokens=$T tok/s=$R" | tee -a "$LOG"
done
echo "DONE $(date +%H:%M:%S)" >> "$LOG"
