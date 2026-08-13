#!/usr/bin/env bash
# Audible experiment-complete alert. Runs >=15s so it's heard from another room.
MSG="${1:-Experiment complete}"
osascript -e 'set volume output volume 85' 2>/dev/null
# spoken announcement first (clear + distinctive), twice
say -r 165 "$MSG. Results are ready." 2>/dev/null &
# ~15s of alternating chimes (Glass 1.65s + Hero) regardless of speech length
for i in 1 2 3 4 5; do
  afplay /System/Library/Sounds/Glass.aiff 2>/dev/null
  afplay /System/Library/Sounds/Hero.aiff  2>/dev/null
done
wait
say -r 165 "$MSG" 2>/dev/null
# macOS notification-centre banner as a visual backup
osascript -e "display notification \"$MSG\" with title \"colibri experiment\" sound name \"Glass\"" 2>/dev/null
