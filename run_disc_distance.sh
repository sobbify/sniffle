#!/usr/bin/env bash
set -e
cd "$(dirname "$(readlink -f "$0")")"
export DISPLAY="${DISPLAY:-:0}"
unset QT_QPA_PLATFORM
SERVO_ARGS=()
if [ -n "${SERVO_PIN:-}" ]; then SERVO_ARGS+=(--servo-pin "$SERVO_PIN"); fi
exec python disc_distance.py run \
  --diameter 0.237 --focal-length 320 --width 640 --height 480 --fps 15 --camera-quality 70 --buffer-count 3 \
  --process-every 1 --black 95 --contrast 8 --min-area 250 --min-ratio 0.35 \
  --min-circularity 0.45 --max-area-error 0.50 --min-sharpness 4 --smooth 7 \
  --min-command-confidence 55 --turn-gain 0.8 --stop-distance 0.45 \
  --approach-distance 2.0 --source auto --no-self-tune "${SERVO_ARGS[@]}" "$@"
