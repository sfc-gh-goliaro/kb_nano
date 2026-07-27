#!/bin/bash
# Start the watchdog in its own session so no process-group kill can take it.
# It has died twice mid-session: manual cleanup and the scheduler's own straggler
# kills both operate on process groups, and anything sharing the watchdog's group
# goes with them. setsid puts it out of reach; check-ins verify it is alive.
ROOT=/home/yak/b200_repro
if pgrep -f "bash $ROOT/watchdog.sh" > /dev/null; then
  echo "watchdog already running: $(pgrep -f "bash $ROOT/watchdog.sh" | tr '\n' ' ')"
  exit 0
fi
setsid nohup bash "$ROOT/watchdog.sh" > /dev/null 2>&1 < /dev/null &
sleep 3
pgrep -af "bash $ROOT/watchdog.sh" || { echo "FAILED to start watchdog"; exit 1; }
