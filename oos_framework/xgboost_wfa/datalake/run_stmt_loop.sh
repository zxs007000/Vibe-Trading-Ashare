#!/bin/bash
# 自重启 wrapper：驱动进程退出后自动拉起，直到 BUILD_DONE 出现。
cd /workspace/stockworm_build
LOG=/workspace/stockworm_build/restart.log
echo "$(date) wrapper start" >> "$LOG"
while [ ! -f /workspace/stocklake/BUILD_DONE ]; do
  python3.11 drive_stmt.py >> "$LOG" 2>&1
  rc=$?
  echo "$(date) driver exited rc=$rc, restarting in 5s" >> "$LOG"
  sleep 5
done
echo "$(date) BUILD_DONE present, wrapper exit" >> "$LOG"
