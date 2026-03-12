#!/usr/bin/env bash
set -Eeuo pipefail

start_ts=$(date +%s)
echo "[TIMER] 开始: $(date '+%F %T')"

exit_code=0
"$@" || exit_code=$?

end_ts=$(date +%s)
elapsed=$(( end_ts - start_ts ))
h=$(( elapsed / 3600 ))
m=$(( (elapsed % 3600) / 60 ))
s=$(( elapsed % 60 ))
printf "[TIMER] 结束: %s  总耗时: %02d:%02d:%02d (秒: %d)\n" "$(date '+%F %T')" "$h" "$m" "$s" "$elapsed"

exit "$exit_code"


