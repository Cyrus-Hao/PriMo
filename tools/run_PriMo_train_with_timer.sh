#!/usr/bin/env bash
set -Eeuo pipefail

if [[ "${CONDA_DEFAULT_ENV:-}" != "mpsfm" ]]; then
  if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate mpsfm
  else
    echo "[WARN] conda 不可用，无法自动激活 mpsfm 环境。"
  fi
fi

# libgomp requires a positive integer value.
if [[ -z "${OMP_NUM_THREADS:-}" || ! "${OMP_NUM_THREADS}" =~ ^[1-9][0-9]*$ ]]; then
  export OMP_NUM_THREADS=1
fi

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


