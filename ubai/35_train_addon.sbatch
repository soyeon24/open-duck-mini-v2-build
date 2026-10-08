#!/bin/bash
#SBATCH --job-name=duck-addon
#SBATCH --partition=gpu1,gpu4,gpu5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# ── 머리 위 물건(carry) · 스케이트(skate) PPO 학습 (2026-10-08 준비) ─────────────
# 진입점은 addon_runner.py — runner.py 는 걷기 잡과 같이 쓰므로 건드리지 않는다.
# 과제 설정은 환경변수 (carry.py / skate.py 머리말). 처음부터 학습한다: carry 는 critic 관측이
# 걷기보다 9칸 많아서 걷기 체크포인트를 그대로 못 이어받는다.
# 제출은 36_submit_addons.sh 로 (파일·설정 확인 후 3잡).
#
#   ADDON  carry | skate (필수)
#   OUT    출력 폴더 (REPO 아래)
#   STEPS  학습 스텝 (기본 300M)
set -uo pipefail

REPO="${REPO:-$HOME/Open_Duck_Playground}"
ADDON="${ADDON:?ADDON=carry|skate 를 줄 것}"
OUT="${OUT:-checkpoints_$ADDON}"
STEPS="${STEPS:-300000000}"

mkdir -p "$REPO/$OUT" "$REPO/logs"
module purge 2>/dev/null || true
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export TF_CPP_MIN_LOG_LEVEL=2

cd "$REPO"
echo "=== node $(hostname) / job $SLURM_JOB_ID / part $SLURM_JOB_PARTITION / start $(date) ==="
echo "[addon] $ADDON -> $OUT, $STEPS steps"
env | grep -E '^(CARRY_|SKATE_|LIN_VEL_|HEAD_POS_W|FR_RAND_|SEED|ALIVE_W)' | sort
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# -u: 버퍼링하면 살아 있는 잡의 로그가 몇 분씩 비어 죽은 잡과 구별이 안 된다 (30_train_standup 주석)
.venv/bin/python -u playground/open_duck_mini_v2/addon_runner.py \
  --env "$ADDON" --num_timesteps "$STEPS" --output_dir "$OUT"
RC=$?
echo "=== end $(date) / exit $RC ==="
exit $RC
