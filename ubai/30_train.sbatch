#!/bin/bash
#SBATCH --job-name=duck-rl
#SBATCH --partition=gpu1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# ── Open Duck Mini v2 : PPO 학습 (UBAI) ──────────────────────────
# 사용법 (게이트 노드, $HOME/Open_Duck_Playground 안에서):
#   스모크테스트 : SEGMENT=5000000 TOTAL=5000000 CHAIN=0 sbatch $HOME/ubai/30_train.sbatch
#   본 학습      : sbatch --export=ALL,CHAIN=1 $HOME/ubai/30_train.sbatch
#
# 확인됨(2026-08-31): 파티션은 MaxTime=UNLIMITED 이지만 계정 association MaxWall=2-00:00:00.
# 즉 잡 1개당 최대 48시간. 48h 안에 안 끝나면 CHAIN 으로 이어붙일 것.
# 후속 잡은 최신 체크포인트에서 자동으로 이어받는다.
set -uo pipefail

REPO="${REPO:-$HOME/Open_Duck_Playground}"
SELF="${SELF:-$HOME/ubai/30_train.sbatch}"
CKPT="$REPO/checkpoints"
PROGRESS="$CKPT/.progress"
DONE="$CKPT/.done"

TASK="${TASK:-flat_terrain_backlash}"
SEGMENT="${SEGMENT:-300000000}"   # 잡 1개가 돌릴 스텝 수 (walltime 무제한이라 한 번에 다 돌림)
TOTAL="${TOTAL:-300000000}"       # 누적 목표 스텝 수
CHAIN="${CHAIN:-0}"               # 뒤에 더 붙일 잡 개수

mkdir -p "$CKPT" "$REPO/logs"

if [ -f "$DONE" ]; then
  echo "[chain] $DONE 존재 → 목표 달성. 종료."
  exit 0
fi

# ── 후속 잡 예약: 이 잡이 어떻게 끝나든(afterany) 이어서 실행 ──
NEXT=""
if [ "$CHAIN" -gt 0 ]; then
  NEXT=$(sbatch --parsable --dependency=afterany:"$SLURM_JOB_ID" \
    --export=ALL,CHAIN=$((CHAIN-1)),SEGMENT="$SEGMENT",TOTAL="$TOTAL",TASK="$TASK",REPO="$REPO",SELF="$SELF" \
    "$SELF") || NEXT=""
  echo "[chain] 후속 잡 예약됨: ${NEXT:-실패} (남은 체인 $((CHAIN-1)))"
fi

# ── 환경 ──
module purge 2>/dev/null || true   # module 의 CUDA 11.6 이 jax cuda12 wheel 과 충돌하므로 로드하지 않음
export JAX_PLATFORMS=cuda          # GPU 못 잡으면 CPU 로 조용히 넘어가지 않고 즉시 실패
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export TF_CPP_MIN_LOG_LEVEL=2
# export MUJOCO_GL=egl             # 렌더링 관련 에러가 나면 주석 해제

cd "$REPO"
echo "=== node $(hostname) / job $SLURM_JOB_ID / start $(date) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# ── 최신 체크포인트 탐색 (디렉토리명이 ..._<step> 형식) ──
RESUME=$(find "$CKPT" -mindepth 1 -maxdepth 1 -type d 2>/dev/null \
         | grep -E '_[0-9]+$' | awk -F_ '{print $NF"\t"$0}' | sort -n | tail -1 | cut -f2)
if [ -n "$RESUME" ]; then
  echo "[resume] 이어받기: $RESUME"
  set -- --restore_checkpoint_path "$RESUME"
else
  echo "[resume] 체크포인트 없음 → 처음부터"
  set --
fi

ACCUM=$(cat "$PROGRESS" 2>/dev/null || echo 0)
echo "[progress] 누적 ${ACCUM} / 목표 ${TOTAL} · 이번 세그먼트 ${SEGMENT}"

.venv/bin/python playground/open_duck_mini_v2/runner.py \
  --task "$TASK" \
  --num_timesteps "$SEGMENT" \
  "$@"
RC=$?

echo "=== end $(date) / exit $RC ==="

if [ "$RC" -eq 0 ]; then
  ACCUM=$((ACCUM + SEGMENT))
  echo "$ACCUM" > "$PROGRESS"
  echo "[progress] 누적 갱신: $ACCUM"
  if [ "$ACCUM" -ge "$TOTAL" ]; then
    touch "$DONE"
    [ -n "$NEXT" ] && scancel "$NEXT" && echo "[chain] 목표 달성 → 후속 잡 $NEXT 취소"
  fi
fi
exit $RC
