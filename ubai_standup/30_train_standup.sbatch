#!/bin/bash
#SBATCH --job-name=duck-standup
#SBATCH --partition=gpu1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# ── Open Duck Mini v2 : 기립(standup) PPO 학습 ────────────────────
# ~/ubai/30_train.sbatch (걷기용) 를 기반으로 함. 그 파일은 건드리지 않는다.
#
# 사용법 (게이트 노드에서):
#   스모크 : SEGMENT=2000000 TOTAL=2000000 OUTDIR=checkpoints_standup_smoke \
#            sbatch --export=ALL,SEGMENT,TOTAL,OUTDIR $HOME/ubai_standup/30_train_standup.sbatch
#   본학습 : sbatch --export=ALL,CHAIN=1 $HOME/ubai_standup/30_train_standup.sbatch
#
# 걷기와 다른 점:
#   - runner 가 standup_runner.py (runner.py 를 수정하지 않기 위한 별도 진입점)
#   - scene_standup.xml 사용: 몸통/머리/다리에 바닥 전용 충돌 박스 8개 추가.
#     원본은 충돌 geom 이 발바닥 2개뿐이라 넘어지면 몸이 바닥을 통과해서
#     기립 학습이 원천적으로 불가능하다.
#   - 충돌체가 늘어난 만큼 step/s 는 걷기(약 68,700)보다 낮을 것으로 예상.
set -uo pipefail

REPO="${REPO:-$HOME/Open_Duck_Playground}"
SELF="${SELF:-$HOME/ubai_standup/30_train_standup.sbatch}"
OUTDIR="${OUTDIR:-checkpoints_standup}"
CKPT="$REPO/$OUTDIR"
PROGRESS="$CKPT/.progress"
DONE="$CKPT/.done"

SEGMENT="${SEGMENT:-150000000}"
TOTAL="${TOTAL:-150000000}"
CHAIN="${CHAIN:-0}"

mkdir -p "$CKPT" "$REPO/logs"

if [ -f "$DONE" ]; then
  echo "[chain] $DONE 존재 → 목표 달성. 종료."
  exit 0
fi

NEXT=""
if [ "$CHAIN" -gt 0 ]; then
  NEXT=$(sbatch --parsable --dependency=afterany:"$SLURM_JOB_ID" \
    --export=ALL,CHAIN=$((CHAIN-1)),SEGMENT="$SEGMENT",TOTAL="$TOTAL",OUTDIR="$OUTDIR",REPO="$REPO",SELF="$SELF" \
    "$SELF") || NEXT=""
  echo "[chain] 후속 잡 예약됨: ${NEXT:-실패} (남은 체인 $((CHAIN-1)))"
fi

module purge 2>/dev/null || true
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export TF_CPP_MIN_LOG_LEVEL=2

cd "$REPO"
echo "=== node $(hostname) / job $SLURM_JOB_ID / start $(date) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

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

.venv/bin/python playground/open_duck_mini_v2/standup_runner.py \
  --output_dir "$OUTDIR" \
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
