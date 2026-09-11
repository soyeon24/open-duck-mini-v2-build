#!/bin/bash
#SBATCH --job-name=duck-head
#SBATCH --partition=gpu5,gpu4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# ── 걷기 + 머리 추종 정책 학습 ────────────────────────────────────
# 배경: 원본 joystick.py 에는 cost_head_pos 가 정의만 되어 있고 보상에 등록되지
#       않아, 학습된 정책이 머리 명령(neck_pitch/head_pitch/head_yaw/head_roll)을
#       완전히 무시했다. 실측 결과 head_yaw 를 +-1.5 로 줘도 실제 관절은 -0.23 에서
#       꿈쩍도 안 했다. (BEST_WALK_ONNX_2 커뮤니티 정책도 동일)
# 수정: joystick.py 에 head_pos=-1.0 등록 + stand_still 은 ignore_head=True
#       rewards.py 의 cost_head_pos 에 only_when_moving 플래그 추가(정지 중에도 추종)
#
# 파티션: gpu3(A6000 Ada)는 2026-09-04 기준 여유 0장 / 대기 56잡이라 제외.
#         gpu5(A6000, 여유 16/24) + gpu4(A6000, 여유 22/116) 를 함께 요청한다.
#         A6000 은 3090 과 같은 GA102 계열이라 속도는 비슷하고 VRAM 만 2배(48G).
#         num_envs=8192 는 24G 로도 충분하므로 이득은 속도가 아니라 대기시간이다.
#
# 웜스타트: 300M 걷기 체크포인트에서 이어받는다. 관측(101)/행동(14) 구조가
#           동일해서 그대로 복원된다. 걷기는 이미 되니 머리만 배우면 된다.
#           ※ brax 는 Adam 상태를 저장하지 않으므로 초반에 잠시 흔들릴 수 있다.
set -uo pipefail

REPO="${REPO:-$HOME/Open_Duck_Playground}"
OUT="${OUT:-checkpoints_head}"
TASK="${TASK:-flat_terrain_backlash}"
STEPS="${STEPS:-150000000}"
WARM="${WARM:-$REPO/checkpoints/2026_08_31_144425_300482560}"

mkdir -p "$REPO/$OUT" "$REPO/logs"
module purge 2>/dev/null || true
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export TF_CPP_MIN_LOG_LEVEL=2

cd "$REPO"
echo "=== node $(hostname) / job $SLURM_JOB_ID / part $SLURM_JOB_PARTITION / start $(date) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

ARGS=(--task "$TASK" --num_timesteps "$STEPS" --output_dir "$OUT")

# 이 출력 폴더에 이미 진행분이 있으면 그쪽을 우선 이어받고, 없으면 웜스타트
RESUME=$(find "$REPO/$OUT" -mindepth 1 -maxdepth 1 -type d 2>/dev/null \
         | grep -E '_[0-9]+$' | awk -F_ '{print $NF"\t"$0}' | sort -n | tail -1 | cut -f2)
if [ -n "$RESUME" ]; then
  echo "[resume] 진행분 이어받기: $RESUME"
  ARGS+=(--restore_checkpoint_path "$RESUME")
elif [ -d "$WARM" ]; then
  echo "[warm] 걷기 정책에서 웜스타트: $WARM"
  ARGS+=(--restore_checkpoint_path "$WARM")
else
  echo "[warm] 웜스타트 대상 없음 → 처음부터"
fi

.venv/bin/python playground/open_duck_mini_v2/runner.py "${ARGS[@]}"
RC=$?
echo "=== end $(date) / exit $RC ==="
exit $RC
