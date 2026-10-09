#!/bin/bash
# obstacle 2차 (2026-10-09 준비). gate1. 1차(1004181/2)는 벽은 늘었지만(obst_x 정면 0.222 도 버팀, 벽 8/9)
# 경사 10° 이상은 16판 전부 그대로 넘어졌다. 토크(1.86→5.0)를 올려도 같은 자리에서 넘어져서 힘 문제가 아니다.
# 1차 학습에서는 명령이 옆·뒤·회전으로도 나와 경사를 실제로 밟는 판이 적었던 것으로 본다.
#
# 바꾼 것: OBST_HOLD=1 (벽·경사 판은 앞으로 0.08~0.222, 옆·회전 0, 에피소드 내내 고정),
#          경사 50% / 벽 30% / 없음 20%, 장애물 0.15~0.5 m 앞. obst_x 151M 에서 이어받는다.
# | 잡           | 경사     |
# |--------------|----------|
# | duck-obst2   | 3~15°    |
# | duck-obst2e  | 3~10°    |   (쉬운 쪽부터 되는지)
#   DRY=1 bash ~/ubai/42_submit_obstacle2.sh / bash ~/ubai/42_submit_obstacle2.sh
set -euo pipefail
REPO="${REPO:-$HOME/Open_Duck_Playground}"
SB="$HOME/ubai/35_train_addon.sbatch"
WARM=$(ls -1dt "$REPO"/checkpoints_obst_x/*_[0-9]*/ | head -1); WARM="${WARM%/}"
echo "warm: $WARM"
grep -q 'OBST_HOLD' "$REPO/playground/open_duck_mini_v2/obstacle.py" || { echo "!! obstacle.py 가 2차 판이 아니다"; exit 1; }
unset OUT STEPS ADDON RESTORE RESTORE_FROM OBST_P_WALL OBST_P_RAMP OBST_DIST_LO OBST_DIST_HI OBST_WALL_ANG \
      OBST_SLOPE_LO OBST_SLOPE_HI OBST_HOLD OBST_VX_LO LIN_VEL_X LIN_VEL_Y HEAD_POS_W FR_RAND_LO FR_RAND_HI \
      ALIVE_W SEED CARRY_OBJECT CARRY_ACTOR_OBS STANDUP_ACTION REF_FRACTION
cd "$REPO"
C="LIN_VEL_Y=0.2 HEAD_POS_W=0.0 FR_RAND_LO=1.40 FR_RAND_HI=1.90 OBST_HOLD=1 OBST_P_WALL=0.3 OBST_P_RAMP=0.5 OBST_DIST_LO=0.15 OBST_DIST_HI=0.5"
for hi in 15 10; do
  # shellcheck disable=SC2086
  env $C OBST_SLOPE_HI=$hi JAX_PLATFORMS=cpu TF_CPP_MIN_LOG_LEVEL=2 .venv/bin/python -c "
import sys; sys.path.insert(0, '.')
from playground.open_duck_mini_v2 import obstacle as t
c = t.default_config(); env = t.Obstacle(c)
print('hold', c.obst_hold, 'obs', {k: v[0] for k, v in env.observation_size.items()})
" 2>&1 | grep -E '^\[obstacle\]|^hold|Error'
done
for d in checkpoints_obst2 checkpoints_obst2e; do
  [ -z "$(ls -A "$REPO/$d" 2>/dev/null)" ] || { echo "!! $d 비어 있지 않다"; exit 1; }
done
[ "${DRY:-0}" = "1" ] && { echo "=== DRY ==="; exit 0; }
E="STEPS=150000000,LIN_VEL_Y=0.2,HEAD_POS_W=0.0,FR_RAND_LO=1.40,FR_RAND_HI=1.90,ADDON=obstacle,RESTORE=$WARM,OBST_HOLD=1,OBST_P_WALL=0.3,OBST_P_RAMP=0.5,OBST_DIST_LO=0.15,OBST_DIST_HI=0.5"
J1=$(sbatch --parsable --job-name=duck-obst2  --time=06:00:00 --export="ALL,OUT=checkpoints_obst2,OBST_SLOPE_HI=15,$E" "$SB")
J2=$(sbatch --parsable --job-name=duck-obst2e --time=06:00:00 --export="ALL,OUT=checkpoints_obst2e,OBST_SLOPE_HI=10,$E" "$SB")
echo "$J1 duck-obst2 / $J2 duck-obst2e"
echo "$(date '+%F %T') obstacle2 $J1 $J2" >> "$HOME/ubai/addon_jobs.txt"
squeue -u "$USER"
