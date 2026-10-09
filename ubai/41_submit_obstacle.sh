#!/bin/bash
# 벽에 부딪히기 · 경사로 (obstacle) 2잡 (2026-10-09 준비). gate1 에서 돌린다. 35_train_addon.sbatch 그대로.
#
# 왜: 걷기 학습 씬은 발만 충돌해서 몸이 벽을 통과한다 — 걷기 정책은 벽을 한 번도 겪은 적이 없다.
#   scene_obst_train.xml 에서 hp0dy2 를 굴리면 25판 중 10판만 버틴다 (정면·30° 벽 0.222 m/s 에서 3초,
#   경사 10° 이상은 오르막 시작에서 전부 넘어짐). SIM_NOTES 2026-10-09.
#
# | 잡            | 경사 범위 | 시작            | 출력                 |
# |---------------|-----------|-----------------|----------------------|
# | duck-obst     | 3~15°     | hp0dy2 300M     | checkpoints_obst     |
# | duck-obst-x   | 3~25°     | hp0dy2 300M     | checkpoints_obst_x   |
# 공통: 벽 35% / 경사 35% / 없음 30%, 명령 쪽 0.25~0.8 m, 비스듬 ±70°. 관측이 걷기와 같아서(101/212)
# hp0dy2 에서 이어받는다 (이어받기 1003978 PASS). 걷기 설정은 hp0dy2 와 같다. 각 150M.
#
#   DRY=1 bash ~/ubai/41_submit_obstacle.sh     # 확인까지
#   bash ~/ubai/41_submit_obstacle.sh           # 제출까지
set -euo pipefail

REPO="${REPO:-$HOME/Open_Duck_Playground}"
P="$REPO/playground/open_duck_mini_v2"
SB="$HOME/ubai/35_train_addon.sbatch"
WARM="checkpoints_hp0dy2/2026_09_28_102045_300482560"

unset LIN_VEL_X LIN_VEL_Y CMD_AXIS_ZERO HEAD_POS_W ALIVE_W HEAD_ACTION_SCALE SEED \
      FR_RAND_LO FR_RAND_HI OUT STEPS ADDON SKATE_IMIT_W CARRY_OBJECT CARRY_W CARRY_SIGMA \
      CARRY_PUSH_MAX CARRY_MASS_LO CARRY_MASS_HI CARRY_ACTOR_OBS WARM_ TASK RESTORE RESTORE_FROM \
      OBST_P_WALL OBST_P_RAMP OBST_DIST_LO OBST_DIST_HI OBST_WALL_ANG OBST_SLOPE_LO OBST_SLOPE_HI \
      REF_FRACTION STANDUP_ACTION

echo "=== 1) 파일 ==="
for f in addons.py obstacle.py addon_runner.py xmls/open_duck_mini_v2_obstacle.xml xmls/scene_obst_train.xml; do
  [ -f "$P/$f" ] || { echo "!! 없음 $f"; exit 1; }
  echo "  ok   $f"
done
grep -q '"obstacle"' "$P/addon_runner.py" || { echo "!! addon_runner.py 에 obstacle 이 없다"; exit 1; }
grep -q 'RESTORE_FROM' "$SB" || { echo "!! $SB 가 이어받기 판이 아니다"; exit 1; }
[ -d "$REPO/$WARM" ] || { echo "!! $WARM 없음"; exit 1; }

echo "=== 2) 환경 만들기 (로그인 노드 CPU) ==="
cd "$REPO"
C="LIN_VEL_Y=0.2 HEAD_POS_W=0.0 FR_RAND_LO=1.40 FR_RAND_HI=1.90"
for hi in 15 25; do
  # shellcheck disable=SC2086
  out=$(env $C OBST_SLOPE_HI=$hi JAX_PLATFORMS=cpu TF_CPP_MIN_LOG_LEVEL=2 .venv/bin/python -c "
import sys; sys.path.insert(0, '.')
from playground.open_duck_mini_v2 import obstacle as t
env = t.Obstacle(t.default_config())
print('obs', {k: v[0] for k, v in env.observation_size.items()}, 'act', env.action_size)
" 2>&1) && rc=0 || rc=$?
  printf '%s\n' "$out" | grep -E '^\[obstacle\]|^obs' || true
  [ "$rc" -eq 0 ] || { echo "!! 환경 생성 실패 (exit $rc)"; printf '%s\n' "$out" | tail -20; exit 1; }
  { printf '%s\n' "$out" | grep -q "'state': 101" && printf '%s\n' "$out" | grep -q "'privileged_state': 212"; } \
    || { echo "!! 관측이 101/212 가 아니다 — hp0dy2 를 못 이어받는다"; exit 1; }
done

for d in checkpoints_obst checkpoints_obst_x; do
  [ -z "$(ls -A "$REPO/$d" 2>/dev/null)" ] || { echo "!! $d 가 비어 있지 않다"; exit 1; }
done

if [ "${DRY:-0}" = "1" ]; then
  echo "=== DRY=1: 여기까지 ==="
  exit 0
fi

echo "=== 3) 제출 ==="
E="STEPS=150000000,LIN_VEL_Y=0.2,HEAD_POS_W=0.0,FR_RAND_LO=1.40,FR_RAND_HI=1.90,ADDON=obstacle,RESTORE=$WARM"
J1=$(sbatch --parsable --job-name=duck-obst   --time=06:00:00 --export="ALL,OUT=checkpoints_obst,OBST_SLOPE_HI=15,$E" "$SB")
J2=$(sbatch --parsable --job-name=duck-obst-x --time=06:00:00 --export="ALL,OUT=checkpoints_obst_x,OBST_SLOPE_HI=25,$E" "$SB")
echo "$J1 duck-obst / $J2 duck-obst-x"
echo "$(date '+%F %T') obstacle $J1 $J2" >> "$HOME/ubai/addon_jobs.txt"
squeue -u "$USER"
