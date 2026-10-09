#!/bin/bash
# 공 carry 커리큘럼: 상자로 걷기를 먼저 배우고 공으로 이어받기 (2026-10-08 준비). gate1 에서 돌린다.
#
# 왜: 공을 보는 carry 를 처음부터 배우면 서서 지키는 쪽으로 굳는다 — ballo(1003981), ballo-a5(1004038),
#   ballo-c1(1004039) 셋 다 20초 이동 8 cm 이하 (SIM_NOTES 2026-10-08). 가중치를 바꿔도 같았다.
#   상자 carry(1003726, 관측 101)는 앞으로 걸었다. 그래서 상자로 먼저 걷게 한 뒤 공으로 옮긴다.
#   이어받기는 1003978 로 PASS. 관측이 같아야 하니 1단계도 CARRY_ACTOR_OBS=1 (관측 104).
#
# | 잡            | 물건 | 시작                         | 출력                  |
# |---------------|------|------------------------------|-----------------------|
# | duck-cur-box  | 상자 | 처음부터                     | checkpoints_cur_box   |
# | duck-cur-ball | 공   | cur_box 의 가장 최근 체크포인트 | checkpoints_cur_ball  |
# cur-ball 은 --dependency=afterok 로 cur-box 가 성공해야 시작한다 (35_train_addon.sbatch 의 RESTORE_FROM).
# 나머지는 ballo 와 같다: LIN_VEL_Y=0.2, HEAD_POS_W=0, 토크 U(1.40,1.90), 각 300M.
#
#   DRY=1 bash ~/ubai/39_submit_ballo_curriculum.sh     # 확인까지
#   bash ~/ubai/39_submit_ballo_curriculum.sh           # 제출까지
set -euo pipefail

REPO="${REPO:-$HOME/Open_Duck_Playground}"
P="$REPO/playground/open_duck_mini_v2"
SB="$HOME/ubai/35_train_addon.sbatch"

# --export=ALL 이라 셸에 남은 값이 다른 잡으로 샌다.
unset LIN_VEL_X LIN_VEL_Y CMD_AXIS_ZERO HEAD_POS_W ALIVE_W HEAD_ACTION_SCALE SEED \
      FR_RAND_LO FR_RAND_HI OUT STEPS ADDON SKATE_IMIT_W CARRY_OBJECT CARRY_W CARRY_SIGMA \
      CARRY_PUSH_MAX CARRY_MASS_LO CARRY_MASS_HI CARRY_ACTOR_OBS WARM TASK RESTORE RESTORE_FROM

echo "=== 1) 파일 ==="
for f in addons.py carry.py addon_runner.py xmls/scene_carry_box.xml xmls/scene_carry_ball.xml; do
  [ -f "$P/$f" ] || { echo "!! 없음 $f"; exit 1; }
  echo "  ok   $f"
done
grep -q 'RESTORE_FROM' "$SB" || { echo "!! $SB 가 이어받기 판이 아니다 (RESTORE_FROM 없음). scp 부터"; exit 1; }
grep -q 'def load_params' "$REPO/playground/common/runner.py" || { echo "!! runner.py 가 이어받기 판이 아니다"; exit 1; }
echo "  ok   35_train_addon.sbatch (RESTORE_FROM) / runner.py (load_params)"

echo "=== 2) 환경 만들기 (로그인 노드 CPU) — 상자·공 관측이 같아야 이어받는다 ==="
cd "$REPO"
C="LIN_VEL_Y=0.2 HEAD_POS_W=0.0 FR_RAND_LO=1.40 FR_RAND_HI=1.90 CARRY_ACTOR_OBS=1"
sizes=()
for o in box ball; do
  # shellcheck disable=SC2086
  out=$(env $C CARRY_OBJECT=$o JAX_PLATFORMS=cpu TF_CPP_MIN_LOG_LEVEL=2 .venv/bin/python -c "
import sys; sys.path.insert(0, '.')
from playground.open_duck_mini_v2 import carry as t
env = t.Carry(t.default_config())
print('obs', {k: v[0] for k, v in env.observation_size.items()}, 'act', env.action_size)
" 2>&1) && rc=0 || rc=$?
  printf '%s\n' "$out" | grep -E '^\[carry\]|^obs' || true
  [ "$rc" -eq 0 ] || { echo "!! 환경 생성 실패 (exit $rc)"; printf '%s\n' "$out" | tail -20; exit 1; }
  sizes+=("$(printf '%s\n' "$out" | grep '^obs')")
done
[ "${sizes[0]}" = "${sizes[1]}" ] || { echo "!! 상자·공 관측 크기가 다르다 — 이어받기 불가"; exit 1; }
printf '%s\n' "${sizes[0]}" | grep -q "'state': 104" || { echo "!! 정책 관측이 104 가 아니다"; exit 1; }

echo "=== 3) RESTORE_FROM 고르기 확인 (지난 ballo 출력으로) ==="
pick=$(ls -1dt "$REPO/checkpoints_carry_ballo"/*_[0-9]*/ 2>/dev/null | head -1)
echo "  checkpoints_carry_ballo -> ${pick%/}"
[ -n "$pick" ] || { echo "!! 체크포인트 폴더 고르기가 안 된다"; exit 1; }

for d in checkpoints_cur_box checkpoints_cur_ball; do
  if [ -n "$(ls -A "$REPO/$d" 2>/dev/null)" ]; then
    echo "!! $d 가 비어 있지 않다. 옮기거나 지우고 다시"; exit 1
  fi
done

if [ "${DRY:-0}" = "1" ]; then
  echo "=== DRY=1: 여기까지. 제출은 DRY 없이 다시 ==="
  exit 0
fi

echo "=== 4) 제출 ==="
E="STEPS=300000000,LIN_VEL_Y=0.2,HEAD_POS_W=0.0,FR_RAND_LO=1.40,FR_RAND_HI=1.90,ADDON=carry,CARRY_ACTOR_OBS=1"
J1=$(sbatch --parsable --job-name=duck-cur-box --time=06:00:00 \
       --export="ALL,OUT=checkpoints_cur_box,CARRY_OBJECT=box,$E" "$SB")
J2=$(sbatch --parsable --job-name=duck-cur-ball --time=06:00:00 \
       --dependency=afterok:$J1 --kill-on-invalid-dep=yes \
       --export="ALL,OUT=checkpoints_cur_ball,CARRY_OBJECT=ball,RESTORE_FROM=checkpoints_cur_box,$E" "$SB")
echo "$J1 duck-cur-box / $J2 duck-cur-ball (afterok:$J1)"
echo "$(date '+%F %T') curriculum $J1 $J2" >> "$HOME/ubai/addon_jobs.txt"
squeue -u "$USER"
echo
echo "cur-ball 이 시작하면 이어받았는지:"
echo "  cd $REPO && grep -h '^\[addon\]\|^\[restore\]\|^STEP: 0 ' logs/duck-cur-ball-$J2.out"
