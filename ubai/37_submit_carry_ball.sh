#!/bin/bash
# 머리 위 공(carry ball) 2잡 (2026-10-08 준비). gate1 에서 돌린다. 35_train_addon.sbatch 를 그대로 쓴다.
#
# 왜: duck-carry(1003726, 5 cm 상자) 300M 은 상자를 6개 명령 전부 20초 안 떨어뜨렸다
#   (eval_carry.py, 학습 조건 dy 0.2 / 토크 1.86. 257M 도 같다). 같은 정책에 공을 올리면
#   서 있기만 해도 0.6초에 떨어진다 (걷기 정책 hp0dy2 도 0.5초). 공은 몸 기울기 1° 에 구르니
#   "덜 흔들리게 걷기" 만으로는 안 되고, 공을 보고 머리로 받쳐야 할 수도 있다. 그래서 둘을 나란히:
#
# | 잡                | 물건 | CARRY_ACTOR_OBS | 묻는 것                                       |
# |-------------------|------|-----------------|-----------------------------------------------|
# | duck-carry-ball   | 공   | 0 (critic 만)   | 공 위치를 모르고도 공을 지키는 걸음이 나오나  |
# | duck-carry-ballo  | 공   | 1 (정책도 봄)   | 쟁반 밑 FSR 로 공 위치를 알면 받칠 수 있나    |
# 나머지는 duck-carry 와 같다: LIN_VEL_Y=0.2, HEAD_POS_W=0, 토크 U(1.40,1.90), 300M, 처음부터
# (ballo 는 정책 관측이 3칸 늘어서 상자 체크포인트를 이어받을 수 없다).
#
#   DRY=1 bash ~/ubai/37_submit_carry_ball.sh     # 환경 생성 확인까지
#   bash ~/ubai/37_submit_carry_ball.sh           # 제출까지
set -euo pipefail

REPO="${REPO:-$HOME/Open_Duck_Playground}"
P="$REPO/playground/open_duck_mini_v2"

# --export=ALL 이라 셸에 남은 값이 다른 잡으로 샌다.
unset LIN_VEL_X LIN_VEL_Y CMD_AXIS_ZERO HEAD_POS_W ALIVE_W HEAD_ACTION_SCALE SEED \
      FR_RAND_LO FR_RAND_HI OUT STEPS ADDON SKATE_IMIT_W CARRY_OBJECT CARRY_W CARRY_SIGMA \
      CARRY_PUSH_MAX CARRY_MASS_LO CARRY_MASS_HI CARRY_ACTOR_OBS WARM TASK

echo "=== 1) 파일 ==="
for f in addons.py carry.py addon_runner.py xmls/open_duck_mini_v2_carry.xml xmls/scene_carry_ball.xml; do
  [ -f "$P/$f" ] || { echo "!! 없음 $f"; exit 1; }
  echo "  ok   $f"
done
[ -f "$HOME/ubai/35_train_addon.sbatch" ] || { echo "!! 35_train_addon.sbatch 없음"; exit 1; }

echo "=== 2) 환경 만들기 (로그인 노드 CPU) ==="
cd "$REPO"
C="LIN_VEL_Y=0.2 HEAD_POS_W=0.0 FR_RAND_LO=1.40 FR_RAND_HI=1.90 CARRY_OBJECT=ball"
for ao in 0 1; do
  echo "--- [ball CARRY_ACTOR_OBS=$ao]"
  # shellcheck disable=SC2086
  out=$(env $C CARRY_ACTOR_OBS=$ao JAX_PLATFORMS=cpu TF_CPP_MIN_LOG_LEVEL=2 .venv/bin/python -c "
import sys; sys.path.insert(0, '.')
from playground.open_duck_mini_v2 import carry as t
env = t.Carry(t.default_config())
print('obs', {k: v[0] for k, v in env.observation_size.items()}, 'act', env.action_size)
" 2>&1) && rc=0 || rc=$?
  printf '%s\n' "$out" | grep -E '^\[(carry|addons)\]|^obs' || true
  [ "$rc" -eq 0 ] || { echo "!! 환경 생성 실패 (exit $rc). 제출 안 함"; printf '%s\n' "$out" | tail -20; exit 1; }
  printf '%s\n' "$out" | grep -q '물건 ball' || { echo "!! 공이 아니다"; exit 1; }
done

for d in checkpoints_carry_ball checkpoints_carry_ballo; do
  if [ -n "$(ls -A "$REPO/$d" 2>/dev/null)" ]; then
    echo "!! $d 가 비어 있지 않다. 옮기거나 지우고 다시"; exit 1
  fi
done

if [ "${DRY:-0}" = "1" ]; then
  echo "=== DRY=1: 여기까지. 제출은 DRY 없이 다시 ==="
  exit 0
fi

echo "=== 3) 제출 ==="
E="STEPS=300000000,LIN_VEL_Y=0.2,HEAD_POS_W=0.0,FR_RAND_LO=1.40,FR_RAND_HI=1.90,ADDON=carry,CARRY_OBJECT=ball"
sub() {
  sbatch --parsable --job-name="$1" --time=06:00:00 --export="ALL,OUT=$2,$3,$E" "$HOME/ubai/35_train_addon.sbatch"
}
J1=$(sub duck-carry-ball  checkpoints_carry_ball  "CARRY_ACTOR_OBS=0")
J2=$(sub duck-carry-ballo checkpoints_carry_ballo "CARRY_ACTOR_OBS=1")
echo "$J1 duck-carry-ball / $J2 duck-carry-ballo"
echo "$(date '+%F %T') ball $J1 $J2" >> "$HOME/ubai/addon_jobs.txt"
squeue -u "$USER"
echo
echo "시작하면 설정이 먹었는지:"
echo "  cd $REPO && grep -h '^\[carry\]\|^Observation\|^STEP' logs/duck-carry-ball-$J1.out logs/duck-carry-ballo-$J2.out | cut -c1-160"
