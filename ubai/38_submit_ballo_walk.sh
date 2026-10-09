#!/bin/bash
# 공을 보는 carry(ballo) 를 걷게 만들기 2잡 (2026-10-08 준비). gate1 에서 돌린다. 35_train_addon.sbatch 그대로.
#
# 왜: duck-carry-ballo(1003981) 300M 은 공을 20초 안 떨어뜨리지만 **제자리에 서서** 지킨다
#   (전진 명령 20초 이동 0.01 m. SIM_NOTES 2026-10-08). alive 20 + carry 5 가 속도 추종 2.5 보다 커서,
#   걷다 공을 잃어 에피소드가 끝나느니 서 있는 게 이득이다. 서 있기가 덜 이득이게 둘을 나란히:
#
# | 잡                 | 바꾼 것        | 묻는 것                                           |
# |--------------------|----------------|---------------------------------------------------|
# | duck-ballo-a5      | ALIVE_W 20 → 5 | 살아 있기 몫을 줄이면(속도 추종 비중↑) 걷나       |
# | duck-ballo-c1      | CARRY_W 5 → 1  | 가운데 유지 몫을 줄이면 걷나 (공은 덜 가운데여도) |
# 속도 추종 가중치(tracking_lin_vel 2.5)는 joystick.py 에 박혀 있고 환경변수가 없어서, 걷기 잡과 같이 쓰는
# 코드를 안 건드리려고 상대 비중으로 올렸다. alive 5 는 09-08 alive5 걷기 잡이 300M 까지 간 값 (걷기 품질은 미측정).
# 나머지는 ballo 와 같다: CARRY_OBJECT=ball, CARRY_ACTOR_OBS=1, LIN_VEL_Y=0.2, HEAD_POS_W=0,
# 토크 U(1.40,1.90), 300M, 처음부터 (관측 104 라 걷기 체크포인트를 못 이어받는다).
#
#   DRY=1 bash ~/ubai/38_submit_ballo_walk.sh     # 환경 생성 확인까지
#   bash ~/ubai/38_submit_ballo_walk.sh           # 제출까지
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
C="LIN_VEL_Y=0.2 HEAD_POS_W=0.0 FR_RAND_LO=1.40 FR_RAND_HI=1.90 CARRY_OBJECT=ball CARRY_ACTOR_OBS=1"
for v in "ALIVE_W=5.0" "CARRY_W=1.0"; do
  echo "--- [$v]"
  # shellcheck disable=SC2086
  out=$(env $C $v JAX_PLATFORMS=cpu TF_CPP_MIN_LOG_LEVEL=2 .venv/bin/python -c "
import sys; sys.path.insert(0, '.')
from playground.open_duck_mini_v2 import carry as t
cfg = t.default_config()
s = cfg.reward_config.scales
print('scales alive', s.alive, 'carry', s.carry, 'tracking_lin_vel', s.tracking_lin_vel, 'tracking_ang_vel', s.tracking_ang_vel)
env = t.Carry(cfg)
print('obs', {k: v[0] for k, v in env.observation_size.items()}, 'act', env.action_size)
" 2>&1) && rc=0 || rc=$?
  printf '%s\n' "$out" | grep -E '^\[carry\]|^obs|^scales' || true
  [ "$rc" -eq 0 ] || { echo "!! 환경 생성 실패 (exit $rc). 제출 안 함"; printf '%s\n' "$out" | tail -20; exit 1; }
  printf '%s\n' "$out" | grep -q "'state': 104" || { echo "!! 정책 관측이 104 가 아니다"; exit 1; }
done

for d in checkpoints_ballo_a5 checkpoints_ballo_c1; do
  if [ -n "$(ls -A "$REPO/$d" 2>/dev/null)" ]; then
    echo "!! $d 가 비어 있지 않다. 옮기거나 지우고 다시"; exit 1
  fi
done

if [ "${DRY:-0}" = "1" ]; then
  echo "=== DRY=1: 여기까지. 제출은 DRY 없이 다시 ==="
  exit 0
fi

echo "=== 3) 제출 ==="
E="STEPS=300000000,LIN_VEL_Y=0.2,HEAD_POS_W=0.0,FR_RAND_LO=1.40,FR_RAND_HI=1.90,ADDON=carry,CARRY_OBJECT=ball,CARRY_ACTOR_OBS=1"
sub() {
  sbatch --parsable --job-name="$1" --time=06:00:00 --export="ALL,OUT=$2,$3,$E" "$HOME/ubai/35_train_addon.sbatch"
}
J1=$(sub duck-ballo-a5 checkpoints_ballo_a5 "ALIVE_W=5.0")
J2=$(sub duck-ballo-c1 checkpoints_ballo_c1 "CARRY_W=1.0")
echo "$J1 duck-ballo-a5 / $J2 duck-ballo-c1"
echo "$(date '+%F %T') ballo-walk $J1 $J2" >> "$HOME/ubai/addon_jobs.txt"
squeue -u "$USER"
echo
echo "시작하면 설정이 먹었는지:"
echo "  cd $REPO && grep -h '^ALIVE_W\|^CARRY_\|^\[carry\]\|^STEP' logs/duck-ballo-a5-$J1.out logs/duck-ballo-c1-$J2.out | cut -c1-160"
