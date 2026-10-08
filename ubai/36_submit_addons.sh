#!/bin/bash
# 머리 위 물건(carry) 1잡 + 스케이트(skate) 2잡 (2026-10-08 준비). gate1 에서 돌린다.
#
# | 잡            | 과제  | 설정                         | 역할                                 |
# |---------------|-------|------------------------------|--------------------------------------|
# | duck-carry    | carry | 5 cm 상자, 질량 U(0.02,0.15) | 본 설정                              |
# | duck-skate    | skate | SKATE_IMIT_W=1.0             | 걷기 흉내를 유지한 채 스케이트       |
# | duck-skate-i0 | skate | SKATE_IMIT_W=0.0             | 흉내 없이 속도만 (밀고 미끄러지기?)  |
# 공통은 hp0dy2 와 같다: LIN_VEL_Y=0.2, HEAD_POS_W=0, 토크 U(1.40,1.90), 300M, 처음부터.
#
# 순서:
#   1) 새 파일(addons/carry/skate/addon_runner + 씬 xml 5개)이 서버에 있는지 — 없으면 scp 부터
#   2) 로그인 노드 CPU 로 환경을 만들어 관측 크기·설정을 찍어 본다 (학습 아님, 1~2분)
#   3) 출력 폴더가 비었는지 확인하고 제출
#
#   DRY=1 bash ~/ubai/36_submit_addons.sh     # 2) 까지만
#   bash ~/ubai/36_submit_addons.sh           # 제출까지
set -euo pipefail

REPO="${REPO:-$HOME/Open_Duck_Playground}"
P="$REPO/playground/open_duck_mini_v2"

# --export=ALL 이라 셸에 남은 값이 다른 잡으로 샌다.
unset LIN_VEL_X LIN_VEL_Y CMD_AXIS_ZERO HEAD_POS_W ALIVE_W HEAD_ACTION_SCALE SEED \
      FR_RAND_LO FR_RAND_HI OUT STEPS ADDON SKATE_IMIT_W CARRY_OBJECT CARRY_W CARRY_SIGMA \
      CARRY_PUSH_MAX CARRY_MASS_LO CARRY_MASS_HI CARRY_ACTOR_OBS

echo "=== 1) 파일 ==="
miss=0
for f in addons.py carry.py skate.py addon_runner.py \
         xmls/open_duck_mini_v2_carry.xml xmls/scene_carry_box.xml xmls/scene_carry_ball.xml \
         xmls/open_duck_mini_v2_skate.xml xmls/scene_skate.xml; do
  if [ -f "$P/$f" ]; then echo "  ok   $f"; else echo "  없음 $f"; miss=1; fi
done
[ "$miss" = 0 ] || { echo "!! 윈도우에서 scp 부터 (ubai/README 의 addon 절)"; exit 1; }

echo "=== 2) 환경 만들기 (로그인 노드 CPU) ==="
cd "$REPO"
C="LIN_VEL_Y=0.2 HEAD_POS_W=0.0 FR_RAND_LO=1.40 FR_RAND_HI=1.90"
for spec in "carry" "skate SKATE_IMIT_W=1.0" "skate SKATE_IMIT_W=0.0"; do
  set -- $spec
  env_name=$1; shift
  echo "--- [$env_name $*]"
  # shellcheck disable=SC2086
  out=$(env $C "$@" JAX_PLATFORMS=cpu TF_CPP_MIN_LOG_LEVEL=2 .venv/bin/python -c "
import sys; sys.path.insert(0, '.')
from playground.open_duck_mini_v2 import $env_name as t
env = getattr(t, '$env_name'.capitalize())(t.default_config())
print('obs', {k: v[0] for k, v in env.observation_size.items()}, 'act', env.action_size)
" 2>&1) && rc=0 || rc=$?
  printf '%s\n' "$out" | grep -E '^\[(carry|skate|addons|joystick)\]|^obs|Error|error' || true
  [ "$rc" -eq 0 ] || { echo "!! 환경 생성 실패 (exit $rc). 제출 안 함"; printf '%s\n' "$out" | tail -20; exit 1; }
done

OUTS="checkpoints_carry checkpoints_skate checkpoints_skate_i0"
for d in $OUTS; do
  if [ -n "$(ls -A "$REPO/$d" 2>/dev/null)" ]; then
    echo "!! $d 가 비어 있지 않다. 옮기거나 지우고 다시"; exit 1
  fi
done

if [ "${DRY:-0}" = "1" ]; then
  echo "=== DRY=1: 여기까지. 제출은 DRY 없이 다시 ==="
  exit 0
fi

echo "=== 3) 제출 ==="
E="STEPS=300000000,LIN_VEL_Y=0.2,HEAD_POS_W=0.0,FR_RAND_LO=1.40,FR_RAND_HI=1.90"
sub() {
  sbatch --parsable --job-name="$1" --export="ALL,OUT=$2,$3,$E" "$HOME/ubai/35_train_addon.sbatch"
}
J1=$(sub duck-carry    checkpoints_carry    "ADDON=carry")
J2=$(sub duck-skate    checkpoints_skate    "ADDON=skate,SKATE_IMIT_W=1.0")
J3=$(sub duck-skate-i0 checkpoints_skate_i0 "ADDON=skate,SKATE_IMIT_W=0.0")
echo "$J1 duck-carry / $J2 duck-skate / $J3 duck-skate-i0"
echo "$(date '+%F %T') $J1 $J2 $J3" >> "$HOME/ubai/addon_jobs.txt"
squeue -u "$USER"
echo
echo "시작하면 (1~2분) 로그 첫머리에서 설정이 먹었는지 본다:"
echo "  cd $REPO && grep -h '^\[carry\]\|^\[skate\]\|^\[addon\]\|^Observation\|^STEP' logs/duck-carry-$J1.out logs/duck-skate-$J2.out | cut -c1-160"
