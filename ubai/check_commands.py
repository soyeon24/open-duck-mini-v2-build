"""joystick.py 가 지금 환경변수로 어떤 명령을 뽑는지 찍는다 (로그인 노드 CPU, 몇 초).

학습이 아니라 명령 난수만 뽑으므로 gate1 에서 돌려도 된다. 스윕 축은 던지기 전에
눈으로 확인한다는 09-28 의 원칙을 스크립트로 박아 둔 것이다.

    cd ~/Open_Duck_Playground
    LIN_VEL_X=0.15 LIN_VEL_Y=0.2 CMD_AXIS_ZERO=0.4 JAX_PLATFORMS=cpu \
        .venv/bin/python ~/ubai/check_commands.py

설정값이 환경변수와 다르거나 분포가 기대와 어긋나면 exit 1.
기대값은 2026-09-29 에 로컬(서버와 같은 jax 0.6.2 / playground 0.0.5)에서 20만 개로 잰 값이다.
"""
import os
import sys

import jax
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.getcwd())
from playground.open_duck_mini_v2 import joystick  # noqa: E402

N = 100_000


class _Cfg:
    def __init__(self, cfg):
        self._config = cfg


def main():
    cfg = joystick.default_config()
    fn = jax.jit(jax.vmap(lambda k: joystick.Joystick.sample_command(_Cfg(cfg), k)))
    c = np.asarray(fn(jax.random.split(jax.random.PRNGKey(0), N)))
    vx, vy, wz = c[:, 0], c[:, 1], c[:, 2]
    zx, zy, zw = np.abs(vx) < 0.02, np.abs(vy) < 0.02, np.abs(wz) < 0.1

    print(f"lin_vel_x {list(cfg.lin_vel_x)}  lin_vel_y {list(cfg.lin_vel_y)}  "
          f"cmd_axis_zero {cfg.cmd_axis_zero}  head_pos_w {cfg.reward_config.scales.head_pos}")
    ok = True
    want_x = os.environ.get("LIN_VEL_X")
    if want_x is not None and list(cfg.lin_vel_x) != [-float(want_x), float(want_x)]:
        print(f"  !! LIN_VEL_X={want_x} 인데 lin_vel_x 가 {list(cfg.lin_vel_x)}")
        ok = False
    want_z = float(os.environ.get("CMD_AXIS_ZERO", "0"))
    if cfg.cmd_axis_zero != want_z:
        print(f"  !! CMD_AXIS_ZERO={want_z} 인데 cmd_axis_zero 가 {cfg.cmd_axis_zero}")
        ok = False

    masked = cfg.cmd_axis_zero > 0
    # (이름, 비율, 마스킹 켰을 때 기대 %, 껐을 때 기대 %)
    rows = [
        ("정지 (셋 다 0)", np.mean((vx == 0) & (vy == 0) & (wz == 0)), 15.8, 10.0),
        ("그냥 후진 (vx<-0.10)", np.mean((vx < -0.10) & zy & zw), 1.6, 0.1),
        ("제자리 게걸음 (|vy|>0.10)", np.mean((np.abs(vy) > 0.10) & zx & zw), 5.8, 0.5),
        ("제자리 회전 (|wz|>0.3)", np.mean((np.abs(wz) > 0.3) & zx & zy), 8.2, 0.8),
    ]
    for name, frac, on, off in rows:
        want = on if masked else off
        print(f"  {name:<24} {100 * frac:6.2f} %   (기대 ~{want})")
        if abs(100 * frac - want) > max(0.5, 0.25 * want):
            print("  !! 기대에서 벗어났다")
            ok = False
    print("OK" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
