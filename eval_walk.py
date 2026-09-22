"""걷기 정책이 명령대로 가는지 채점한다 (뷰어 없음, CPU).

뷰어로 보면 "왼쪽으로 좀 도는 것 같다" 까지밖에 안 나온다. 얼마나 도는지, 명령으로
되돌릴 수 있는 크기인지를 모르면 다음 학습을 어떻게 고칠지도 모른다.

제어 루프는 `mujoco_infer.py` 의 `run()` 과 같은 순서다 (50Hz, action_scale 0.25,
서보 속도 제한 5.24 rad/s). 뷰어만 뺐다. 여기서 루프를 다르게 만들면 뷰어에서 보는
것과 다른 결과가 나와서 측정 자체가 쓸모없어진다.

**정책마다 학습 당시 명령 범위가 다르므로 `ref_range` 를 각자 학습값으로 준다.**
안 그러면 범위 밖 입력 때문에 생긴 실패를 정책 품질로 오독한다 (2026-09-04 의
`lin_vel_y` 정합 전후로 갈린다. SIM_NOTES "명령 범위가 레퍼런스 모션과 안 맞았다").

머리는 정책에 맡긴다 (`direct_head=False`). 학습 조건이 그렇고, 머리를 명령으로
0 에 붙들면 그것만으로 요 드리프트가 생겨 측정이 오염된다.

    .venv\\Scripts\\python.exe eval_walk.py
    .venv\\Scripts\\python.exe eval_walk.py -o from_ubai\\fr186_....onnx --ref_range
"""

import argparse
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(ROOT, "Open_Duck_Playground")
os.chdir(REPO)
sys.path.insert(0, REPO)

import mujoco  # noqa: E402

from playground.open_duck_mini_v2.mujoco_infer import MjInfer  # noqa: E402

REFERENCE = "playground/open_duck_mini_v2/data/polynomial_coefficients.pkl"
SCENE = "playground/open_duck_mini_v2/xmls/scene_flat_terrain_backlash.xml"

FALLEN = 0.5  # up 이 이 아래로 내려가면 넘어지는 중으로 본다

# 기본 비교군. (이름, 파일, ref_range) — ref_range 는 그 정책이 학습된 명령 범위다.
DEFAULT_POLICIES = [
    ("08-31 완성본  dy+-0.2", "2026_08_31_144425_300482560.onnx", False),
    ("head 910949   dy+-0.2", "head_2026_09_04_140643_300482560.onnx", False),
    ("hw8           dy+-0.111", "hw8_2026_09_06_151227_300482560.onnx", True),
    ("fr186         dy+-0.111", "fr186_2026_09_12_174638_300482560.onnx", True),
]

CMDS = [
    ("정지", (0, 0, 0)),
    ("전진", (1, 0, 0)),
    ("좌 게걸음", (0, 1, 0)),
    ("우 게걸음", (0, -1, 0)),
    ("대각 전진+좌", (1, 1, 0)),
    ("대각 전진+우", (1, -1, 0)),
    ("제자리 좌회전", (0, 0, 1)),
    ("전진+좌회전", (1, 0, 1)),
]


def yaw_of(quat):
    w, x, y, z = quat
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def rollout(m, cmd, seconds):
    """한 조건을 굴리고 (전진, 횡변위, 누적 요, 최저 up, 넘어진 시각) 을 돌려준다."""
    m.full_reset()
    m.direct_head = False

    cx, cy, ct = cmd
    m.commands[0] = cx * (m.COMMANDS_RANGE_X[1] if cx > 0 else -m.COMMANDS_RANGE_X[0])
    m.commands[1] = cy * (m.COMMANDS_RANGE_Y[1] if cy > 0 else -m.COMMANDS_RANGE_Y[0])
    m.commands[2] = ct * m.COMMANDS_RANGE_THETA[1]

    base = m.get_floating_base_qpos(m.data.qpos)
    p0, y0 = base[:3].copy(), yaw_of(base[3:7].copy())

    n = int(seconds / (m.sim_dt * m.decimation))
    min_up, fell_at = 1.0, None
    # 요는 매 스텝 차분을 누적한다. 시작과 끝 자세만 빼면 180도를 넘는 회전이
    # 반대 부호로 접혀서, 좌회전 592도가 -127도(우회전)로 읽힌다.
    yaw_acc, yaw_prev = 0.0, y0

    for k in range(n):
        m.imitation_i = (m.imitation_i + m.phase_frequency_factor) % m.PRM.nb_steps_in_period
        ph = m.imitation_i / m.PRM.nb_steps_in_period * 2 * np.pi
        m.imitation_phase = np.array([np.cos(ph), np.sin(ph)])

        action = m.policy.infer(m.get_obs(m.data, m.commands))
        m.last_last_last_action = m.last_last_action.copy()
        m.last_last_action = m.last_action.copy()
        m.last_action = action.copy()

        m.motor_targets = m.default_actuator + action * m.action_scale
        lim = m.max_motor_velocity * (m.sim_dt * m.decimation)
        m.motor_targets = np.clip(
            m.motor_targets, m.prev_motor_targets - lim, m.prev_motor_targets + lim
        )
        m.prev_motor_targets = m.motor_targets.copy()
        m.data.ctrl = m.motor_targets.copy()

        for _ in range(m.decimation):
            mujoco.mj_step(m.model, m.data)

        yaw_now = yaw_of(m.get_floating_base_qpos(m.data.qpos)[3:7])
        yaw_acc += (yaw_now - yaw_prev + np.pi) % (2 * np.pi) - np.pi
        yaw_prev = yaw_now

        up = float(m.get_gravity(m.data)[-1])
        min_up = min(min_up, up)
        if fell_at is None and up < FALLEN:
            fell_at = k * m.sim_dt * m.decimation

    base = m.get_floating_base_qpos(m.data.qpos)
    d = base[:3] - p0
    # 출발 방향 기준으로 되돌려, 앞/옆이 각각 얼마인지 본다.
    fwd = d[0] * np.cos(y0) + d[1] * np.sin(y0)
    lat = -d[0] * np.sin(y0) + d[1] * np.cos(y0)
    return fwd, lat, np.rad2deg(yaw_acc), min_up, fell_at


def main():
    p = argparse.ArgumentParser(description="걷기 정책 명령 추종 측정")
    p.add_argument("-o", "--onnx", nargs="+",
                   help="정책 파일. 안 주면 DEFAULT_POLICIES 비교군을 돈다")
    p.add_argument("--ref_range", action="store_true",
                   help="-o 로 준 정책이 2026-09-04 이후(dy +-0.111)면 붙일 것")
    p.add_argument("--seconds", type=float, default=12.0)
    p.add_argument("--scene", default=SCENE)
    args = p.parse_args()

    if args.onnx:
        policies = [(os.path.basename(o), o, args.ref_range) for o in args.onnx]
    else:
        policies = [(n, os.path.join(ROOT, "from_ubai", f), r)
                    for n, f, r in DEFAULT_POLICIES]

    print(f"{os.path.basename(args.scene)} · {args.seconds:.0f}초/조건 · 머리는 정책이 구동")
    for label, path, rr in policies:
        path = path if os.path.isabs(path) else os.path.join(ROOT, path)
        m = MjInfer(args.scene, REFERENCE, path, standing=False, ref_range=rr)
        print("=" * 76)
        print(f"{label}   (ref_range={rr})")
        print(f"  {'명령':<14} {'전진cm':>8} {'횡cm':>8} {'누적 요°':>10} {'최저 up':>9}  비고")
        for cname, cmd in CMDS:
            fwd, lat, dyaw, mu, fell = rollout(m, cmd, args.seconds)
            note = f"넘어짐 {fell:.1f}s" if fell is not None else ""
            print(f"  {cname:<14} {fwd * 100:8.1f} {lat * 100:8.1f} "
                  f"{dyaw:10.1f} {mu:9.3f}  {note}")


if __name__ == "__main__":
    main()
