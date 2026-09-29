r"""머리 명령을 얼마나 따라가는지 잰다 (뷰어 없음, CPU). `ubai/head_probe2.py` 를 이은 것.

09-06 표(head_pos 스윕)와 같은 방식이다: 같은 씬, 같은 구간(정지 / 전진 / 머리 yaw +1.5 /
머리 yaw -1.5, 5초씩), 구간 앞 1초는 버리고 평균·진폭을 낸다. 스윙 = (+1.5 평균) - (-1.5 평균).
머리는 정책에 맡긴다 (`direct_head=False`). 제어는 뷰어와 같은 `MjInfer.control_step()`.

**정책마다 학습 조건을 맞춘다** — 토크 상한, 명령 범위, 그리고 머리 행동 크기.
`head_as` 는 머리 4축만 action_scale 1.5 로 학습됐다 (나머지는 0.25). 뷰어는 스칼라 0.25 를
곱하므로 그대로 굴리면 머리를 학습 때의 1/6 만 움직인다.

    .venv\Scripts\python.exe eval_head.py
"""

import argparse
import contextlib
import io
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
SCENE = "playground/open_duck_mini_v2/xmls/scene_flat_terrain_backlash_collision.xml"  # 09-06 과 같다
HEAD = {"neck_pitch": 5, "head_pitch": 6, "head_yaw": 7, "head_roll": 8}
FALLEN = 0.5

# (이름, 파일, ref_range, 학습 토크, 머리 action_scale, 메모)
POLICIES = [
    ("hw8 (alive 20)", "hw8_2026_09_06_151227_300482560.onnx", True, 3.23, 0.25, "head_pos -8, 09-06"),
    ("alive10", "alive10_2026_09_08_191904_300482560.onnx", True, 3.23, 0.25, "head_pos -8, 09-08"),
    ("alive5", "alive5_2026_09_08_192133_300482560.onnx", True, 3.23, 0.25, "head_pos -8, 09-08"),
    ("alive2", "alive2_2026_09_08_204049_300482560.onnx", True, 3.23, 0.25, "head_pos -8, 09-08"),
    ("fr186", "fr186_2026_09_12_174638_300482560.onnx", True, 1.86, 0.25, "head_pos -8, 1.86 고정"),
    ("frr", "frr_2026_09_12_201448_300482560.onnx", True, 1.86, 0.25, "head_pos -8, U(1.40,1.90)"),
    ("head_as", "head_as_2026_09_12_215216_300482560.onnx", True, 1.86, 1.5, "frr + 머리 action_scale 1.5"),
]

# (시작 제어스텝, 명령 7개) — head_probe2.py 와 같다. 마지막 구간은 200스텝.
PHASES = [
    (0, [0.0, 0, 0, 0, 0, 0.0, 0]),     # 제자리, 머리 명령 없음
    (250, [0.15, 0, 0, 0, 0, 0.0, 0]),  # 전진, 머리 명령 없음
    (500, [0.0, 0, 0, 0, 0, 1.5, 0]),   # 제자리, head_yaw +1.5
    (750, [0.0, 0, 0, 0, 0, -1.5, 0]),  # 제자리, head_yaw -1.5
]
END = 950
SKIP = 50


def yaw_of(q):
    w, x, y, z = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def probe(path, ref_range, fr, head_as):
    with contextlib.redirect_stdout(io.StringIO()):
        m = MjInfer(SCENE, REFERENCE, path, standing=False, ref_range=ref_range)
        m.model.actuator_forcerange[:] = np.array([-fr, fr])
        m.full_reset()
    scale = np.full(len(m.default_actuator), 0.25)
    scale[5:9] = head_as
    m.action_scale = scale
    m.direct_head, m.heading_hold = False, False
    qs, ups, yaws = [], [], []
    yaw_prev = yaw_of(m.get_floating_base_qpos(m.data.qpos)[3:7])
    yaw_acc = 0.0
    with contextlib.redirect_stdout(io.StringIO()):
        for k in range(END):
            for start, cmd in PHASES:
                if k == start:
                    m.commands = list(cmd)
            m.control_step()
            for _ in range(m.decimation):
                mujoco.mj_step(m.model, m.data)
            qs.append(m.get_actuator_joints_qpos(m.data.qpos).copy())
            ups.append(float(m.get_gravity(m.data)[-1]))
            y = yaw_of(m.get_floating_base_qpos(m.data.qpos)[3:7])
            yaw_acc += (y - yaw_prev + np.pi) % (2 * np.pi) - np.pi
            yaw_prev = y
            yaws.append(np.degrees(yaw_acc))
    return np.array(qs), np.array(ups), np.array(yaws)


def phase(qs, k):
    start = PHASES[k][0]
    end = PHASES[k + 1][0] if k + 1 < len(PHASES) else END
    return qs[start + SKIP:end]


def main():
    ap = argparse.ArgumentParser(description="머리 명령 추종 측정")
    ap.add_argument("--forcerange", type=float, default=None,
                    help="토크 상한을 전부 이 값으로 (안 주면 정책마다 학습값)")
    args = ap.parse_args()

    print(f"{os.path.basename(SCENE)} · 구간 5초씩, 앞 1초 버림 · 머리는 정책이 구동 · 단위 rad")
    print(f"  {'정책':<16}{'토크':>6}{'머리as':>7}{'정지 yaw':>10}{'+1.5':>8}{'-1.5':>8}{'스윙':>7}"
          f"{'전진 중 pitch/yaw 진폭':>24}{'머리 돌리는 동안 몸통':>22}{'최저up':>8}  비고")
    for name, f, rr, fr, head_as, memo in POLICIES:
        fr = args.forcerange if args.forcerange is not None else fr
        qs, ups, yaws = probe(os.path.join(ROOT, "from_ubai", f), rr, fr, head_as)
        yi = HEAD["head_yaw"]
        rest, fwd, plus, minus = (phase(qs, k) for k in range(4))
        swing = plus[:, yi].mean() - minus[:, yi].mean()
        amp = lambda q, j: q[:, j].max() - q[:, j].min()
        # 머리를 돌리라고 한 10초 동안 몸통이 얼마나 돌았나 (머리 대신 몸으로 돌리는지)
        body = yaws[END - 1] - yaws[PHASES[2][0]]
        print(f"  {name:<16}{fr:6.2f}{head_as:7.2f}{rest[:, yi].mean():+10.3f}{plus[:, yi].mean():+8.3f}"
              f"{minus[:, yi].mean():+8.3f}{swing:7.3f}"
              f"{amp(fwd, HEAD['head_pitch']):>13.3f} / {amp(fwd, yi):.3f}"
              f"{body:>18.1f}°{ups.min():8.3f}  {memo}"
              f"{'  넘어짐' if ups.min() < FALLEN else ''}", flush=True)


if __name__ == "__main__":
    main()
