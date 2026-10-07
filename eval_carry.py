r"""지금 걷기 정책에 머리 위 상자를 올리면 몇 초 버티나 (carry 학습 전 기준선).

carry 학습 환경과 같은 판정이다: 쟁반 기준 물건이 쟁반 아래로 가거나 가운데서 8 cm 넘게
벗어나면 떨어진 것. 머리는 정책 출력 그대로 둔다 (걷기 정책은 머리를 home 근처에 둔다).

    .venv\Scripts\python.exe eval_carry.py              # 표
    .venv\Scripts\python.exe eval_carry.py --view fwd   # 그 명령으로 MuJoCo 뷰어
"""
import argparse
import os
import sys
import time

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.join(ROOT, "Open_Duck_Playground"))
sys.path.insert(0, os.getcwd())

import mujoco  # noqa: E402
from playground.open_duck_mini_v2.mujoco_infer import MjInfer, resolve_policy  # noqa: E402

REF = "playground/open_duck_mini_v2/data/polynomial_coefficients.pkl"
CMDS = {  # 이름: (dx, dy, dyaw)
    "stand": (0.0, 0.0, 0.0),
    "fwd": (0.15, 0.0, 0.0),
    "fwd_max": (0.222, 0.0, 0.0),
    "back": (-0.1, 0.0, 0.0),
    "turn": (0.0, 0.0, 0.8),
    "side": (0.0, 0.15, 0.0),
}
TRAY_HALF = 0.06


def obj_rel(m):
    tb, ob = m.model.body("carry_tray").id, m.model.body("carry_object").id
    R = m.data.xmat[tb].reshape(3, 3)
    return R.T @ (m.data.xpos[ob] - m.data.xpos[tb])


def dropped(rel):
    return rel[2] < 0.0 or np.hypot(rel[0], rel[1]) > TRAY_HALF + 0.02


def make(args, obj):
    onnx, rr, dy, fr = resolve_policy(args.onnx_model_path)
    m = MjInfer(f"playground/open_duck_mini_v2/xmls/scene_carry_{obj}.xml", REF, onnx, False, rr, dy)
    m.model.actuator_forcerange[:] = np.array([-fr, fr])
    m.full_reset()
    m.direct_head = False
    return m


def run(args, obj, cmd, seconds):
    m = make(args, obj)
    m.commands[:3] = list(CMDS[cmd])
    n = int(seconds / (m.sim_dt * m.decimation))
    worst = 0.0
    for k in range(n):
        m.control_step()
        for _ in range(m.decimation):
            mujoco.mj_step(m.model, m.data)
        rel = obj_rel(m)
        worst = max(worst, float(np.hypot(rel[0], rel[1])))
        if dropped(rel):
            return (k + 1) * m.sim_dt * m.decimation, worst
    return None, worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--onnx_model_path", default=None)
    ap.add_argument("--object", default="box", choices=["box", "ball"])
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--view", default=None, choices=list(CMDS))
    args = ap.parse_args()

    if args.view:
        m = make(args, args.object)
        m.commands[:3] = list(CMDS[args.view])
        print(f">>> {args.object} / 명령 {args.view} {CMDS[args.view]}  (Ctrl+C 로 끝)")
        with mujoco.viewer.launch_passive(m.model, m.data, show_left_ui=False,
                                          show_right_ui=False) as v:
            t_drop = None
            while v.is_running():
                t0 = time.time()
                m.control_step()
                for _ in range(m.decimation):
                    mujoco.mj_step(m.model, m.data)
                if t_drop is None and dropped(obj_rel(m)):
                    t_drop = m.data.time
                    print(f">>> {t_drop:.1f}초에 떨어뜨렸다")
                v.sync()
                time.sleep(max(0.0, m.sim_dt * m.decimation - (time.time() - t0)))
        return

    print(f"| 명령 | 떨어뜨린 시각 ({args.seconds:.0f}초 중) | 가운데서 최대 이탈 |")
    print("|---|---|---|")
    for cmd in CMDS:
        t, worst = run(args, args.object, cmd, args.seconds)
        print(f"| {cmd} {CMDS[cmd]} | {'안 떨어뜨림' if t is None else f'{t:.1f} s'} "
              f"| {worst * 100:.1f} cm |")


if __name__ == "__main__":
    main()
