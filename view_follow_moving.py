r"""걷는 사람 따라가기를 MuJoCo 창으로 본다 (`eval_follow_moving.py` 와 같은 대본·같은 제어).

채점은 `eval_follow_moving.py` 가 하고, 이건 눈으로 보는 용도다. 사람(mocap)을 같은 대본으로
걸리고 뷰어의 F 와 같은 코드(`MjInfer.control_step` -> `follow_step`)로 따라가게 한다. 설정도
뷰어 기본 그대로다. 대본을 차례로 돌린다 — 사람이 멈추고 HOLD 초 지나면 다음 대본.

바닥에 궤적을 찍는다: 파란 점 = 오리, 회색 점 = 사람, 빨간 점 = 오리가 사람을 놓친 순간.

    .venv\Scripts\python.exe view_follow_moving.py                          # 다섯 대본, 0.2 m/s
    .venv\Scripts\python.exe view_follow_moving.py --scenario approach --speed 0.1

창에서 N = 다음 대본으로. 창을 닫으면 끝난다.
"""

import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# 대본(SCENARIOS)과 사람 걸리기(Walker)는 채점과 같은 것을 쓴다. import 하면 저 파일이
# Open_Duck_Playground 로 chdir 하고 경로를 잡아 준다.
import eval_follow_moving as efm  # noqa: E402
import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402

from playground.open_duck_mini_v2.mujoco_infer import MjInfer, resolve_policy  # noqa: E402

HOLD_S = 8.0          # 사람이 멈춘 뒤 이만큼 보고 다음 대본
TRAIL_EVERY_S = 0.2   # 궤적 점 간격


def main():
    ap = argparse.ArgumentParser(description="걷는 사람 따라가기 보기")
    ap.add_argument("--scenario", nargs="+", choices=list(efm.SCENARIOS), default=list(efm.SCENARIOS))
    ap.add_argument("--speed", type=float, default=0.2, help="사람 걷는 속도 m/s (오리는 0.11~0.15)")
    ap.add_argument("-o", "--onnx_model_path", type=str, default=None)
    ap.add_argument("--forcerange", type=float, default=None)
    args = ap.parse_args()

    onnx, rr, lvy, fr = resolve_policy(args.onnx_model_path, True, None, args.forcerange)
    m = MjInfer(efm.SCENE, efm.REFERENCE, onnx, False, rr, lvy)
    if fr is not None:
        m.model.actuator_forcerange[:] = np.array([-fr, fr])
    # 오프스크린 렌더러를 **창보다 먼저** 만든다. 창이 뜬 뒤에 만들면 세그폴트로 죽는다
    # (mujoco_infer.run 과 같은 이유, 2026-09-23).
    m.render_head()
    ctrl_dt = m.sim_dt * m.decimation

    state = {"skip": False}

    def on_key(keycode):
        if keycode == ord("N"):
            state["skip"] = True

    print(f">>> 정책 {os.path.basename(onnx)} · 토크 ±{fr} · 사람 {args.speed:.1f} m/s · "
          f"대본 {', '.join(args.scenario)}  (N = 다음, 창 닫으면 끝)")
    with mujoco.viewer.launch_passive(m.model, m.data, show_left_ui=False, show_right_ui=False,
                                      key_callback=on_key) as v:
        v.cam.lookat[:] = [1.0, 0.2, 0.0]
        v.cam.distance, v.cam.elevation, v.cam.azimuth = 3.6, -55.0, 200.0
        for name in args.scenario:
            if not v.is_running():
                break
            m.full_reset()
            m.direct_head, m.heading_hold = False, False
            walker = efm.Walker(name, args.speed)
            walker.put(m.data)
            mujoco.mj_forward(m.model, m.data)
            m.follow, m.follow_lost, m.target_world_deg, m.avoid_side = True, 0, None, 0
            state["skip"] = False
            trail, lost_n, k, dmin = [], 0, 0, 9.0
            print(f"\n=== {efm.SCENARIOS[name][0]} ({name}) · {args.speed:.1f} m/s")
            while v.is_running() and not state["skip"]:
                t0 = time.time()
                t = k * ctrl_dt
                base = m.get_floating_base_qpos(m.data.qpos)
                walker.step(t, ctrl_dt, base[:2])
                walker.put(m.data)
                m.control_step()
                seen = m.follow_lost == 0
                lost_n += not seen
                for _ in range(m.decimation):
                    mujoco.mj_step(m.model, m.data)
                k += 1
                base = m.get_floating_base_qpos(m.data.qpos)
                d = float(np.hypot(*(walker.pos - base[:2])))
                dmin = min(dmin, d)

                if k % max(1, int(round(TRAIL_EVERY_S / ctrl_dt))) == 0:
                    trail.append((base[0], base[1], seen, walker.pos[0], walker.pos[1]))
                with v.lock():
                    scn = v.user_scn
                    scn.ngeom = 0
                    for (rx, ry, sn, px, py) in trail[-(scn.maxgeom // 2 - 1):]:
                        for (x, y, rgba, r) in ((px, py, (0.35, 0.35, 0.35, 0.9), 0.012),
                                                (rx, ry, (0.15, 0.45, 0.9, 0.9) if sn
                                                 else (0.9, 0.15, 0.1, 1.0), 0.015)):
                            if scn.ngeom >= scn.maxgeom:
                                break
                            mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                                                np.array([r, 0, 0]), np.array([x, y, 0.005]),
                                                np.eye(3).ravel(), np.array(rgba, dtype=np.float32))
                            scn.ngeom += 1
                v.sync()

                if walker.done_t is not None and t >= walker.done_t + HOLD_S:
                    break
                wait = ctrl_dt - (time.time() - t0)
                if wait > 0:
                    time.sleep(wait)
            print(f"    끝 거리 {d:.2f} m · 최소 {dmin:.2f} m · 놓침 {100.0 * lost_n / max(k, 1):.0f}%"
                  f"{' (N 으로 넘김)' if state['skip'] else ''}")
        print("\n>>> 대본 끝. 창을 닫으면 종료")
        while v.is_running():
            time.sleep(0.1)


if __name__ == "__main__":
    main()
