r"""초파리 뇌가 오리를 조종한다.

    머리 카메라 → 발목 밴드 방위 → 좌/우 LC9 (시각 투사 뉴런) 포아송 자극
      → FlyWire 783 전뇌 LIF (13.9만 뉴런, Shiu et al. 2024) 20 ms
      → 하행 뉴런 발화율 → 걷기 정책의 속도 명령 (dx, dyaw)

LC9 를 고른 이유: 전진 하행 뉴런 P9(DNp09)의 주 입력이고, P9 는 수컷이 암컷을 쫓을 때 필요하다
(Bidaye et al. 2020). 커넥톰에서도 왼쪽 LC9 → 왼쪽 DNp09 만 켜진다 (flybrain/probe.py 표).

배선 (flybrain/probe.py 로 정했다. 손으로 정한 건 게인과 아래 두 줄뿐이다):
    전진  = KX · (DNp09_L + DNp09_R) − KB · (MDN 평균)
    회전  = KY · ((DNp09_L − DNp09_R) + (DNa02_L − DNa02_R))   (왼쪽 +, 요 명령과 같은 부호)
    DNa01 은 안 쓴다. 이 모델에서는 자극 반대쪽에서 켜져 조향과 부호가 반대다.
뇌 밖에 둔 것: 밴드가 안 보이면 자극 0 (뇌가 조용해져 선다), 가까우면 자극을 줄인다
(LC9 같은 작은 물체 검출기는 물체가 시야를 덮을 만큼 커지면 덜 반응한다 — 거칠게 흉내).

뇌는 실시간보다 ~7배 느리다. 뷰어는 그만큼 느리게 돈다.

    .venv\Scripts\python.exe fly_pilot.py                    # 뷰어 (느림)
    .venv\Scripts\python.exe fly_pilot.py --seconds 20 --no_view   # 채점만
    .venv\Scripts\python.exe fly_pilot.py --person 2.0 -0.8 --start_yaw 40
"""
import argparse
import os
import sys
import time

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "flybrain"))
os.chdir(os.path.join(ROOT, "Open_Duck_Playground"))
sys.path.insert(0, os.getcwd())

import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402

import band_tracker  # noqa: E402
from lif import Brain  # noqa: E402
from playground.open_duck_mini_v2.mujoco_infer import MjInfer, resolve_policy  # noqa: E402

REF = "playground/open_duck_mini_v2/data/polynomial_coefficients.pkl"
SCENE = "playground/open_duck_mini_v2/xmls/scene_person.xml"

LC9_HZ = 100.0        # 표적이 한쪽 눈에만 있을 때 그쪽 LC9 자극 세기
SIDE_DEG = 8.0        # 좌/우 나눔의 폭. 방위 ±8° 면 73% / 27%
NEAR_M, FAR_M = 0.45, 0.9   # 이보다 가까우면 자극 0, 멀면 다 줌
WIN_MS = 100.0        # 발화율 창
KX, KY, KB = 0.0035, 0.012, 0.004


class FlyPilot:
    def __init__(self, m, brain):
        self.m, self.B = m, brain
        self.lc9 = {s: brain.find("LC9", s) for s in ("left", "right")}
        self.dn = {(d, s): brain.find(d, s) for d in ("DNp09", "DNa02", "DNa01", "MDN")
                   for s in ("left", "right")}
        self.hist = []
        self.n_win = max(1, int(round(WIN_MS / (m.sim_dt * m.decimation * 1000))))
        self.last = {}

    def rate(self, key):
        counts = np.sum([h[key] for h in self.hist], axis=0)
        return float(np.mean(counts)) * 1000.0 / (len(self.hist) * self.m.sim_dt * self.m.decimation * 1000)

    def step(self):
        m = self.m
        img = m.render_head()
        res = band_tracker.track(img, float(m.model.cam_fovy[m.follow_cam_id]))
        if res is None:
            l = r = 0.0
            bearing = dist = float("nan")
        else:
            bearing, dist = res["bearing_deg"], res["distance_m"]
            near = np.clip((dist - NEAR_M) / (FAR_M - NEAR_M), 0.0, 1.0)
            pl = 1.0 / (1.0 + np.exp(-bearing / (SIDE_DEG / np.log(3))))
            l, r = LC9_HZ * near * pl, LC9_HZ * near * (1 - pl)
        self.B.set_stim([(self.lc9["left"], l), (self.lc9["right"], r)])
        ctrl_ms = m.sim_dt * m.decimation * 1000
        c = self.B.run(ctrl_ms)
        self.hist.append({k: c[v] for k, v in self.dn.items()})
        self.hist = self.hist[-self.n_win:]
        R = {k: self.rate(k) for k in self.dn}
        fwd = KX * (R[("DNp09", "left")] + R[("DNp09", "right")]) \
            - KB * 0.5 * (R[("MDN", "left")] + R[("MDN", "right")])
        yaw = KY * ((R[("DNp09", "left")] - R[("DNp09", "right")])
                    + (R[("DNa02", "left")] - R[("DNa02", "right")]))
        m.commands[0] = float(np.clip(fwd, m.COMMANDS_RANGE_X[0], m.COMMANDS_RANGE_X[1]))
        m.commands[1] = 0.0
        m.commands[2] = float(np.clip(yaw, -1.0, 1.0))
        self.last = dict(bearing=bearing, dist=dist, lc9=(l, r), R=R)
        m.control_step()


def build(args):
    onnx, rr, dy, fr = resolve_policy(args.onnx_model_path)
    m = MjInfer(SCENE, REF, onnx, False, rr, dy)
    m.model.actuator_forcerange[:] = np.array([-fr, fr])
    m.full_reset()
    m.direct_head = False
    m.heading_hold = False
    m.place([0.0, 0.0], args.start_yaw, args.person)
    m.render_head()          # 렌더러를 뷰어보다 먼저 (mujoco_infer.run 주석 참고)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--onnx_model_path", default=None)
    ap.add_argument("--person", type=float, nargs=2, default=[2.0, 0.6])
    ap.add_argument("--start_yaw", type=float, default=0.0)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--no_view", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    m = build(args)
    pilot = FlyPilot(m, Brain(seed=args.seed))
    ctrl_dt = m.sim_dt * m.decimation
    n = int(args.seconds / ctrl_dt)
    p = np.array(args.person)
    log = []

    def tick(k):
        pilot.step()
        for _ in range(m.decimation):
            mujoco.mj_step(m.model, m.data)
        base = m.get_floating_base_qpos(m.data.qpos)
        d_true = float(np.linalg.norm(base[:2] - p))
        log.append((k * ctrl_dt, d_true, pilot.last["bearing"], m.commands[0], m.commands[2],
                    *pilot.last["lc9"],
                    *(pilot.last["R"][(dn, s)] for dn in ("DNp09", "DNa02", "MDN")
                      for s in ("left", "right"))))
        if k % 25 == 0:
            L = log[-1]
            print(f"{L[0]:5.1f}s 거리 {L[1]:.2f} m  방위 {L[2]:+6.1f}°  LC9 {L[5]:3.0f}/{L[6]:3.0f} Hz"
                  f"  DNp09 {L[7]:3.0f}/{L[8]:3.0f}  DNa02 {L[9]:3.0f}/{L[10]:3.0f}  MDN {L[11]:2.0f}/{L[12]:2.0f}"
                  f"  → dx {L[3]:+.3f} yaw {L[4]:+.2f}", flush=True)

    t0 = time.time()
    if args.no_view:
        for k in range(n):
            tick(k)
    else:
        with mujoco.viewer.launch_passive(m.model, m.data, show_left_ui=False,
                                          show_right_ui=False) as v:
            for k in range(n):
                if not v.is_running():
                    break
                tick(k)
                v.sync()
    L = np.array(log)
    seen = np.isfinite(L[:, 2])
    print(f"\n== {L[-1, 0]:.0f}초 (벽시계 {time.time() - t0:.0f}s): 거리 {L[0, 1]:.2f} → {L[-1, 1]:.2f} m"
          f" (최소 {L[:, 1].min():.2f}) · 밴드 보인 비율 {seen.mean():.0%}"
          f" · |방위| 중앙값 {np.nanmedian(np.abs(L[:, 2])):.1f}°")
    out = os.path.join(ROOT, "head_cam_out", "fly_pilot_log.npy")
    np.save(out, L)
    print(f"로그 {out}")


if __name__ == "__main__":
    main()
