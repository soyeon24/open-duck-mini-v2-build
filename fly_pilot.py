r"""초파리 뇌가 오리를 조종한다.

    머리 카메라 → 발목 밴드 방위 → 좌/우 LC9 (시각 투사 뉴런) 포아송 자극
      → FlyWire 783 전뇌 LIF (13.9만 뉴런, Shiu et al. 2024) 20 ms
      → 하행 뉴런 발화율 → 걷기 정책의 속도 명령 (dx, dyaw)

LC9 를 고른 이유: 전진 하행 뉴런 P9(DNp09)의 주 입력이고, P9 는 수컷이 암컷을 쫓을 때 필요하다
(Bidaye et al. 2020). 커넥톰에서도 왼쪽 LC9 → 왼쪽 DNp09 만 켜진다 (flybrain/probe.py 표).

배선 (flybrain/probe.py 로 정했다):
    전진  = KX · (DNp09_L + DNp09_R) − KB · (MDN 평균)
    회전  = KY · ((DNp09_L − DNp09_R) + (DNa02_L − DNa02_R))   (왼쪽 +, 요 명령과 같은 부호)
    DNa01 은 안 쓴다. 이 모델에서는 자극 반대쪽에서 켜져 조향과 부호가 반대다.

좌우 LC9 는 서로를 누른다 (2026-10-07 격자): 150/0 Hz 면 DNp09 74/0 인데 100/100 이면 22/4,
50/50 이면 5/1. 그래서 정면 표적에 자극을 반반 나누면 전진이 꺼진다 (첫 판: 20초에 0.24 m).
좌우 나눔을 날카롭게 해서(SIDE_DEG 3°) 늘 한쪽이 이기게 한다 — 파리처럼 좌우로 꺾으며 쫓는다.
가까워지면 자극이 줄어 좌우가 다시 서로를 누르고, 전진이 꺼져 사람 앞에서 선다 (0.68~0.70 m).
LC9 150 Hz: 2.09 → 0.70 m (0.70 까지 14초) · 200 Hz: → 0.68 m (10초). 200 을 기본값으로 둔다.

뇌 밖에 둔 것: 밴드가 안 보이면 자극 0 (뇌가 조용해져 선다), 가까우면 자극을 줄인다
(LC9 같은 작은 물체 검출기는 물체가 시야를 덮을 만큼 커지면 덜 반응한다 — 거칠게 흉내).

뇌는 실시간보다 ~7배(렌더 포함 ~20배) 느리다. 그래서 먼저 헤드리스로 굴려 궤적을 저장하고,
뷰어는 그걸 실시간으로 다시 튼다 (화면 위에 LC9 입력과 DN 발화율이 같이 뜬다).

    .venv\Scripts\python.exe fly_pilot.py                       # 굴리고 → 뷰어로 재생
    .venv\Scripts\python.exe fly_pilot.py --no_view --tag a     # 굴리기만 (채점)
    .venv\Scripts\python.exe fly_pilot.py --replay head_cam_out\fly_pilot_a.npz
"""
import argparse
import os
import sys
import time

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = os.path.dirname(os.path.abspath(__file__))
CWD0 = os.getcwd()          # --replay 상대경로는 실행한 자리 기준 (아래에서 폴더를 옮긴다)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "flybrain"))
os.chdir(os.path.join(ROOT, "Open_Duck_Playground"))
sys.path.insert(0, os.getcwd())

import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402

REF = "playground/open_duck_mini_v2/data/polynomial_coefficients.pkl"
SCENE = "playground/open_duck_mini_v2/xmls/scene_person.xml"
OUT = os.path.join(ROOT, "head_cam_out")
DNS = ("DNp09", "DNa02", "MDN")


class FlyPilot:
    def __init__(self, m, brain, a):
        self.m, self.B, self.a = m, brain, a
        self.lc9 = {s: brain.find("LC9", s) for s in ("left", "right")}
        self.dn = {(d, s): brain.find(d, s) for d in DNS for s in ("left", "right")}
        self.ctrl_ms = m.sim_dt * m.decimation * 1000
        self.alpha = 1.0 - np.exp(-self.ctrl_ms / a.tau_ms)
        self.R = {k: 0.0 for k in self.dn}      # 지수 평활한 발화율 (Hz)
        self.last = {}

    def step(self, band_tracker):
        m, a = self.m, self.a
        img = m.render_head()
        res = band_tracker.track(img, float(m.model.cam_fovy[m.follow_cam_id]))
        head_yaw = float(np.degrees(m.data.qpos[m.model.joint("head_yaw").qposadr[0]]))
        if res is None:
            l = r = 0.0
            bearing = dist = float("nan")
            # 안 보이면 머리는 마지막으로 돌린 자리에 둔다 (사람이 사라진 쪽을 계속 본다).
        else:
            # 몸통 기준 방향 = 화면 속 방위 + 머리 각도. 머리를 돌려도 뇌가 받는 좌우는 몸 기준이다.
            bearing = res["bearing_deg"] + (head_yaw if a.head else 0.0)
            dist = res["distance_m"]
            if a.dist_from == "ground":
                # 밴드를 바닥 높이로 투영한 위치에서 잰다 — 기존 추종(follow_step)과 같은 거리.
                # 밴드 폭 거리는 옆에서 보면 두 발목이 겹쳐 2배로 부푼다. 그 거리로는 0.4 m 에서도
                # 자극이 그대로라 오리가 파고들어 밴드가 카메라 아래로 빠졌다 (2026-10-07, 0.1 m/s 판들).
                pxy = m._person_xy(res, img)
                if pxy is not None:
                    dist = float(np.hypot(*(np.asarray(pxy) - m.data.cam_xpos[m.follow_cam_id][:2])))
            near = np.clip((dist - a.near) / (a.far - a.near), 0.0, 1.0)
            pl = 1.0 / (1.0 + np.exp(-bearing / (a.side_deg / np.log(3))))
            l, r = a.lc9_hz * near * pl, a.lc9_hz * near * (1 - pl)
            if a.head:
                # 머리는 뇌 밖에서 사람을 화면 가운데 둔다 — 기존 추종(mujoco_infer)의 머리와 같은
                # 방식·같은 한계(±25°). 다른 점은 사람만 본다는 것 (기존은 갈 방향과 반반, 바닥스캔 때문).
                m.direct_head = True
                m.commands[5] = float(np.radians(np.clip(bearing, -a.head_max, a.head_max)))
        self.B.set_stim([(self.lc9["left"], l), (self.lc9["right"], r)])
        c = self.B.run(self.ctrl_ms)
        for k, ids in self.dn.items():
            hz = float(np.mean(c[ids])) * 1000.0 / self.ctrl_ms
            self.R[k] += self.alpha * (hz - self.R[k])
        R = self.R
        fwd = a.kx * (R[("DNp09", "left")] + R[("DNp09", "right")]) \
            - a.kb * 0.5 * (R[("MDN", "left")] + R[("MDN", "right")])
        yaw = a.ky * ((R[("DNp09", "left")] - R[("DNp09", "right")])
                      + (R[("DNa02", "left")] - R[("DNa02", "right")]))
        m.commands[0] = float(np.clip(fwd, m.COMMANDS_RANGE_X[0], m.COMMANDS_RANGE_X[1]))
        m.commands[1] = 0.0
        m.commands[2] = float(np.clip(yaw, -1.0, 1.0))
        self.last = dict(bearing=bearing, dist=dist, lc9=(l, r), R=dict(R), head=head_yaw)
        m.control_step()


def run(a):
    import band_tracker
    from lif import Brain
    from playground.open_duck_mini_v2.mujoco_infer import MjInfer, resolve_policy

    onnx, rr, dy, fr = resolve_policy(a.onnx_model_path)
    m = MjInfer(SCENE, REF, onnx, False, rr, dy)
    m.model.actuator_forcerange[:] = np.array([-fr, fr])
    m.full_reset()
    m.direct_head = False
    m.heading_hold = False
    m.place([0.0, 0.0], a.start_yaw, a.person)
    m.render_head()
    pilot = FlyPilot(m, Brain(seed=a.seed), a)
    ctrl_dt = m.sim_dt * m.decimation
    p = np.array(a.person)
    log, qpos = [], []
    t0 = time.time()
    for k in range(int(a.seconds / ctrl_dt)):
        pilot.step(band_tracker)
        for _ in range(m.decimation):
            mujoco.mj_step(m.model, m.data)
        base = m.get_floating_base_qpos(m.data.qpos)
        L = pilot.last
        log.append((k * ctrl_dt, float(np.linalg.norm(base[:2] - p)), L["bearing"],
                    m.commands[0], m.commands[2], *L["lc9"],
                    *(L["R"][(d, s)] for d in DNS for s in ("left", "right"))))
        qpos.append(m.data.qpos.copy())
        if k % 50 == 0:
            g = log[-1]
            print(f"{g[0]:5.1f}s 거리 {g[1]:.2f} m  방위 {g[2]:+6.1f}°  LC9 {g[5]:3.0f}/{g[6]:3.0f}"
                  f"  DNp09 {g[7]:3.0f}/{g[8]:3.0f}  DNa02 {g[9]:3.0f}/{g[10]:3.0f}"
                  f"  → dx {g[3]:+.3f} yaw {g[4]:+.2f}", flush=True)
    L = np.array(log)
    near = np.nonzero(L[:, 1] < a.near + 0.15)[0]
    print(f"\n== [{a.tag}] {L[-1, 0]:.0f}초 (벽시계 {time.time() - t0:.0f}s): 거리 {L[0, 1]:.2f} → "
          f"{L[-1, 1]:.2f} m (최소 {L[:, 1].min():.2f})"
          f" · {a.near + 0.15:.2f} m 안 도착 {'%.1f초' % L[near[0], 0] if len(near) else '못 함'}"
          f" · 밴드 보인 비율 {np.isfinite(L[:, 2]).mean():.0%}"
          f" · 평균 dx {L[:, 3].mean():.3f}")
    path = os.path.join(OUT, f"fly_pilot_{a.tag}.npz")
    np.savez(path, log=L, qpos=np.array(qpos), person=p, ctrl_dt=ctrl_dt, args=str(vars(a)))
    print(f"궤적 {path}")
    return path


def replay(path, loop=True):
    z = np.load(path)
    L, Q, p, dt = z["log"], z["qpos"], z["person"], float(z["ctrl_dt"])
    P = z["person_traj"] if "person_traj" in z else None
    from playground.open_duck_mini_v2 import base
    from etils import epath
    model = mujoco.MjModel.from_xml_string(epath.Path(SCENE).read_text(), assets=base.get_assets())
    data = mujoco.MjData(model)
    data.mocap_pos[0] = [p[0], p[1], 0.0]
    print(f">>> 재생 {os.path.basename(path)} — 실시간. 창을 닫으면 끝")
    with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as v:
        v.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        v.cam.trackbodyid = model.body("base").id if model.nbody > 1 else 0
        v.cam.distance, v.cam.elevation, v.cam.azimuth = 1.6, -25, 200
        while v.is_running():
            for k in range(len(Q)):
                if not v.is_running():
                    break
                t0 = time.time()
                data.qpos[:] = Q[k]
                if P is not None:     # 걷는 사람 대본 (eval_fly_moving.py)
                    data.mocap_pos[0] = [P[k, 0], P[k, 1], 0.0]
                    h = P[k, 2] / 2.0
                    data.mocap_quat[0] = [np.cos(h), 0.0, 0.0, np.sin(h)]
                mujoco.mj_forward(model, data)
                g = L[k]
                b = "안 보임" if not np.isfinite(g[2]) else f"{g[2]:+.0f}°"
                v.set_texts([(mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                              "time\nperson\nbearing (body)\nLC9 in  L / R\nDNp09    L / R\n"
                              "DNa02    L / R\ncommand dx / yaw" + ("\nhead yaw" if len(g) > 13 else ""),
                              f"{g[0]:.1f} s\n{g[1]:.2f} m\n{b}\n{g[5]:.0f} / {g[6]:.0f} Hz\n"
                              f"{g[7]:.0f} / {g[8]:.0f} Hz\n{g[9]:.0f} / {g[10]:.0f} Hz\n"
                              f"{g[3]:+.3f} / {g[4]:+.2f}"
                              + (f"\n{g[13]:+.0f}°" if len(g) > 13 else ""))])
                v.sync()
                time.sleep(max(0.0, dt - (time.time() - t0)))
            if not loop:
                break
            time.sleep(1.0)


def make_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--onnx_model_path", default=None)
    ap.add_argument("--person", type=float, nargs=2, default=[2.0, 0.6])
    ap.add_argument("--start_yaw", type=float, default=0.0)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--no_view", action="store_true", help="굴리기만 하고 재생은 안 한다")
    ap.add_argument("--replay", default=None, help="저장한 궤적(npz)을 뷰어로 재생만")
    g = ap.add_argument_group("배선")
    g.add_argument("--lc9_hz", type=float, default=200.0, help="한쪽 눈에만 보일 때 그쪽 LC9 자극")
    g.add_argument("--side_deg", type=float, default=3.0, help="방위 ±이 값에서 75%%/25%% 로 나뉜다")
    g.add_argument("--near", type=float, default=0.45, help="이보다 가까우면 자극 0 (m)")
    g.add_argument("--far", type=float, default=0.9, help="이보다 멀면 자극을 다 준다 (m)")
    g.add_argument("--tau_ms", type=float, default=150.0, help="발화율 지수 평활 시정수")
    g.add_argument("--kx", type=float, default=0.0035, help="전진 게인 (m/s per Hz)")
    g.add_argument("--ky", type=float, default=0.006, help="회전 게인 (rad/s per Hz)")
    g.add_argument("--kb", type=float, default=0.004, help="MDN 후진 게인")
    g.add_argument("--no_head", dest="head", action="store_false",
                   help="머리를 안 돌린다 (2026-10-07 첫 비교 2/10 이 이 조건)")
    g.add_argument("--head_max", type=float, default=25.0, help="머리 요 한계 (도). 기존 추종과 같다")
    g.add_argument("--dist_from", choices=["ground", "width"], default="ground",
                   help="거리: 밴드 바닥 투영(기존 추종과 같음) / 밴드 폭 (옆에서 2배로 부푼다)")
    return ap


def main():
    ap = make_parser()
    a = ap.parse_args()
    if a.replay:
        replay(os.path.join(CWD0, a.replay))
        return
    path = run(a)
    if not a.no_view:
        replay(path)


if __name__ == "__main__":
    main()
