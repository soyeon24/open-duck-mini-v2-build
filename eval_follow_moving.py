r"""걷는 사람을 따라가는지 잰다 (뷰어 없음, 빈 바닥).

`eval_follow.py` 는 사람을 세워 둔다. 실제로 따라다닐 사람은 걷는다 — 멀어지고, 앞을
가로지르고, 방향을 틀고, 오리 쪽으로 다가온다. 사람(mocap)을 대본대로 걸리고 뷰어의 F 와
**같은 코드**(`MjInfer.control_step` -> `follow_step`)로 따라가게 한 뒤, 정답 위치로 잰다.
설정도 뷰어 기본 그대로다 (회피·머리 추종·몸통 사람 고정 켬).

오리는 초속 11~15 cm 다. 사람의 보통 걸음(초속 1 m 넘게)은 애초에 못 따라간다. 그래서 사람
속도를 0.1 / 0.2 / 0.3 m/s 로 올려 가며 어디서 끊기는지 본다.

보는 값 (대본 x 속도마다 한 줄):
  놓침      카메라가 밴드를 못 잡은 제어스텝 비율, 가장 길게 놓친 시간
  걷는 중   사람이 걷는 동안 최대 거리
  끝        사람이 멈추고 HOLD 초 뒤, 마지막 3초 평균 거리 (0.55 m 에서 서도록 짜여 있다)
  최소      가장 가까웠던 거리. 사람은 충돌체가 없어 부딪혀도 지나가므로 거리로 본다
  방위      몸통 정면 기준 사람 방위의 절대평균 (정답 위치로)
  우회      회피(A*)가 경유점을 잡은 스텝 비율. **빈 바닥이므로 0 이어야 한다.**
            0 이 아니면 걸어간 사람의 발을 장애물로 기억한 것이다

    .venv\Scripts\python.exe eval_follow_moving.py
    .venv\Scripts\python.exe eval_follow_moving.py --scenario away approach --speeds 0.2
    .venv\Scripts\python.exe eval_follow_moving.py --no_avoid
"""

import argparse
import concurrent.futures as cf
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(ROOT, "Open_Duck_Playground")
sys.path.insert(0, ROOT)
os.chdir(REPO)
sys.path.insert(0, REPO)

import mujoco  # noqa: E402

from playground.open_duck_mini_v2.mujoco_infer import MjInfer, resolve_policy  # noqa: E402

REFERENCE = "playground/open_duck_mini_v2/data/polynomial_coefficients.pkl"
SCENE = "playground/open_duck_mini_v2/xmls/scene_person.xml"   # 장애물 없는 씬

HOLD_S = 15.0        # 사람이 멈춘 뒤 더 굴리는 시간. 0.13 m/s 로 2 m 를 따라잡는 시간이다
END_S = 3.0          # "끝 거리" 를 평균낼 마지막 구간
TOO_CLOSE_M = 0.35   # 이보다 가까우면 붙은 것 (몸통 반길이 ~0.1 + 사람 발 0.11 + 여유)
# "다가오기" 는 대본이 사람을 25 cm 까지 붙이므로 최소 거리로는 못 가른다. 사람이 멈춘 뒤
# 끝 거리로 본다 — 이보다 가까우면 물러나지 못한 것이다.
APPROACH_BACK_M = 0.45
LOST_S = 5.0         # 이보다 오래 연달아 놓치면 놓친 것
FAR_M = 0.9          # 끝 거리가 이보다 멀면 못 따라온 것 (정지 거리 0.55 + 카메라 거리 오차)
APPROACH_STOP_M = 0.25

# 오리는 원점에서 +x 를 보고 선다. 사람은 처음에 화각(±31°) 안에 있어야 한다 —
# 처음부터 안 보이면 탐색이 기본 방향(왼쪽)으로 돌아 대본과 무관한 실패가 섞인다.
# (이름, 설명, 경유점). "approach" 는 경유점 대신 오리 쪽으로 걸어온다.
_ARC = [(np.cos(a), np.sin(a)) for a in np.radians(np.linspace(0.0, 180.0, 19))]
SCENARIOS = {
    "away": ("멀어지기", [(1.0, 0.0), (2.5, 0.0)]),
    "cross": ("앞을 가로지르기", [(1.2, -0.6), (1.2, 0.8)]),
    "turn": ("멀어지다 왼쪽으로 꺾기", [(1.0, 0.0), (2.0, 0.0), (2.0, 1.0)]),
    "approach": ("다가와서 25 cm 앞에 서기", [(2.0, 0.0)]),
    "around": ("오리 둘레 반 바퀴 (반경 1 m)", _ARC),
}


class Walker:
    """사람의 대본. 매 제어스텝 위치와 바라보는 방향을 낸다.

    사람 모델은 제 +x 를 본다 (발끝이 +x). 걷는 방향을 보게 돌려야 옆에서 볼 때
    두 발목이 겹친다 — 실제로 카메라 거리 추정이 부풀려지는 조건이 그것이다.
    """

    def __init__(self, name, speed):
        self.name, self.speed = name, float(speed)
        self.pts = np.array(SCENARIOS[name][1], dtype=float)
        self.pos = self.pts[0].copy()
        d = self.pts[1] - self.pts[0] if len(self.pts) > 1 else -self.pos
        self.heading = float(np.arctan2(d[1], d[0]))
        self.seg = 1
        self.done_t = None

    def step(self, t, dt, robot_xy):
        if self.done_t is not None:
            return
        step = self.speed * dt
        if self.name == "approach":
            d = np.asarray(robot_xy) - self.pos
            r = float(np.hypot(d[0], d[1]))
            if r - step <= APPROACH_STOP_M:
                self.done_t = t
                return
            self.heading = float(np.arctan2(d[1], d[0]))
            self.pos = self.pos + step * d / r
            return
        while step > 0 and self.seg < len(self.pts):
            d = self.pts[self.seg] - self.pos
            r = float(np.hypot(d[0], d[1]))
            if r > 1e-9:
                self.heading = float(np.arctan2(d[1], d[0]))
            if r <= step:
                self.pos = self.pts[self.seg].copy()
                step -= r
                self.seg += 1
            else:
                self.pos = self.pos + step * d / r
                step = 0.0
        if self.seg >= len(self.pts):
            self.done_t = t

    def put(self, data):
        data.mocap_pos[0] = [self.pos[0], self.pos[1], 0.0]
        h = self.heading / 2.0
        data.mocap_quat[0] = [np.cos(h), 0.0, 0.0, np.sin(h)]


def run_one(job):
    """(대본, 속도) 한 판. 프로세스마다 따로 부른다 (렌더러가 프로세스마다 하나)."""
    name, speed, pol, no_avoid, remember_target, no_backoff = job
    onnx, rr, lin_vel_y, fr = pol
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        m = MjInfer(SCENE, REFERENCE, onnx, False, rr, lin_vel_y)
        if fr is not None:
            m.model.actuator_forcerange[:] = np.array([-fr, fr])
        m.full_reset()
    m.direct_head = False
    m.heading_hold = False
    if no_avoid:
        m.avoid = False
    if remember_target:
        m.target_not_obstacle = False
    if no_backoff:
        m.backoff = False
    walker = Walker(name, speed)
    walker.put(m.data)
    mujoco.mj_forward(m.model, m.data)
    # 뷰어에서 F 를 눌렀을 때와 같게 시작한다 (mujoco_infer 의 F 키 처리).
    m.follow, m.follow_lost, m.target_world_deg, m.avoid_side = True, 0, None, 0

    ctrl_dt = m.sim_dt * m.decimation
    # 걷는 시간의 상한을 넉넉히 잡고, 멈춘 뒤 HOLD_S 를 더 굴린다.
    walk_len = float(np.sum(np.hypot(*np.diff(walker.pts, axis=0).T))) if len(walker.pts) > 1 else 2.0
    t_max = walk_len / speed + 2.0 + HOLD_S
    log = []
    k = 0
    with contextlib.redirect_stdout(io.StringIO()):   # 렌더러 생성 등 뷰어 로그
        while True:
            t = k * ctrl_dt
            base = m.get_floating_base_qpos(m.data.qpos)
            walker.step(t, ctrl_dt, base[:2])
            walker.put(m.data)
            m.control_step()
            seen = m.follow_lost == 0
            detour = bool(m.avoid and m.detour_wp is not None)
            for _ in range(m.decimation):
                mujoco.mj_step(m.model, m.data)
            base = m.get_floating_base_qpos(m.data.qpos)
            rel = walker.pos - base[:2]
            bear = (np.degrees(np.arctan2(rel[1], rel[0])) - m.body_yaw_deg() + 180.0) % 360.0 - 180.0
            log.append((t, base[0], base[1], walker.pos[0], walker.pos[1],
                        float(np.hypot(rel[0], rel[1])), bear, seen, detour,
                        float(m.get_gravity(m.data)[-1]), walker.done_t is None,
                        m.commands[0], m.commands[1], m.commands[2]))
            k += 1
            if walker.done_t is not None and t >= walker.done_t + HOLD_S:
                break
            if t > t_max + 30.0:   # 대본이 안 끝나는 경우 (approach 에서 오리가 계속 물러나는 등)
                break
    return name, speed, np.array(log, dtype=float), walker.done_t


def summarize(log, done_t, ctrl_dt, name=None):
    t, d, bear, seen, detour, up, moving = (log[:, 0], log[:, 5], log[:, 6], log[:, 7] > 0.5,
                                            log[:, 8] > 0.5, log[:, 9], log[:, 10] > 0.5)
    miss = ~seen
    longest, run = 0, 0
    for x in miss:
        run = run + 1 if x else 0
        longest = max(longest, run)
    end = t >= t[-1] - END_S
    s = dict(
        miss=100.0 * miss.mean(), longest=longest * ctrl_dt,
        d_walk=float(d[moving].max()) if moving.any() else float("nan"),
        d_end=float(d[end].mean()), d_min=float(d.min()),
        bear=float(np.abs(bear).mean()), detour=100.0 * detour.mean(),
        fell=bool(up.min() < 0.5), walk_s=float(done_t) if done_t is not None else float("nan"),
    )
    why = []
    if s["fell"]:
        why.append("넘어짐")
    # "다가오기" 는 대본이 사람을 카메라 사각(밴드가 화각 아래로 빠지는 0.46 m 안)으로
    # 밀어 넣으므로, 다가오는 동안과 물러나는 동안은 제어기가 뭘 하든 못 본다. 그래서
    # 연달아 놓친 시간 대신 **끝 END_S 초에 다시 보이는가**로 본다.
    if (seen[end].mean() < 0.5) if name == "approach" else (s["longest"] > LOST_S):
        why.append("놓침")
    if s["d_end"] > FAR_M:
        why.append("멀어짐")
    if (s["d_end"] < APPROACH_BACK_M) if name == "approach" else (s["d_min"] < TOO_CLOSE_M):
        why.append("붙음")
    if s["detour"] > 0:
        why.append("헛우회")
    s["verdict"] = "OK" if not why else "/".join(why)
    return s


def plot(results, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for f in ("Malgun Gothic", "AppleGothic", "NanumGothic"):
        if any(f in x.name for x in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = f
            break
    plt.rcParams["axes.unicode_minus"] = False

    names = [n for n in SCENARIOS if any(r[0] == n for r in results)]
    speeds = sorted({r[1] for r in results})
    fig, axes = plt.subplots(len(names), len(speeds), figsize=(4.2 * len(speeds), 3.6 * len(names)),
                             squeeze=False)
    for (name, speed, log, done_t, s) in results:
        ax = axes[names.index(name)][speeds.index(speed)]
        t, rx, ry, px, py, seen = log[:, 0], log[:, 1], log[:, 2], log[:, 3], log[:, 4], log[:, 7] > 0.5
        ax.plot(px, py, color="0.25", lw=2.0, label="사람")
        ax.plot(rx, ry, color="tab:blue", lw=1.6, label="오리")
        ax.scatter(rx[~seen], ry[~seen], s=5, color="tab:red", zorder=3, label="놓침")
        for tt in np.arange(0.0, t[-1] + 1e-9, 5.0):   # 같은 시각끼리 가는 선으로 잇는다
            i = int(np.argmin(np.abs(t - tt)))
            ax.plot([rx[i], px[i]], [ry[i], py[i]], color="0.75", lw=0.8, zorder=1)
            ax.annotate(f"{tt:.0f}", (px[i], py[i]), fontsize=7, color="0.35",
                        xytext=(3, 3), textcoords="offset points")
        ax.plot(px[-1], py[-1], marker="*", ms=12, color="0.1")
        ax.plot(rx[-1], ry[-1], marker="o", ms=6, color="tab:blue")
        # 대본마다 같은 축척으로 보이게, 최소 2.6 x 2.2 m 창을 잡는다 (납작해지지 않게).
        xs, ys = np.r_[rx, px, 0.0], np.r_[ry, py, 0.0]
        cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
        hx = max(1.3, (xs.max() - xs.min()) / 2 + 0.3)
        hy = max(1.1, (ys.max() - ys.min()) / 2 + 0.3)
        ax.set_xlim(cx - hx, cx + hx)
        ax.set_ylim(cy - hy, cy + hy)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.set_title(f"{SCENARIOS[name][0]} · {speed:.1f} m/s — {s['verdict']}\n"
                     f"끝 {s['d_end']:.2f} m · 최소 {s['d_min']:.2f} m · 놓침 {s['miss']:.0f}%",
                     fontsize=9)
    axes[0][0].legend(fontsize=7, loc="upper left")
    fig.suptitle(title + "\n숫자 = 초, 회색 선 = 같은 시각의 오리와 사람", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.5 / fig.get_figheight()))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="걷는 사람 따라가기 채점")
    ap.add_argument("-o", "--onnx_model_path", type=str, default=None,
                    help="안 주면 기본 정책(DEFAULT_POLICY)을 학습 조건째로")
    ap.add_argument("--no_ref_range", action="store_true")
    ap.add_argument("--lin_vel_y", type=float, default=None)
    ap.add_argument("--forcerange", type=float, default=None)
    ap.add_argument("--scenario", nargs="+", choices=list(SCENARIOS), default=list(SCENARIOS))
    ap.add_argument("--speeds", type=float, nargs="+", default=[0.1, 0.2, 0.3],
                    help="사람 걷는 속도 m/s. 오리는 0.11~0.15")
    ap.add_argument("--no_avoid", action="store_true",
                    help="회피를 끄고 잰다. 빈 바닥에서 회피가 따라가기를 방해하는지 가를 때")
    ap.add_argument("--no_backoff", action="store_true",
                    help="물러나기를 끈다 (09-29 이전 동작, 사람이 다가오면 그냥 선다)")
    ap.add_argument("--remember_target", action="store_true",
                    help="사람 발도 장애물로 기억한다 (target_not_obstacle 끔 = 09-29 이전 동작)")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--plot", type=str, default=os.path.join(ROOT, "head_cam_out", "follow_moving.png"),
                    help="위에서 본 궤적 그림. 빈 문자열이면 안 그린다")
    args = ap.parse_args()
    # 위에서 REPO 로 chdir 했으므로 상대경로는 저장소 루트 기준으로 되돌린다.
    if args.plot and not os.path.isabs(args.plot):
        args.plot = os.path.join(ROOT, args.plot)

    onnx, rr, lvy, fr = resolve_policy(args.onnx_model_path, not args.no_ref_range,
                                       args.lin_vel_y, args.forcerange)
    if not os.path.isabs(onnx) and not os.path.exists(onnx):
        onnx = os.path.join(ROOT, onnx)     # -o 는 저장소 루트 기준 (eval_walk 와 같다)
    pol = (os.path.abspath(onnx), rr, lvy, fr)
    jobs = [(n, v, pol, args.no_avoid, args.remember_target, args.no_backoff)
            for n in args.scenario for v in args.speeds]
    print(f"정책 {os.path.basename(onnx)} (ref_range={rr}, dy ±{lvy}, 토크 ±{fr}) · "
          f"회피 {'끔' if args.no_avoid else '켬'}"
          f"{' · 사람 발도 기억 (09-29 이전)' if args.remember_target else ''}"
          f"{' · 물러나기 끔' if args.no_backoff else ''}"
          f" · 빈 바닥 · 사람이 멈춘 뒤 {HOLD_S:.0f}초 더")
    with cf.ProcessPoolExecutor(max_workers=args.workers) as ex:
        outs = list(ex.map(run_one, jobs))

    ctrl_dt = 0.02
    results = []
    print(f"\n  {'대본':<24}{'속도':>5}{'걸은초':>7}{'놓침%':>7}{'최장놓침':>8}"
          f"{'걷는중최대':>10}{'끝거리':>8}{'최소':>7}{'방위°':>7}{'우회%':>7}  판정")
    for name, speed, log, done_t in outs:
        s = summarize(log, done_t, ctrl_dt, name)
        results.append((name, speed, log, done_t, s))
        print(f"  {SCENARIOS[name][0]:<24}{speed:5.1f}{s['walk_s']:7.1f}{s['miss']:7.0f}"
              f"{s['longest']:7.1f}s{s['d_walk']:10.2f}{s['d_end']:8.2f}{s['d_min']:7.2f}"
              f"{s['bear']:7.1f}{s['detour']:7.0f}  {s['verdict']}")
    ok = sum(r[4]["verdict"] == "OK" for r in results)
    print(f"\n  OK {ok}/{len(results)}   (끝거리 > {FAR_M} m 멀어짐 · 최소 < {TOO_CLOSE_M} m 붙음"
          f" (다가오기는 끝거리 < {APPROACH_BACK_M} m) · "
          f"{LOST_S:.0f}초 넘게 연달아 놓침 (다가오기는 끝 {END_S:.0f}초에 안 보임) · 빈 바닥 우회 = 헛우회)")
    if args.plot:
        os.makedirs(os.path.dirname(args.plot), exist_ok=True)
        plot(results, args.plot, f"{os.path.basename(onnx)} · 회피 {'끔' if args.no_avoid else '켬'}"
                                 f"{' · 사람 발도 기억' if args.remember_target else ''}")
        print(f"  그림: {args.plot}")


if __name__ == "__main__":
    main()
