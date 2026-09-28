r"""출발점에서 도착점까지 혼자 가는지 잰다 (뷰어 없음).

방향키로 모는 것과 다른 점은 하나다: 명령을 사람이 아니라 오리가 만든다.
순서는 제자리 선회로 사람 찾기 -> 추종 -> 정지이고, 뷰어에서 N 을 누르면
도는 것과 **같은 코드**(`MjInfer.goto_step`)를 쓴다. 여기서 루프를 따로 만들면
화면과 측정값이 조용히 갈린다.

출발 방위를 여러 개 돌려 본다. 사람을 정면에 두고 시작하면 그냥 직진해도
도착해서, 스스로 찾아간 건지 알 수가 없다. 등지고 시작(180°)하는 판이 이
기능의 실제 시험이다.

장애물은 아직 놓지 않는다 (scene_person.xml). 빈 바닥에서 도착도 못 하면
회피를 얹어 봐야 실패 원인을 가를 수 없다.

    .venv\Scripts\python.exe eval_goto.py
    .venv\Scripts\python.exe eval_goto.py --yaw 180 --seconds 40 --verbose
    .venv\Scripts\python.exe eval_goto.py -o ../from_ubai/fr186_....onnx --ref_range
"""

import argparse
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
SCENE = "playground/open_duck_mini_v2/xmls/scene_person.xml"
# 기본 정책은 mujoco_infer.DEFAULT_POLICY (2026-09-28 부터 hp0dy2 시드 0) 를
# 학습 조건째로 쓴다. -o 를 주면 그 정책의 조건(--ref_range, --lin_vel_y,
# --forcerange)을 직접 맞춰 줘야 한다. 09-23 까지의 기본은 head 910949 였다.

FALLEN = 0.5

# 출발 방위(도). 0 이 사람을 대충 정면에 두는 쪽이다.
DEFAULT_YAWS = [0.0, 90.0, 180.0, -90.0]


def twist_base(m, deg):
    """가던 오리의 몸통을 그 자리에서 deg 만큼 홱 돌려놓는다.

    누가 건드렸을 때 다시 찾아가는지 보려고 쓴다. 방위를 바꾸는 것만으로
    충분하다 — 이 구조에서 방향은 매 제어스텝 카메라로 새로 재기 때문에,
    틀어졌을 때 돌아오는지는 곧 "안 보이게 됐을 때 다시 찾는지" 와 같은 질문이다.
    """
    qpos = m.data.qpos
    base = m.get_floating_base_qpos(qpos).copy()
    w, x, y, z = base[3:7]
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    h = (yaw + np.radians(deg)) / 2.0
    # 롤·피치는 버린다. 걷는 중 몸통 기울기는 작고(최저 up 0.99), 여기서 보려는
    # 것은 요 하나다. 온전한 합성을 하려다 틀리면 시험 자체가 의심스러워진다.
    base[3:7] = [np.cos(h), 0.0, 0.0, np.sin(h)]
    m.set_floating_base_qpos(base, qpos)
    mujoco.mj_forward(m.model, m.data)


def run_one(m, start, yaw, person, seconds, verbose=False, twist=None, avoid=False,
            face_target=True):
    m.full_reset()
    m.direct_head = False
    m.place(start, yaw, person)
    # 빈 바닥이면 끈다. 장애물 씬에서는 --avoid 로 켠다 — 뷰어의 N 은 회피를
    # 켠 채로 돌므로(`self.avoid = True` 기본값) 그 조건을 재려면 켜야 한다.
    m.avoid = avoid
    m.face_target = face_target
    m.start_goto()

    ctrl_dt = m.sim_dt * m.decimation
    n = int(seconds / ctrl_dt)
    scan_t, arrive_t, min_up = None, None, 1.0
    log = []
    twist_at = None if twist is None else int(twist[0] / ctrl_dt)
    twist_t, regain_t, lost_after = None, None, False
    for k in range(n):
        if twist_at is not None and k == twist_at:
            twist_base(m, twist[1])
            twist_t = k * ctrl_dt
            lost_after = True
            print("    >>> {:.1f}초에 몸통을 {:+.0f}° 돌려놓음".format(twist_t, twist[1]))
        m.control_step()
        if lost_after and regain_t is None and m.follow_lost == 0 and twist_t is not None                 and k * ctrl_dt > twist_t:
            regain_t = k * ctrl_dt - twist_t
        for _ in range(m.decimation):
            mujoco.mj_step(m.model, m.data)

        t = k * ctrl_dt
        if scan_t is None and m.goto_phase != "scan":
            scan_t = t
        if arrive_t is None and m.goto_phase == "arrived":
            arrive_t = t
        base = m.get_floating_base_qpos(m.data.qpos)
        d = float(np.linalg.norm(base[:2] - m.data.mocap_pos[0][:2]))
        up = float(m.get_gravity(m.data)[-1])
        min_up = min(min_up, up)
        log.append((t, d))
        if arrive_t is not None or up < FALLEN:
            break

    t, dist = (np.array(c) for c in zip(*log))
    if verbose:
        for i in range(0, len(t), max(1, len(t) // 12)):
            print("      {:5.1f}s  {:.2f} m".format(t[i], dist[i]))
    base = m.get_floating_base_qpos(m.data.qpos)
    return dict(scan_t=scan_t, arrive_t=arrive_t, d0=dist[0], d1=dist[-1],
                min_up=min_up, t_end=t[-1], regain_t=regain_t,
                pos=(float(base[0]), float(base[1])))


def main():
    ap = argparse.ArgumentParser(description="출발점 -> 도착점 자율 이동 측정")
    ap.add_argument("-o", "--onnx_model_path", type=str, default=None,
                    help="안 주면 기본 정책(DEFAULT_POLICY)을 학습 조건째로")
    ap.add_argument("--model_path", type=str, default=SCENE)
    ap.add_argument("--seconds", type=float, default=35.0)
    ap.add_argument("--start", type=float, nargs=2, default=[0.0, 0.0])
    ap.add_argument("--person", type=float, nargs=2, default=[1.90, 0.15],
                    help="도착점 x y (여기 사람이 서 있다)")
    ap.add_argument("--yaw", type=float, nargs="+", default=None,
                    help="시험할 출발 방위(도). 안 주면 0/90/180/-90")
    ap.add_argument("--random", type=int, default=0, metavar="N",
                    help="출발 방위를 무작위로 N 판. 네 방향만 보면 그 넷에만 "
                         "맞춰진 걸 못 잡는다")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--twist", type=float, nargs=2, default=None,
                    metavar=("T", "DEG"),
                    help="T 초에 몸통을 DEG 만큼 홱 돌려놓는다 (건드렸을 때 "
                         "다시 찾아가는지). 예: --twist 8 150")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--avoid", action="store_true",
                    help="장애물 회피를 켠다 (뷰어 N 과 같은 조건). "
                         "장애물 씬(scene_obstacles.xml)을 줄 때 붙일 것")
    ap.add_argument("--no_face_target", action="store_true",
                    help="우회할 때 몸통을 갈 방향으로 튼다 (기본은 몸통을 사람에 "
                         "고정하고 게걸음으로 비킨다)")
    ap.add_argument("--ref_range", action="store_true",
                    help="2026-09-04 이후 좁은 dy(±0.111)로 학습된 정책이면 붙일 것. "
                         "기본 정책(head 910949)은 넓은 dy 라 붙이면 안 된다.")
    ap.add_argument("--lin_vel_y", type=float, default=None,
                    help="게걸음 범위만 덮어쓴다. hp0dy2 정책은 --ref_range --lin_vel_y 0.2")
    ap.add_argument("--forcerange", type=float, default=None,
                    help="토크 상한[N·m]. 씬 XML 은 ±3.23 고정인데 fr186 계열은 "
                         "±1.86 으로 학습됐다. 학습값과 다르면 걸음이 딴판이 된다")
    args = ap.parse_args()

    if args.random:
        yaws = list(np.random.default_rng(args.seed).uniform(-180, 180, args.random))
    elif args.yaw is not None:
        yaws = args.yaw
    else:
        yaws = DEFAULT_YAWS
    (args.onnx_model_path, args.ref_range, args.lin_vel_y,
     args.forcerange) = resolve_policy(args.onnx_model_path, args.ref_range,
                                       args.lin_vel_y, args.forcerange)
    m = MjInfer(args.model_path, REFERENCE, args.onnx_model_path,
                False, args.ref_range, args.lin_vel_y)
    if args.forcerange is not None:
        m.model.actuator_forcerange[:] = np.array([-args.forcerange,
                                                   args.forcerange])

    d_goal = float(np.hypot(args.person[0] - args.start[0],
                            args.person[1] - args.start[1]))
    print("\n" + "=" * 72)
    print(" 출발 ({:.2f}, {:.2f})  ->  도착 ({:.2f}, {:.2f})   거리 {:.2f} m   제한 {:.0f}초"
          .format(args.start[0], args.start[1], args.person[0], args.person[1],
                  d_goal, args.seconds))
    print(" {} · 회피 {} · 정지 반경 {:.2f} m".format(
        os.path.basename(args.model_path), "ON" if args.avoid else "OFF",
        m.FOLLOW_STOP_M))
    if args.avoid:
        print(" 우회: {}".format("몸통을 갈 방향으로" if args.no_face_target
                                 else "몸통은 사람 쪽, 게걸음으로"))
    print("=" * 72)
    print("  {:>7} {:>9} {:>9} {:>9} {:>8}  {}".format(
        "출발방위", "발견[s]", "도착[s]", "최종거리", "최저up", "결과"))

    ok = 0
    for yaw in yaws:
        r = run_one(m, args.start, yaw, args.person, args.seconds, args.verbose,
                    args.twist, args.avoid, not args.no_face_target)
        if r["min_up"] < FALLEN:
            verdict = "넘어짐 {:.1f}s".format(r["t_end"])
        elif r["arrive_t"] is not None:
            verdict = "도착"
            ok += 1
        elif r["scan_t"] is None:
            verdict = "사람 못 찾음"
        else:
            verdict = "시간 초과 ({:+.2f} m)".format(r["d1"] - r["d0"])
        if args.twist is not None:
            verdict += "   재포착 " + ("못함" if r["regain_t"] is None
                                     else "{:.1f}초".format(r["regain_t"]))
        print("  {:+7.0f}° {:>9} {:>9} {:9.2f} {:8.3f}  {}".format(
            yaw,
            "-" if r["scan_t"] is None else "{:.1f}".format(r["scan_t"]),
            "-" if r["arrive_t"] is None else "{:.1f}".format(r["arrive_t"]),
            r["d1"], r["min_up"], verdict))

    print("\n  도착 {}/{}".format(ok, len(yaws)))


if __name__ == "__main__":
    main()
