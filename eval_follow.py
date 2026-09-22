r"""루프를 닫은 채로 굴려서 정말 사람에게 가는지 잰다 (뷰어 없음).

뷰어에서 F 를 누르면 도는 것과 **같은 코드**(`MjInfer.follow_step`)를 쓴다. 여기서
루프를 따로 만들면 뷰어에서 보는 것과 다른 결과가 나와 측정이 쓸모없어진다.

보는 값은 셋이다:
  거리    — 줄어드는가. 이게 추종의 정의다.
  방위각  — 0 으로 수렴하는가. 사람을 정면에 두고 가는가.
  검출률  — 시야에 붙들고 있는가.

`--no_follow` 로 같은 조건에서 추종을 끄고 굴리면 비교가 된다. fr186 은 직진
명령만 줘도 왼쪽으로 도는 드리프트가 있어서, 그게 얼마나 되는지와 추종이 그걸
덮는지를 이 두 판으로 가른다.

    .venv\Scripts\python.exe eval_follow.py
    .venv\Scripts\python.exe eval_follow.py --no_follow
    .venv\Scripts\python.exe eval_follow.py --person 2.4 -0.9
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

from eval_bearing import truth_bearing, BAND_Z  # noqa: E402
from playground.open_duck_mini_v2.mujoco_infer import MjInfer  # noqa: E402

REFERENCE = "playground/open_duck_mini_v2/data/polynomial_coefficients.pkl"
SCENE = "playground/open_duck_mini_v2/xmls/scene_obstacles.xml"
ONNX = "../from_ubai/fr186_2026_09_12_174638_300482560.onnx"


def yaw_of(quat):
    w, x, y, z = quat
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--onnx_model_path", type=str, default=ONNX)
    ap.add_argument("--model_path", type=str, default=SCENE)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--person", type=float, nargs=2, default=[2.4, 0.9],
                    help="사람 위치 x y. 기본값은 일부러 옆으로 틀어 놓았다 — "
                         "정면에 두면 그냥 직진해도 가까워져서 추종인지 알 수 없다.")
    ap.add_argument("--no_follow", action="store_true",
                    help="추종을 끄고 전진 명령만 준다 (드리프트 비교용)")
    ap.add_argument("--kp", type=float, default=None, help="추종 요 게인 (도당)")
    ap.add_argument("--align", type=float, default=None, help="이보다 틀어지면 제자리 선회")
    ap.add_argument("--quiet", action="store_true", help="표만 빼고 요약만")
    ap.add_argument("--no_avoid", action="store_true", help="장애물 회피를 끈다")
    ap.add_argument("--no_head", action="store_true", help="머리 표적추종을 끈다")
    ap.add_argument("--no_sidestep", action="store_true", help="게걸음 회피를 끈다")
    ap.add_argument("--fovy", type=float, default=None,
                    help="머리 카메라 세로 화각을 바꿔 본다 (기본 49 = Pi cam v2)")
    ap.add_argument("--aim", type=float, default=None,
                    help="머리 조준: 1.0 사람만, 0.0 갈 방향만, 0.5 가운데")
    ap.add_argument("--no_ref_range", action="store_true")
    args = ap.parse_args()

    m = MjInfer(args.model_path, REFERENCE, args.onnx_model_path,
                False, not args.no_ref_range)
    m.full_reset()
    m.direct_head = False
    m.data.mocap_pos[0] = [args.person[0], args.person[1], 0.0]
    mujoco.mj_forward(m.model, m.data)

    cam_id = m.model.camera("head_cam").id
    if args.kp is not None:
        m.FOLLOW_KP = args.kp
    if args.align is not None:
        m.FOLLOW_ALIGN_DEG = args.align
    if args.no_avoid:
        m.avoid = False
    if args.no_head:
        m.head_track = False
    if args.no_sidestep:
        m.use_sidestep = False
    if args.aim is not None:
        m.HEAD_AIM_BLEND = args.aim
    if args.fovy is not None:
        m.model.cam_fovy[cam_id] = args.fovy
    if args.no_follow:
        m.commands[0] = m.COMMANDS_RANGE_X[1]
    else:
        m.follow = True

    ctrl_dt = m.sim_dt * m.decimation
    n = int(args.seconds / ctrl_dt)
    y0 = yaw_of(m.get_floating_base_qpos(m.data.qpos)[3:7])

    # 요는 매 스텝 차분을 누적한다 (eval_walk.py 와 같은 이유). 시작과 끝 자세만
    # 빼면 180° 를 넘는 회전이 반대 부호로 접혀서 좌회전이 우회전으로 읽힌다.
    yaw_acc, yaw_prev = 0.0, y0
    log, miss, blocked_n = [], 0, 0
    for k in range(n):
        # 뷰어의 run() 과 **같은 함수**를 쓴다. 여기서 루프를 복제하면 언젠가
        # 한쪽만 고쳐져 측정값과 화면이 조용히 갈린다 (실제로 DIRECT_HEAD 를
        # 빠뜨려 머리 추종이 통째로 무효가 된 적 있다).
        m.control_step()
        if m.follow:
            if m.follow_lost:
                miss += 1
            if m.follow_blocked:
                blocked_n += 1
        for _ in range(m.decimation):
            mujoco.mj_step(m.model, m.data)

        base = m.get_floating_base_qpos(m.data.qpos)
        p = m.data.mocap_pos[0]
        gt_b, _ = truth_bearing(m.data, cam_id, [p[0], p[1], BAND_Z])
        yaw_now = yaw_of(base[3:7])
        yaw_acc += (yaw_now - yaw_prev + np.pi) % (2 * np.pi) - np.pi
        yaw_prev = yaw_now
        log.append((k * ctrl_dt,
                    float(np.linalg.norm(base[:2] - p[:2])),
                    gt_b,
                    np.degrees(yaw_acc),
                    float(m.get_gravity(m.data)[-1])))

    t, dist, bear, yaw, up = (np.array(c) for c in zip(*log))
    mode = ("추종 OFF (전진 명령만)" if args.no_follow
            else "화각%.0f° 회피%s 머리%s(aim %.1f) 게걸음%s" % (m.model.cam_fovy[cam_id],
                                      "O" if m.avoid else "X",
                                      "O" if m.head_track else "X",
                                      m.HEAD_AIM_BLEND,
                                      "O" if m.use_sidestep else "X"))
    print("\n" + "=" * 62)
    print(" {}   사람 ({:.1f}, {:+.1f})   {:.0f}초".format(
        mode, args.person[0], args.person[1], args.seconds))
    print("=" * 62)
    print("    {:>5} {:>8} {:>9} {:>9}".format("t[s]", "거리[m]", "방위[°]", "요[°]"))
    if not args.quiet:
        for i in range(0, len(t), max(1, len(t) // 10)):
            print("    {:5.1f} {:8.2f} {:9.1f} {:9.1f}".format(t[i], dist[i], bear[i], yaw[i]))

    print("\n    거리   {:.2f} m -> {:.2f} m   ({:+.2f} m)".format(
        dist[0], dist[-1], dist[-1] - dist[0]))
    print("    방위   {:+.1f}° -> {:+.1f}°   마지막 3초 절대평균 {:.1f}°".format(
        bear[0], bear[-1], np.abs(bear[t > t[-1] - 3.0]).mean()))
    print("    요     누적 {:+.1f}°".format(yaw[-1]))
    _b = m.get_floating_base_qpos(m.data.qpos)
    _p = m.data.mocap_pos[0]
    print("    최종   오리({:.2f},{:.2f})  사람({:.2f},{:.2f})".format(
        _b[0], _b[1], _p[0], _p[1]))
    if not args.no_follow:
        print("    표적 놓친 제어스텝 {}/{} ({:.0f}%)".format(miss, n, 100 * miss / n))
        print("    막혔다고 판정한 스텝 {}/{} ({:.0f}%)".format(blocked_n, n, 100 * blocked_n / n))
    if up.min() < 0.5:
        print("    ⚠ 넘어졌다 (최저 up {:.2f})".format(up.min()))


if __name__ == "__main__":
    main()
