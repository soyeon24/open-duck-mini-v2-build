r"""색추종이 뽑은 방위각이 맞는지 정답과 맞대본다.

사람이 mocap 이라 위치를 정확히 알고, 카메라 자세도 알고 있다. 그래서 **정답
방위각을 해석적으로 계산할 수 있다.** 눈으로 "대충 사람 쪽을 보는 것 같다" 가 아니라
도(degree) 로 오차를 잰다. 여기서 오차가 크면 그 위에 뭘 얹어도 소용없다.

두 가지를 잰다:
  1. 정지 스윕 — 오리를 home 자세로 세워두고 사람을 여러 방위/거리에 놓는다.
     흔들림 없는 조건에서 **색추종 자체의 정확도**를 본다.
  2. 보행 중  — 걸으면서 매 프레임 잰다. 카메라가 피치 18.5°/롤 14.1° 로 흔들리는
     조건에서 **살아남는지**를 본다. 이게 색추종을 고른 이유이기도 하다.

제어 루프는 `eval_walk.py` 의 `rollout()` 과 같은 순서다 (50Hz).

    .venv\Scripts\python.exe eval_bearing.py
    .venv\Scripts\python.exe eval_bearing.py --seconds 10
"""

import argparse
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(ROOT, "Open_Duck_Playground")
sys.path.insert(0, ROOT)          # band_tracker 는 프로젝트 루트에 있다
os.chdir(REPO)
sys.path.insert(0, REPO)

import mujoco  # noqa: E402
from PIL import Image  # noqa: E402

import band_tracker  # noqa: E402
from playground.open_duck_mini_v2.mujoco_infer import MjInfer  # noqa: E402

REFERENCE = "playground/open_duck_mini_v2/data/polynomial_coefficients.pkl"
SCENE = "playground/open_duck_mini_v2/xmls/scene_obstacles.xml"
ONNX = "../from_ubai/fr186_2026_09_12_174638_300482560.onnx"

# 밴드 중심은 사람 원점에서 이만큼 위다 (scene_obstacles.xml 의 person_band_* pos z).
BAND_Z = 0.09


def truth_bearing(data, cam_id, person_xyz):
    """카메라 프레임에서 본 밴드의 정답 방위/고도 [deg]. 왼쪽/위가 양수.

    카메라 좌표계는 -Z 를 보고 +X 가 오른쪽, +Y 가 위다 (MuJoCo 관례).
    비전이 내놓는 값과 같은 프레임에서 재야 공정한 비교가 된다.
    """
    R = data.cam_xmat[cam_id].reshape(3, 3)
    p = R.T @ (np.asarray(person_xyz, float) - data.cam_xpos[cam_id])
    fwd = -p[2]
    return (-np.degrees(np.arctan2(p[0], fwd)),
            np.degrees(np.arctan2(p[1], fwd)))


def annotate(img, res):
    """검출 위치에 십자선을 긋는다. 눈으로 확인용."""
    out = img.copy()
    if res is None:
        out[:4, :] = [255, 0, 0]          # 미검출이면 위쪽에 빨간 띠
        return out
    u, v = int(round(res["u"])), int(round(res["v"]))
    h, w, _ = out.shape
    if 0 <= u < w:
        out[:, max(0, u - 1):min(w, u + 2)] = [255, 0, 255]
    if 0 <= v < h:
        out[max(0, v - 1):min(h, v + 2), :] = [255, 0, 255]
    return out


def contact_sheet(frames, cols, path):
    if not frames:
        return
    h, w, _ = frames[0].shape
    rows = (len(frames) + cols - 1) // cols
    sheet = np.full((rows * h, cols * w, 3), 255, np.uint8)
    for i, f in enumerate(frames):
        r_, c_ = divmod(i, cols)
        sheet[r_ * h:(r_ + 1) * h, c_ * w:(c_ + 1) * w] = f
    Image.fromarray(sheet).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--onnx_model_path", type=str, default=ONNX)
    ap.add_argument("--model_path", type=str, default=SCENE)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--out", type=str, default=os.path.join(ROOT, "head_cam_out"))
    ap.add_argument("--no_ref_range", action="store_true")
    ap.add_argument("--cam_pitch", type=float, default=0.0,
                    help="카메라를 이 각도(도)만큼 **추가로** 아래로 기울인다. 모델에 "
                         "이미 10° 하향이 들어 있으므로 0 이 기본 설정이고, 10 을 주면 "
                         "합쳐서 20° 가 된다. 마운트 각도를 다시 잡을 때 쓰는 A/B 용.")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    m = MjInfer(args.model_path, REFERENCE, args.onnx_model_path,
                False, not args.no_ref_range)
    m.full_reset()
    m.direct_head = False

    cam_id = m.model.camera("head_cam").id

    if args.cam_pitch:
        # 카메라 로컬 X 축(오른쪽) 둘레로 돌린다. 오른손 법칙으로 +φ 는 시선을 위로
        # 올리므로, 아래로 기울이려면 부호를 뒤집는다. 마운트 쿼터니언에 뒤에서
        # 곱해야 카메라 자기 축 기준 회전이 된다.
        q_rot = np.zeros(4)
        mujoco.mju_axisAngle2Quat(q_rot, np.array([1.0, 0.0, 0.0]),
                                  -np.radians(args.cam_pitch))
        q_new = np.zeros(4)
        mujoco.mju_mulQuat(q_new, m.model.cam_quat[cam_id].copy(), q_rot)
        m.model.cam_quat[cam_id] = q_new
        print("\n>>> 카메라를 {:.0f}° 아래로 기울였다".format(args.cam_pitch))
    fovy = float(m.model.cam_fovy[cam_id])
    rend = mujoco.Renderer(m.model, height=args.height, width=args.width)

    def look():
        rend.update_scene(m.data, camera="head_cam")
        img = rend.render().copy()
        return img, band_tracker.track(img, fovy)

    bar = "=" * 66

    # ---- 1. 정지 스윕 -----------------------------------------------------
    # 카메라 가로 화각이 약 63° 이므로 ±31° 밖은 화면을 벗어난다. ±25 까지만 본다.
    print("\n{}\n 1. 정지 스윕 - 흔들림 없는 조건에서 색추종 정확도\n{}".format(bar, bar))
    print("   세로 화각 {:.0f}°, 해상도 {}x{}".format(fovy, args.width, args.height))
    print("\n   {:>5} {:>8} {:>8} {:>7} {:>6} {:>8}".format(
        "거리", "정답방위", "비전", "오차", "픽셀", "거리추정"))
    print("   " + "-" * 50)

    cam_p = m.data.cam_xpos[cam_id].copy()
    errs, dist_err, sweep_frames = [], [], []
    for dist in (1.0, 1.6, 2.4):
        for ang in (-25, -12, 0, 12, 25):
            a = np.radians(ang)
            # 카메라 정면(+X) 기준으로 각도를 주어 사람을 배치한다.
            px = cam_p[0] + dist * np.cos(a)
            py = cam_p[1] + dist * np.sin(a)
            m.data.mocap_pos[0] = [px, py, 0.0]
            mujoco.mj_forward(m.model, m.data)

            gt_b, _ = truth_bearing(m.data, cam_id, [px, py, BAND_Z])
            img, res = look()
            sweep_frames.append(annotate(img, res))

            if res is None:
                print("   {:5.1f} {:8.1f} {:>8} {:>7} {:6d} {:>8}".format(
                    dist, gt_b, "미검출", "-", 0, "-"))
                continue
            e = res["bearing_deg"] - gt_b
            errs.append(e)
            dist_err.append(res["distance_m"] - dist)
            print("   {:5.1f} {:8.1f} {:8.1f} {:+7.2f} {:6d} {:6.2f}m".format(
                dist, gt_b, res["bearing_deg"], e, res["pixels"], res["distance_m"]))

    if errs:
        errs = np.array(errs)
        print("\n   방위 오차: 평균 {:+.2f}°  절대평균 {:.2f}°  최대 {:.2f}°".format(
            errs.mean(), np.abs(errs).mean(), np.abs(errs).max()))
        de = np.array(dist_err)
        print("   거리 추정 오차: 절대평균 {:.2f} m  (거친 값이다, 방위만큼 못 믿는다)".format(
            np.abs(de).mean()))

    contact_sheet(sweep_frames, 5, os.path.join(args.out, "bearing_sweep.png"))

    # ---- 2. 보행 중 -------------------------------------------------------
    print("\n{}\n 2. 보행 중 - 카메라가 흔들리는 조건에서 살아남는가\n{}".format(bar, bar))
    m.full_reset()
    m.direct_head = False
    m.data.mocap_pos[0] = [1.90, 0.15, 0.0]
    m.commands[0] = m.COMMANDS_RANGE_X[1]      # 최대 전진
    mujoco.mj_forward(m.model, m.data)

    ctrl_dt = m.sim_dt * m.decimation
    n = int(args.seconds / ctrl_dt)
    every = max(1, int(round(1.0 / (10.0 * ctrl_dt))))   # 10Hz 로 본다

    walk_err, miss, seen, clipped_n, walk_frames = [], 0, 0, 0, []
    for k in range(n):
        m.imitation_i = (m.imitation_i + m.phase_frequency_factor) % m.PRM.nb_steps_in_period
        ph = m.imitation_i / m.PRM.nb_steps_in_period * 2 * np.pi
        m.imitation_phase = np.array([np.cos(ph), np.sin(ph)])

        action = m.policy.infer(m.get_obs(m.data, m.commands))
        m.last_last_last_action = m.last_last_action.copy()
        m.last_last_action = m.last_action.copy()
        m.last_action = action.copy()
        m.motor_targets = m.default_actuator + action * m.action_scale
        lim = m.max_motor_velocity * ctrl_dt
        m.motor_targets = np.clip(m.motor_targets,
                                  m.prev_motor_targets - lim, m.prev_motor_targets + lim)
        m.prev_motor_targets = m.motor_targets.copy()
        m.data.ctrl = m.motor_targets.copy()
        for _ in range(m.decimation):
            mujoco.mj_step(m.model, m.data)

        if k % every:
            continue
        seen += 1
        pxy = m.data.mocap_pos[0]
        gt_b, _ = truth_bearing(m.data, cam_id, [pxy[0], pxy[1], BAND_Z])
        img, res = look()
        if len(walk_frames) < 12:
            walk_frames.append(annotate(img, res))
        if res is None:
            miss += 1
        else:
            walk_err.append(res["bearing_deg"] - gt_b)
            clipped_n += int(res["clipped"])

    contact_sheet(walk_frames, 4, os.path.join(args.out, "bearing_walk.png"))

    print("   {:.0f}초 전진, {}프레임 (10Hz)".format(args.seconds, seen))
    print("   검출률 {:.0f}%  (미검출 {}프레임)".format(
        100 * (seen - miss) / max(seen, 1), miss))
    if walk_err:
        we = np.array(walk_err)
        print("   방위 오차: 평균 {:+.2f}°  절대평균 {:.2f}°  최대 {:.2f}°".format(
            we.mean(), np.abs(we).mean(), np.abs(we).max()))
        print("   검출된 것 중 화면에 잘린 것 {}/{} ({:.0f}%)".format(
            clipped_n, len(walk_err), 100 * clipped_n / len(walk_err)))
    print("\n   저장: {}  (bearing_sweep.png, bearing_walk.png)".format(args.out))


if __name__ == "__main__":
    main()
