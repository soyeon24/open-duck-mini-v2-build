r"""오리가 걷는 동안 머리 카메라가 무엇을 보는지 뽑고, 얼마나 흔들리는지 잰다.

사람을 보고 찾아가려면 카메라 영상이 쓸 만해야 하는데, 이 로봇은 걸을 때 몸통이
피치로 출렁이고 머리도 같이 출렁인다. 프레임마다 지평선이 기울면 "이미지 아래쪽에서
바닥색이 아닌 픽셀 = 장애물" 같은 방법이 전부 깨진다. 뷰어로 보면 "좀 흔들리는 것
같다" 까지밖에 안 나오므로, 여기서 **도(degree) 로 잰다.**

제어 루프는 `eval_walk.py` 의 `rollout()` 과 같은 순서다 (50Hz, action_scale 0.25,
서보 속도 제한 5.24 rad/s). 루프를 다르게 만들면 뷰어에서 보는 것과 다른 영상이
나와서 측정이 쓸모없어진다.

씬은 `scene_obstacles.xml` (재생 전용). 학습 씬과 바닥·마찰·home 은 같고, 카메라와
장애물과 사람(mocap, 발목 형광밴드)이 추가돼 있다.

    .venv\Scripts\python.exe capture_head_cam.py
    .venv\Scripts\python.exe capture_head_cam.py --seconds 6 --cmd 1 0 0
    .venv\Scripts\python.exe capture_head_cam.py --out C:\some\dir

라이브로 돌려보고 싶으면 이 스크립트가 아니라 기존 뷰어에 씬만 갈아끼우면 된다:
    cd Open_Duck_Playground
    ..\.venv\Scripts\python.exe playground\open_duck_mini_v2\mujoco_infer.py ^
        -o ..\from_ubai\fr186_2026_09_12_174638_300482560.onnx --ref_range ^
        --model_path playground\open_duck_mini_v2\xmls\scene_obstacles.xml
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
from PIL import Image  # noqa: E402

from playground.open_duck_mini_v2.mujoco_infer import MjInfer  # noqa: E402

REFERENCE = "playground/open_duck_mini_v2/data/polynomial_coefficients.pkl"
SCENE = "playground/open_duck_mini_v2/xmls/scene_obstacles.xml"
ONNX = "../from_ubai/fr186_2026_09_12_174638_300482560.onnx"
FALLEN = 0.5  # eval_walk.py 와 같은 판정선 (gravity z)


def cam_pitch_roll_deg(data, cam_id):
    """카메라의 월드 피치/롤. 피치는 시선이 수평에서 얼마나 위아래로 벗어났는지,
    롤은 지평선이 얼마나 기울었는지. 둘 다 0 이면 완벽히 수평인 영상이다."""
    R = data.cam_xmat[cam_id].reshape(3, 3)
    fwd = -R[:, 2]          # 카메라는 자기 -Z 를 본다
    up = R[:, 1]
    pitch = np.degrees(np.arcsin(np.clip(fwd[2], -1, 1)))
    # 롤: 카메라 up 벡터가 월드 수직면에서 얼마나 돌아갔나
    right = R[:, 0]
    roll = np.degrees(np.arctan2(right[2], up[2]))
    return pitch, roll


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--onnx_model_path", type=str, default=ONNX)
    ap.add_argument("--model_path", type=str, default=SCENE)
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--cmd", type=float, nargs=3, default=[1.0, 0.0, 0.0],
                    help="정규화 명령 (전진, 게걸음, 회전). 각각 -1..1")
    ap.add_argument("--fps", type=float, default=10.0, help="프레임 저장 주기")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--out", type=str, default=os.path.join(ROOT, "head_cam_out"))
    ap.add_argument("--no_ref_range", action="store_true",
                    help="fr186 이전 정책을 쓸 때만. 기본은 레퍼런스 정합 범위.")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    m = MjInfer(args.model_path, REFERENCE, args.onnx_model_path,
                False, not args.no_ref_range)
    m.full_reset()
    m.direct_head = False      # 머리는 정책에 맡긴다 (학습 조건이 그렇다)

    cx, cy, ct = args.cmd
    m.commands[0] = cx * (m.COMMANDS_RANGE_X[1] if cx > 0 else -m.COMMANDS_RANGE_X[0])
    m.commands[1] = cy * (m.COMMANDS_RANGE_Y[1] if cy > 0 else -m.COMMANDS_RANGE_Y[0])
    m.commands[2] = ct * m.COMMANDS_RANGE_THETA[1]

    cam_id = m.model.camera("head_cam").id
    renderer = mujoco.Renderer(m.model, height=args.height, width=args.width)

    ctrl_dt = m.sim_dt * m.decimation           # 0.02 s (50Hz)
    n = int(args.seconds / ctrl_dt)
    every = max(1, int(round(1.0 / (args.fps * ctrl_dt))))

    frames, pitches, rolls = [], [], []
    fell_at = None

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
        m.motor_targets = np.clip(
            m.motor_targets, m.prev_motor_targets - lim, m.prev_motor_targets + lim
        )
        m.prev_motor_targets = m.motor_targets.copy()
        m.data.ctrl = m.motor_targets.copy()

        for _ in range(m.decimation):
            mujoco.mj_step(m.model, m.data)

        # 흔들림은 매 제어 스텝마다 잰다 (프레임 저장보다 촘촘하게).
        p, r = cam_pitch_roll_deg(m.data, cam_id)
        pitches.append(p)
        rolls.append(r)

        if fell_at is None and float(m.get_gravity(m.data)[-1]) < FALLEN:
            fell_at = k * ctrl_dt

        if k % every == 0:
            renderer.update_scene(m.data, camera="head_cam")
            img = renderer.render().copy()
            frames.append(img)
            Image.fromarray(img).save(
                os.path.join(args.out, f"frame_{len(frames):03d}.png"))

    # ── 컨택트 시트: 흔들림을 한눈에 보려면 프레임을 나란히 놔야 한다 ──
    if frames:
        cols = min(4, len(frames))
        rows = (len(frames) + cols - 1) // cols
        h, w, _ = frames[0].shape
        sheet = np.full((rows * h, cols * w, 3), 255, np.uint8)
        for i, f in enumerate(frames):
            rr, cc = divmod(i, cols)
            sheet[rr * h:(rr + 1) * h, cc * w:(cc + 1) * w] = f
        Image.fromarray(sheet).save(os.path.join(args.out, "contact_sheet.png"))

    pitches, rolls = np.array(pitches), np.array(rolls)
    print(f"\n명령 (전진,게걸음,회전) = {args.cmd}   {args.seconds}s   프레임 {len(frames)}장")
    print(f"저장: {args.out}")
    if fell_at is not None:
        print(f"⚠ {fell_at:.2f}s 에 넘어졌다 — 아래 숫자는 넘어지는 동작까지 포함한 값이다.")
    print("\n  카메라 흔들림 (걷는 내내)")
    print(f"    피치  평균 {pitches.mean():+6.1f}°   진폭(p2p) {np.ptp(pitches):5.1f}°"
          f"   표준편차 {pitches.std():4.1f}°")
    print(f"    롤    평균 {rolls.mean():+6.1f}°   진폭(p2p) {np.ptp(rolls):5.1f}°"
          f"   표준편차 {rolls.std():4.1f}°")
    fovy = m.model.cam_fovy[cam_id]
    print(f"\n  세로 화각이 {fovy:.0f}° 다. 피치 진폭 {np.ptp(pitches):.1f}° 는 "
          f"화면 높이의 {np.ptp(pitches)/fovy*100:.0f}% 만큼 위아래로 쓸린다는 뜻이다.")


if __name__ == "__main__":
    main()
