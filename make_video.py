r"""색추종이 도는 걸 눈으로 보는 영상(GIF)을 만든다.

왼쪽은 오리를 따라다니는 3인칭, 오른쪽은 머리 카메라다. 오른쪽 위에 색추종이
찍은 방위각과 정답 방위각을 같이 띄운다 — 두 숫자가 붙어 다니면 되는 것이다.

사람은 mocap 이라 좌우로 움직인다. 표적이 가만히 있으면 방위각이 안 변해서
추종이 되는 건지 그냥 정면을 찍는 건지 구분이 안 되기 때문이다.

ffmpeg/imageio 가 없어서 PIL 로 GIF 를 굽는다 (새 의존성 없이). 용량이 크므로
패널을 줄여 놨다. 더 길게 뽑으려면 --seconds 를, 더 선명하게 하려면 --panel 을 키울 것.

제어 루프는 `eval_walk.py` 의 `rollout()` 과 같은 순서다 (50Hz).

    .venv\Scripts\python.exe make_video.py
    .venv\Scripts\python.exe make_video.py --seconds 12 --panel 520 390
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
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

import band_tracker  # noqa: E402
from eval_bearing import truth_bearing, BAND_Z  # noqa: E402
from playground.open_duck_mini_v2.mujoco_infer import MjInfer  # noqa: E402

REFERENCE = "playground/open_duck_mini_v2/data/polynomial_coefficients.pkl"
SCENE = "playground/open_duck_mini_v2/xmls/scene_obstacles.xml"
ONNX = "../from_ubai/fr186_2026_09_12_174638_300482560.onnx"


# 한글 글리프가 있는 폰트를 먼저 찾는다. arial 로 그리면 전부 두부(□)가 된다.
KOREAN_FONTS = ("malgun.ttf", "malgunbd.ttf", "NanumGothic.ttf", "gulim.ttc", "batang.ttc")
FALLBACK_FONTS = ("arial.ttf", "segoeui.ttf", "DejaVuSans.ttf")


def load_font(size):
    """(폰트, 한글 가능 여부) 를 돌려준다. 한글이 안 되면 라벨을 영문으로 쓴다."""
    for name in KOREAN_FONTS:
        try:
            return ImageFont.truetype(name, size), True
        except OSError:
            continue
    for name in FALLBACK_FONTS:
        try:
            return ImageFont.truetype(name, size), False
        except OSError:
            continue
    return ImageFont.load_default(), False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--onnx_model_path", type=str, default=ONNX)
    ap.add_argument("--model_path", type=str, default=SCENE)
    ap.add_argument("--seconds", type=float, default=9.0)
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--panel", type=int, nargs=2, default=[440, 330],
                    help="패널 하나의 가로 세로. 둘을 좌우로 붙인다.")
    ap.add_argument("--out", type=str, default=os.path.join(ROOT, "head_cam_out", "follow.gif"))
    ap.add_argument("--follow", action="store_true",
                    help="추종+회피를 켠다. 사람은 제자리에 두고 오리가 찾아간다.")
    ap.add_argument("--no_avoid", action="store_true", help="회피만 끈다")
    ap.add_argument("--goto", action="store_true",
                    help="자율 이동(N)을 켠다. 돌면서 사람을 찾고 -> 가고 -> 선다.")
    ap.add_argument("--start_yaw", type=float, default=None,
                    help="출발 방위(도). --goto 와 같이 쓴다. 180 이면 등지고 시작")
    ap.add_argument("--person", type=float, nargs=2, default=None,
                    help="사람(=도착점) x y. --goto 면 제자리에 세워 둔다")
    ap.add_argument("--no_ref_range", action="store_true")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    PW, PH = args.panel

    m = MjInfer(args.model_path, REFERENCE, args.onnx_model_path,
                False, not args.no_ref_range)
    m.full_reset()
    m.direct_head = False
    if args.goto:
        m.place(None, args.start_yaw, args.person)
        if args.no_avoid:
            m.avoid = False
        m.start_goto()
    elif args.follow:
        m.follow = True
        if args.no_avoid:
            m.avoid = False
    else:
        m.commands[0] = m.COMMANDS_RANGE_X[1]

    cam_id = m.model.camera("head_cam").id
    fovy = float(m.model.cam_fovy[cam_id])

    # 카메라 렌더러는 머리 시점용과 3인칭용을 따로 둔다 (해상도가 같으니 하나로도
    # 되지만, 나중에 각각 다른 크기로 뽑고 싶을 때를 위해 분리해 둔다).
    rend = mujoco.Renderer(m.model, height=PH, width=PW)
    chase = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(chase)
    # MuJoCo 의 azimuth 는 **시선 방향**이지 카메라 위치가 아니다. 카메라는
    # lookat - distance*forward 에 놓이므로, 오리 뒤(-X)에서 사람 쪽(+X)을 보려면
    # 시선이 +X 여야 하고 그건 azimuth 0 근처다. 180 이나 140 을 주면 카메라가
    # 사람 너머로 넘어가 사람이 전경을 가려버린다 (실제로 그렇게 나왔다).
    chase.azimuth, chase.elevation = 18, -18
    # 거리와 주시점은 매 프레임 오리와 사람을 둘 다 담도록 잡는다. 오리만 크게
    # 잡으면 표적과의 관계가 안 보여서 이 패널이 할 일이 없어진다.

    font, kr = load_font(max(13, PH // 22))
    small, _ = load_font(max(11, PH // 28))
    L = ({"vision": "비전", "truth": "정답", "err": "오차",
          "miss": "미검출", "third": "3인칭"} if kr else
         {"vision": "vision", "truth": "truth", "err": "err",
          "miss": "NOT FOUND", "third": "3rd person"})

    ctrl_dt = m.sim_dt * m.decimation
    n = int(args.seconds / ctrl_dt)
    every = max(1, int(round(1.0 / (args.fps * ctrl_dt))))

    # 사람 위치와 좌우 진폭. 오리가 9초에 약 1.2 m 간다.
    # 진폭을 키우면 방위각 변화는 잘 보이지만, 오리 자체의 요 드리프트가 더해져
    # 표적이 화각(가로 약 ±31°) 밖으로 나가버린다. 2.2 m 에서 ±0.45 m 면 약 ±11°.
    PX0, PY0, SWEEP = 2.2, 0.15, 0.45

    frames = []
    for k in range(n):
        t = k * ctrl_dt

        # 표적을 좌우로 움직인다. 가만히 있으면 방위각이 안 변해서 추종이
        # 되는 건지 그냥 정면을 보는 건지 구분이 안 된다.
        if args.goto:
            pass                       # 도착점은 가만히 있어야 도착점이다
        elif args.follow:
            m.data.mocap_pos[0] = [2.4, 0.9, 0.0]
        else:
            m.data.mocap_pos[0] = [PX0, PY0 + SWEEP * np.sin(2 * np.pi * t / 6.0), 0.0]

        # 제어 한 스텝. 뷰어와 채점 스크립트가 쓰는 **그 함수**다. 예전에는
        # 여기에 루프를 복제해 뒀는데, 그러면 한쪽만 고쳐졌을 때 영상과 측정이
        # 조용히 갈린다.
        m.control_step()
        for _ in range(m.decimation):
            mujoco.mj_step(m.model, m.data)

        if k % every:
            continue

        # --- 머리 카메라 ---
        rend.update_scene(m.data, camera="head_cam")
        head = rend.render().copy()
        res = band_tracker.track(head, fovy)
        p = m.data.mocap_pos[0]
        gt_b, _ = truth_bearing(m.data, cam_id, [p[0], p[1], BAND_Z])

        # --- 3인칭 (오리와 사람을 둘 다 담는다) ---
        duck = m.get_floating_base_qpos(m.data.qpos)[:3]
        sep = float(np.linalg.norm(duck[:2] - p[:2]))
        chase.lookat[:] = 0.5 * (duck + np.array([p[0], p[1], 0.30]))
        # 둘이 가까워져도 화면이 너무 좁아지지 않게 바닥을 깔아 둔다.
        chase.distance = max(2.6, 1.3 + 0.9 * sep)
        rend.update_scene(m.data, camera=chase)
        third = rend.render().copy()

        canvas = Image.new("RGB", (PW * 2, PH), (255, 255, 255))
        canvas.paste(Image.fromarray(third), (0, 0))
        canvas.paste(Image.fromarray(head), (PW, 0))
        dr = ImageDraw.Draw(canvas)

        # 머리 카메라 패널 위에 십자선과 숫자를 얹는다.
        if res is not None:
            u, v = PW + res["u"], res["v"]
            dr.line([(u, 0), (u, PH)], fill=(255, 0, 255), width=2)
            dr.line([(PW, v), (PW * 2, v)], fill=(255, 0, 255), width=2)
            if args.goto:
                phase = {"scan": "사람 찾는 중", "go": "이동 중",
                         "arrived": "도착"}[m.goto_phase]
                txt = "{} {:+5.1f}°   {}".format(
                    L["vision"], res["bearing_deg"], phase if kr else m.goto_phase)
            elif args.follow:
                sd = {1: "<<", -1: ">>", 0: "|"}[m.avoid_side]
                txt = "{} {:+5.1f}°   여유 {:4.1f}m   우회 {}   {}".format(
                    L["vision"], res["bearing_deg"], m.follow_free, sd,
                    "막힘" if m.follow_blocked else "")
            else:
                txt = "{} {:+6.1f}°   {} {:+6.1f}°   {} {:+.1f}°".format(
                    L["vision"], res["bearing_deg"], L["truth"], gt_b,
                    L["err"], res["bearing_deg"] - gt_b)
            col = (20, 20, 20)
        else:
            txt = "{}          {} {:+6.1f}°".format(L["miss"], L["truth"], gt_b)
            col = (200, 0, 0)

        dr.rectangle([PW, 0, PW * 2, 26], fill=(255, 255, 255))
        dr.text((PW + 6, 5), txt, fill=col, font=font)
        dr.rectangle([0, 0, PW, 24], fill=(255, 255, 255))
        dr.text((6, 4), "{}   t={:.1f}s".format(L["third"], t), fill=(20, 20, 20), font=small)
        dr.line([(PW, 0), (PW, PH)], fill=(120, 120, 120), width=2)

        # 화면 아래에 방위각 막대. 숫자보다 눈에 먼저 들어온다.
        cx = PW + PW / 2
        dr.rectangle([PW + 4, PH - 20, PW * 2 - 4, PH - 4], fill=(245, 245, 245))
        dr.line([(cx, PH - 20), (cx, PH - 4)], fill=(150, 150, 150), width=1)
        if res is not None:
            # 가로 화각의 절반(약 31°)을 패널 절반에 대응시킨다.
            half = np.degrees(np.arctan(np.tan(np.radians(fovy / 2)) * PW / PH))
            bx = cx - res["bearing_deg"] / half * (PW / 2 - 6)
            dr.rectangle([bx - 4, PH - 19, bx + 4, PH - 5], fill=(200, 0, 200))

        frames.append(canvas.convert("P", palette=Image.ADAPTIVE, colors=128))

    if not frames:
        print("프레임이 없다")
        return

    frames[0].save(args.out, save_all=True, append_images=frames[1:],
                   duration=int(1000 / args.fps), loop=0, optimize=True)
    mb = os.path.getsize(args.out) / 1e6
    print("\n{}프레임, {:.0f}fps, {}x{}  ->  {}  ({:.1f} MB)".format(
        len(frames), args.fps, PW * 2, PH, args.out, mb))


if __name__ == "__main__":
    main()
