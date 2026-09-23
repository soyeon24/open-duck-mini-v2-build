r"""화살표 키로 몰았을 때 어떻게 걷는지 영상(GIF)으로 뽑는다.

`make_video.py` 는 색추종 데모용이라 오리가 스스로 명령을 만든다. 이건 반대로
**사람이 키를 누르는 쪽**이다. 뷰어에서 키를 누르는 것과 같은 명령을 대본대로
넣고, 화면 위에 지금 눌린 키와 그때까지 쌓인 요 쏠림을 같이 띄운다.

대본을 쓰는 이유는 손으로 누르면 매번 달라져서 정책끼리 비교가 안 되기 때문이다.
같은 대본을 다른 정책에 먹이면 그대로 비교 영상이 된다.

토크 상한을 반드시 명시한다 (`--forcerange`). 씬 XML 은 ±3.23 고정인데 fr186
계열은 ±1.86 으로 학습됐고, 안 맞추면 학습한 적 없는 걸음이 찍힌다. 자세한 것은
SIM_NOTES "fr186 은 요 제어가 무너진 게 아니었다".

제어는 `MjInfer.control_step()` 하나를 쓴다. 여기에 루프를 복제하면 영상과
채점이 조용히 갈린다.

    .venv\Scripts\python.exe make_drive_video.py --forcerange 1.86
    .venv\Scripts\python.exe make_drive_video.py -o from_ubai\fr186_....onnx \
        --ref_range --forcerange 1.86 --out head_cam_out\drive_fr186.gif
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
from PIL import Image, ImageDraw  # noqa: E402

from make_video import load_font  # noqa: E402
from playground.open_duck_mini_v2.mujoco_infer import MjInfer  # noqa: E402

REFERENCE = "playground/open_duck_mini_v2/data/polynomial_coefficients.pkl"
SCENE = "playground/open_duck_mini_v2/xmls/scene_flat_terrain_backlash.xml"
ONNX = "../BEST_WALK_ONNX_2.onnx"

# 대본은 **키 갈래별로 나눈다.** 화살표(↑↓←→)는 몸을 평행이동시키는 키고,
# 선회는 Q/E 다. 둘을 한 영상에 섞으면 "쏠린 것" 과 "돌라고 시킨 것" 이 눈으로
# 구분이 안 된다. 사람 따라가기는 명령을 오리가 만드는 별개 기능이라
# `make_video.py --follow` / `--goto` 쪽이다.
# (라벨, 초, (x, y, theta)) — 값은 각 축 명령 범위에 대한 비율이다.
SCRIPTS = {
    "arrows": [
        ("—",  1.0, (0, 0, 0)),
        ("↑",  5.0, (1, 0, 0)),
        ("↓",  3.0, (-1, 0, 0)),
        ("←",  3.0, (0, 1, 0)),
        ("→",  3.0, (0, -1, 0)),
        ("—",  1.0, (0, 0, 0)),
    ],
    # 방위 유지가 있고 없고를 눈으로 가르려면 ↑ 를 오래 눌러야 한다. arrows 대본은
    # ↑ 가 5초뿐이라 5초짜리 쏠림은 둘 다 작아서 차이가 안 보인다.
    "straight": [
        ("—",  1.0, (0, 0, 0)),
        ("↑", 20.0, (1, 0, 0)),
        ("—",  1.0, (0, 0, 0)),
    ],
    "turn": [
        ("—",  1.0, (0, 0, 0)),
        ("Q",  4.0, (0, 0, 1)),
        ("—",  1.0, (0, 0, 0)),
        ("E",  4.0, (0, 0, -1)),
        ("—",  1.0, (0, 0, 0)),
    ],
}

KEYS = {"arrows": ["↑", "↓", "←", "→"], "straight": ["↑"],
        "turn": ["Q", "E"]}


def yaw_of(q):
    w, x, y, z = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def main():
    ap = argparse.ArgumentParser(description="화살표 제어 영상 뽑기")
    ap.add_argument("-o", "--onnx_model_path", type=str, default=ONNX)
    ap.add_argument("--model_path", type=str, default=SCENE)
    ap.add_argument("--ref_range", action="store_true",
                    help="2026-09-04 이후(dy ±0.111) 정책이면 붙일 것. 원본엔 붙이지 말 것")
    ap.add_argument("--forcerange", type=float, default=None,
                    help="토크 상한[N·m]. 정책이 학습된 값을 줄 것")
    ap.add_argument("--fps", type=float, default=12.0)
    ap.add_argument("--panel", type=int, nargs=2, default=[560, 400])
    ap.add_argument("--out", type=str,
                    default=os.path.join(ROOT, "head_cam_out", "drive.gif"))
    ap.add_argument("--heading_hold", action="store_true",
                    help="방위 유지를 켜고 찍는다 (뷰어 기본값과 같은 상태). "
                         "안 주면 정책 맨몸이 찍힌다")
    ap.add_argument("--script", choices=sorted(SCRIPTS), default="arrows",
                    help="arrows = 전진/후진/게걸음, turn = 제자리 선회(Q/E)")
    ap.add_argument("--en", action="store_true", help="라벨을 영어로")
    args = ap.parse_args()
    # 이 스크립트는 REPO 로 chdir 한 뒤라, 상대경로를 그대로 두면 결과물이
    # Open_Duck_Playground 안에 떨어진다. 부른 사람이 기대하는 곳은 여기다.
    if not os.path.isabs(args.out):
        args.out = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    PW, PH = args.panel
    m = MjInfer(args.model_path, REFERENCE, args.onnx_model_path,
                False, args.ref_range)
    if args.forcerange is not None:
        m.model.actuator_forcerange[:] = np.array([-args.forcerange,
                                                   args.forcerange])
    m.full_reset()
    m.direct_head = False
    m.heading_hold = args.heading_hold

    font, kr = load_font(max(15, PH // 20))
    small, _ = load_font(max(12, PH // 28))
    big, _ = load_font(max(20, PH // 14))
    kr = kr and not args.en
    L = ({"drift": "쏠림", "yaw": "누적 요", "torque": "토크 상한",
          "hold": "방위 유지", "stop": "정지"} if kr else
         {"drift": "drift", "yaw": "yaw turned", "torque": "torque limit",
          "hold": "heading hold", "stop": "stop"})
    # arrows 대본은 요를 시킨 적이 없으니 쌓인 요가 곧 쏠림이다. turn 대본은
    # 도는 게 목적이라 같은 숫자가 "얼마나 돌았나" 가 된다. 라벨을 갈라 준다.
    metric = L["yaw"] if args.script == "turn" else L["drift"]
    keys = KEYS[args.script]

    rend = mujoco.Renderer(m.model, height=PH, width=PW)
    chase = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(chase)
    chase.azimuth, chase.elevation, chase.distance = 130, -18, 1.5

    ctrl_dt = m.sim_dt * m.decimation
    every = max(1, int(round(1.0 / (args.fps * ctrl_dt))))

    base = m.get_floating_base_qpos(m.data.qpos)
    p0, y0 = base[:3].copy(), yaw_of(base[3:7].copy())
    yaw_acc, yaw_prev = 0.0, y0

    frames = []
    k = 0
    for label, secs, (cx, cy, ct) in SCRIPTS[args.script]:
        # 비율 -> 실제 명령. 음수 쪽은 그 축의 하한을 쓴다.
        m.commands[0] = cx * (m.COMMANDS_RANGE_X[1] if cx > 0 else -m.COMMANDS_RANGE_X[0])
        m.commands[1] = cy * (m.COMMANDS_RANGE_Y[1] if cy > 0 else -m.COMMANDS_RANGE_Y[0])
        m.commands[2] = ct * m.COMMANDS_RANGE_THETA[1]
        for _ in range(int(secs / ctrl_dt)):
            m.control_step()
            for _ in range(m.decimation):
                mujoco.mj_step(m.model, m.data)

            yaw_now = yaw_of(m.get_floating_base_qpos(m.data.qpos)[3:7])
            step = (yaw_now - yaw_prev + np.pi) % (2 * np.pi) - np.pi
            yaw_prev = yaw_now
            # arrows 대본에서는 요 명령이 늘 0 이라 그대로 쏠림이 된다.
            # turn 대본에서는 선회 구간만 쌓아 "얼마나 돌았나" 를 보여준다.
            if (ct == 0) == (args.script != "turn"):
                yaw_acc += step
            k += 1
            if k % every:
                continue

            # 카메라는 오리를 따라가되 **출발 방위에 고정**한다. 오리를 따라
            # 돌면 화면 안에서는 늘 똑바로 걷는 것처럼 보여서 쏠림이 안 보인다.
            duck = m.get_floating_base_qpos(m.data.qpos)[:3]
            chase.lookat[:] = duck
            rend.update_scene(m.data, camera=chase)
            img = Image.fromarray(rend.render().copy())
            dr = ImageDraw.Draw(img)

            # 키 표시등
            x = 14
            for key in keys:
                on = key == label
                box = (x, 12, x + 40, 52)
                dr.rectangle(box, fill=(60, 120, 230) if on else (235, 235, 235),
                             outline=(40, 40, 40), width=2)
                dr.text((x + 20, 32), key, font=font, anchor="mm",
                        fill=(255, 255, 255) if on else (120, 120, 120))
                x += 46
            if label == "—":
                dr.text((x + 8, 32), L["stop"], font=small, anchor="lm",
                        fill=(120, 120, 120))

            d = m.get_floating_base_qpos(m.data.qpos)[:3] - p0
            fwd = d[0] * np.cos(y0) + d[1] * np.sin(y0)
            lat = -d[0] * np.sin(y0) + d[1] * np.cos(y0)
            deg = np.rad2deg(yaw_acc)
            dr.text((14, PH - 58),
                    "{} {:+5.1f}°".format(metric, deg), font=big,
                    fill=(200, 40, 40) if (args.script != "turn" and abs(deg) > 15)
                    else (30, 30, 30))
            dr.text((14, PH - 26),
                    "{:+.0f} / {:+.0f} cm   ({} ±{:.2f} N·m, {} {})".format(
                        fwd * 100, lat * 100, L["torque"],
                        args.forcerange if args.forcerange else 3.23,
                        L["hold"], "ON" if args.heading_hold else "OFF"),
                    font=small, fill=(90, 90, 90))
            dr.text((PW - 12, PH - 26), os.path.basename(args.onnx_model_path),
                    font=small, anchor="rs", fill=(90, 90, 90))
            frames.append(img)

    frames[0].save(args.out, save_all=True, append_images=frames[1:],
                   duration=int(1000 / args.fps), loop=0, optimize=True)
    mb = os.path.getsize(args.out) / 1e6
    print("{}  ({} 프레임, {:.1f} MB)".format(args.out, len(frames), mb))
    print("최종 {} {:+.1f}°".format(metric, np.rad2deg(yaw_acc)))


if __name__ == "__main__":
    main()
