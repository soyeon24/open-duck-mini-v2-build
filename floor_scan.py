r"""바닥 평면으로 앞이 얼마나 비었는지 잰다 (단안 free-space). **mujoco 의존성 없음.**

`band_tracker.py` 와 같이 Raspberry Pi 로 옮겨갈 파일이다. 입력은 RGB 배열과
IMU 에서 읽은 자세 두 개(피치/롤)뿐이다.

원리: 바닥에 서 있는 물체는 **바닥과 맞닿은 밑변**이 이미지에 찍힌다. 그 밑변의
세로 위치를 알면 카메라 높이와 각도로 거리가 나온다. 화면을 세로로 쪼개 각 구역마다
"바닥이 아닌 것이 처음 나타나는 높이"를 찾으면, 방위별 여유거리 — 즉 싸구려
라이다 한 줄이 된다.

**바닥을 채도로 가른다.** 밝기로 가르면 그림자가 장애물이 된다. 이 씬에서 잰 값:
    바닥(그림자·격자선 포함) 채도 0.003 | 하늘 0.112 | 장애물 0.21 | 사람 0.27
바닥만 0 에 붙어 있어서 0.08 로 자르면 깨끗하게 갈린다.
⚠ **이 문턱값은 이 씬 것이다.** 실제 바닥은 채도가 0 이 아니다. 현장에서 같은 방식
(영역별 채도 측정)으로 다시 잡을 것. 방법은 그대로 쓰되 숫자는 믿지 말 것.

롤은 마스크를 통째로 되돌려 지운다. 걷는 동안 롤이 14° 나 흔들려서(capture_head_cam.py
측정값) 보정 없이는 지평선이 기울어 거리 계산이 통째로 틀어진다.
"""

import numpy as np
from PIL import Image

from band_tracker import rgb_to_hsv

# 이보다 채도가 낮으면 바닥으로 본다. 위 주석의 측정값에서 온 숫자다.
FLOOR_SAT_MAX = 0.08

# 한 구역에서 이 비율 이상이 '바닥 아님' 이어야 그 줄을 장애물 밑변으로 인정한다.
# 렌더 노이즈 픽셀 몇 개에 속지 않으려는 것.
SECTOR_HIT_FRAC = 0.25

# 지평선 근처는 거리가 발산한다. 이보다 멀면 그냥 "비었다" 로 본다.
MAX_RANGE_M = 4.0


def min_visible_range(fovy_deg, cam_height_m, pitch_down_deg):
    """이보다 가까운 바닥은 **화면 아래로 벗어나 보이지 않는다.**

    화면 맨 아랫줄이 바라보는 지점이 곧 최소 가시거리다. 이 안쪽으로 들어온
    장애물은 카메라에서 사라지고, free_space 는 그 구역을 "최소 가시거리만큼
    비었다" 로 보고한다 — 실제보다 멀다고 답하는 쪽이라 위험한 방향의 오차다.

    그래서 회피는 **이 거리에 닿기 전에** 결정해야 한다. clearance 를 이 값보다
    넉넉히 크게 잡을 것. 카메라를 더 숙이면 줄어들지만 멀리를 못 보게 된다.
    """
    depression = pitch_down_deg + fovy_deg / 2.0
    if depression <= 0.5:
        return float("inf")
    return cam_height_m / np.tan(np.radians(depression))


def _derotate(mask, roll_deg):
    """롤만큼 마스크를 되돌려 지평선을 수평으로 만든다."""
    if abs(roll_deg) < 0.5:
        return mask
    im = Image.fromarray((mask * 255).astype(np.uint8))
    # expand=False 로 크기를 유지한다. 가장자리에 생기는 빈 곳은 0(바닥)이 되는데,
    # 바닥으로 오인해도 가장자리라 구역 판정에 거의 영향이 없다.
    im = im.rotate(-roll_deg, resample=Image.NEAREST, expand=False, fillcolor=0)
    return np.asarray(im) > 127


def free_space(rgb, fovy_deg, cam_height_m, pitch_down_deg, roll_deg=0.0,
               n_sectors=15, floor_sat_max=FLOOR_SAT_MAX):
    """방위별 여유거리를 낸다.

    pitch_down_deg : 카메라 시선이 수평보다 아래로 내려간 각도 (아래가 양수).
                     시뮬에서는 cam_xmat, 실기에서는 IMU 에서 얻는다.
    roll_deg       : 카메라 롤. 지평선이 기운 각도.
    cam_height_m   : 카메라의 지면 높이.

    반환: (bearings_deg, free_m)  — 둘 다 길이 n_sectors 의 배열.
          bearings_deg 는 왼쪽이 양수 (band_tracker 와 같은 부호).
          free_m 은 그 방향으로 장애물 없이 갈 수 있는 거리. 지평선까지 비었으면
          MAX_RANGE_M.
    """
    hsv = rgb_to_hsv(rgb)
    not_floor = hsv[..., 1] > floor_sat_max
    not_floor = _derotate(not_floor, roll_deg)

    hgt, wid = not_floor.shape
    f_px = (hgt / 2.0) / np.tan(np.radians(fovy_deg) / 2.0)

    edges = np.linspace(0, wid, n_sectors + 1).astype(int)
    bearings = np.zeros(n_sectors)
    free = np.full(n_sectors, MAX_RANGE_M)

    for i in range(n_sectors):
        c0, c1 = edges[i], edges[i + 1]
        uc = (c0 + c1) / 2.0
        bearings[i] = -np.degrees(np.arctan2(uc - wid / 2.0, f_px))

        sub = not_floor[:, c0:c1]
        frac = sub.mean(axis=1)
        rows = np.nonzero(frac >= SECTOR_HIT_FRAC)[0]
        if rows.size == 0:
            continue                      # 이 구역은 지평선까지 비었다

        # 바닥에서 위로 올라가며 처음 만나는 것 = 가장 아래쪽 줄 = 가장 가까운 것.
        v = float(rows.max())
        # 시선축 아래로 내려간 각 + 마운트 하향각 = 수평에서의 내림각
        alpha = np.degrees(np.arctan2(v + 0.5 - hgt / 2.0, f_px))
        depression = pitch_down_deg + alpha
        if depression <= 0.5:
            continue                      # 지평선 위 — 거리가 발산한다
        d = cam_height_m / np.tan(np.radians(depression))
        free[i] = min(d, MAX_RANGE_M)

    return bearings, free


def steer(bearings, free, target_bearing_deg, clearance_m,
          robot_half_width_m=0.12, turn_penalty=0.02, prefer_side=0):
    """갈 방향을 고른다. 표적 쪽에 가까우면서 충분히 빈 방위를 찾는다.

    clearance_m : 이만큼은 비어 있어야 갈 만하다고 본다. 보통 표적까지의 거리보다
                  짧게 준다 — 어차피 갈 표적을 장애물로 보고 피하면 영영 못 간다.
    turn_penalty: 표적에서 1° 벗어날 때마다 깎는 점수. 크면 고집스럽게 직진하고
                  작으면 쉽게 옆길로 샌다.
    robot_half_width_m: 로봇 반폭. 지나가려면 좌우로 이만큼은 같이 비어야 한다.

    prefer_side : 지난번에 정한 우회 방향 (+1 왼쪽, -1 오른쪽, 0 없음).

    반환: (고른 방위[deg], 막혔는지 여부, 이번에 정한 우회 방향)
          막혔으면 그나마 가장 열린 방위와 True 를 돌려준다 — 그쪽으로 몸을 돌려
          두면 다음 스텝에 길이 열릴 수 있다. 전진할지 말지는 호출한 쪽이 정한다.
          우회 방향은 호출한 쪽이 들고 있다가 다음 호출에 도로 넣어 주어야 한다.
    """
    # 로봇은 점이 아니다. 한 방위가 비어 있어도 그 옆이 막혀 있으면 어깨가 걸린다.
    # **필요한 각폭은 거리에 따라 달라진다** — 같은 반폭이라도 가까우면 넓은 각을,
    # 멀면 좁은 각을 차지한다. 고정각(12°)을 쓰면 1 m 에서 42 cm 통로를 요구하게
    # 되어, 20 cm 도 안 되는 오리가 지나갈 수 있는 길을 전부 막혔다고 본다.
    half = np.degrees(np.arctan2(robot_half_width_m, np.maximum(free, 0.3)))
    eff = np.array([
        free[np.abs(bearings - b) <= h].min() for b, h in zip(bearings, half)
    ])

    # 표적 쪽이 열려 있으면 곧장 간다. 우회는 여기서 해제된다.
    ti = int(np.argmin(np.abs(bearings - target_bearing_deg)))
    if eff[ti] >= clearance_m:
        return target_bearing_deg, False, 0

    ok = eff >= clearance_m
    if prefer_side:
        # **정한 쪽을 지킨다.** 이게 없으면 몸을 틀자마자 장애물이 정면에서
        # 빠지고, 그러면 다시 표적 쪽으로 꺾어서 결국 장애물 옆구리로 박는다.
        # 매 프레임 새로 고르는 반사적 회피가 실패하는 전형적인 이유다.
        same = ok & (np.sign(bearings - target_bearing_deg) == prefer_side)
        if same.any():
            ok = same

    if not ok.any():
        return float(bearings[int(np.argmax(eff))]), True, prefer_side

    score = eff[ok] - turn_penalty * np.abs(bearings[ok] - target_bearing_deg) * MAX_RANGE_M
    chosen = float(bearings[ok][int(np.argmax(score))])
    side = int(np.sign(chosen - target_bearing_deg)) or prefer_side
    return chosen, False, side
