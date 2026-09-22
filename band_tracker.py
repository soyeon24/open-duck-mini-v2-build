r"""발목 형광밴드를 찾아 방위각(bearing)을 내놓는다. **mujoco 의존성 없음.**

이 파일만 Raspberry Pi 로 옮겨가면 된다. 입력은 RGB 넘파이 배열(H,W,3, uint8)
하나뿐이고, 시뮬이든 실제 카메라든 똑같이 동작한다. mujoco 를 import 하는 순간
이식할 때 다시 짜야 하므로 여기엔 넣지 않는다.

왜 색추종인가: 걷는 동안 카메라 피치가 18.5°, 롤이 14.1° 흔들린다(capture_head_cam.py
측정값). 지평선 기준 방법은 이 흔들림에 다 깨지지만, 채도 높은 색 덩어리는 화면이
기울어도 그대로 남는다. 흔들림에 강한 게 이 방식을 고른 이유다.

밴드가 양 발목에 하나씩 둘이다. 둘을 따로 보지 않고 **한 덩어리의 무게중심**을 쓴다 —
그게 사람의 중심선이고, 우리가 향하고 싶은 방향이다.

실기 이식 메모: OpenCV 가 있으면 `cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)` 가 훨씬 빠르다.
단 OpenCV 의 H 는 0~179, S/V 는 0~255 다. 아래 정규화(0~1) 값에 각각 179, 255 를 곱할 것.
실제 형광밴드는 조명에 따라 색이 밀리므로 HUE/SAT/VAL 은 현장에서 다시 잡아야 한다.
"""

import numpy as np

# 형광 초록 밴드의 HSV 범위 (0~1 정규화).
# 시뮬 밴드는 rgba=(0.15, 1.0, 0.1) → 색상 약 117° = 0.324. 여유를 두고 잡았다.
# 씬에 다른 초록이 없어서 넓게 잡아도 안전하다. 실제 환경에서는 좁혀야 한다.
HUE_RANGE = (0.25, 0.42)
SAT_MIN = 0.40
VAL_MIN = 0.20

# 이보다 픽셀이 적으면 노이즈로 보고 버린다. 640x480 기준.
MIN_PIXELS = 25

# 거리 추정에 쓰는 실제 가로 폭 [m]. **밴드 하나가 아니라 두 밴드를 합친 덩어리**의
# 바깥쪽 끝에서 끝까지다. 시뮬 기준 다리 간격 0.18 + 밴드 지름 0.11 = 0.29.
# (밴드 하나 지름 0.11 을 쓰면 거리가 2.7 배 작게 나온다. 실제로 그렇게 틀렸었다.)
# 사람마다 다리 간격이 다르고 걸을 때 벌어졌다 모였다 하므로, 이 값은 정확할 수
# 없다. 거리는 "가깝다/멀다" 판정에만 쓰고 제어에 직접 물리지 말 것.
BAND_SPAN_M = 0.29


def rgb_to_hsv(rgb):
    """(H,W,3) uint8 RGB -> (H,W,3) float HSV, 전부 0~1. matplotlib 없이 numpy 만."""
    a = rgb.astype(np.float32) / 255.0
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mx = a.max(-1)
    mn = a.min(-1)
    d = mx - mn

    h = np.zeros_like(mx)
    # d == 0 이면 무채색이라 색상이 정의되지 않는다. 0 으로 두면 빨강으로 오인되는데
    # 어차피 채도 문턱에서 걸러지므로 문제되지 않는다.
    nz = d > 1e-6
    ri = nz & (mx == r)
    gi = nz & (mx == g) & ~ri
    bi = nz & (mx == b) & ~ri & ~gi
    h[ri] = ((g - b)[ri] / d[ri]) % 6
    h[gi] = ((b - r)[gi] / d[gi]) + 2
    h[bi] = ((r - g)[bi] / d[bi]) + 4
    h /= 6.0

    s = np.where(mx > 1e-6, d / np.maximum(mx, 1e-6), 0.0)
    return np.stack([h, s, mx], -1)


def band_mask(rgb):
    """밴드 색에 해당하는 픽셀의 불리언 마스크."""
    hsv = rgb_to_hsv(rgb)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return (h >= HUE_RANGE[0]) & (h <= HUE_RANGE[1]) & (s >= SAT_MIN) & (v >= VAL_MIN)


def track(rgb, fovy_deg, min_pixels=MIN_PIXELS):
    """RGB 한 장에서 밴드를 찾는다.

    fovy_deg 는 **세로** 화각이다 (MuJoCo 의 camera fovy 와 같은 정의).
    가로 화각은 종횡비로 유도되므로 따로 받지 않는다.

    반환: dict 또는 None (밴드가 안 보이면)
        bearing_deg : 화면 중앙 기준 좌우 각도. **왼쪽이 양수** (요 명령 부호와 같다)
        elev_deg    : 위아래 각도. 위가 양수
        u, v        : 무게중심 픽셀 좌표
        pixels      : 밴드 픽셀 수
        width_px    : 밴드 덩어리의 가로 폭 (거리 추정에 쓴다)
        distance_m  : 거친 거리 추정. 밴드가 잘리거나 겹치면 크게 틀린다
        clipped     : 덩어리가 화면 가장자리에 닿았다 = 값이 편향돼 있다
    """
    m = band_mask(rgb)
    n = int(m.sum())
    if n < min_pixels:
        return None

    hgt, wid = m.shape
    ys, xs = np.nonzero(m)
    u, v = xs.mean(), ys.mean()

    # 초점거리를 픽셀로. 세로 화각이 기준이므로 세로 해상도로 계산한다.
    f_px = (hgt / 2.0) / np.tan(np.radians(fovy_deg) / 2.0)

    # 이미지 +x 는 오른쪽이고 카메라 +X 도 오른쪽이다. 오른쪽에 있는 표적은
    # 오른쪽으로 돌아야 하고, 요 명령은 왼쪽이 양수다 → 부호를 뒤집는다.
    bearing = -np.degrees(np.arctan2(u - wid / 2.0, f_px))
    elev = -np.degrees(np.arctan2(v - hgt / 2.0, f_px))

    # 가로 폭은 양쪽 밴드를 합친 덩어리의 폭이다 (BAND_SPAN_M 과 같은 정의).
    # 한쪽 밴드가 가려지거나 화면 밖으로 잘리면 폭이 절반이 되어 거리가 두 배로
    # 튄다. 아래 clipped 플래그가 서면 그 값은 버리는 게 낫다.
    width_px = float(xs.max() - xs.min() + 1)
    distance = f_px * BAND_SPAN_M / max(width_px, 1.0)

    # 덩어리가 화면 가장자리에 닿아 있으면 잘린 것이다 — 무게중심도 거리도 편향된다.
    clipped = bool(xs.min() == 0 or xs.max() == wid - 1
                   or ys.min() == 0 or ys.max() == hgt - 1)

    return {
        "bearing_deg": float(bearing),
        "elev_deg": float(elev),
        "u": float(u),
        "v": float(v),
        "pixels": n,
        "width_px": width_px,
        "distance_m": float(distance),
        "clipped": clipped,
    }
