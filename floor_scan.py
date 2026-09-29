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

import heapq

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


class ObstacleMemory:
    """본 장애물을 월드 좌표로 들고 있는다. **카메라 사각에 들어가도 안 사라지게.**

    카메라는 가까운 바닥을 못 본다 (`min_visible_range`, 약 0.54 m). 18 cm 벽에
    30 cm 까지 다가가면 화면 맨 아랫줄이 벽 윗면 높이에 걸려, `free_space` 는
    **벽 너머 바닥을 보고 "1.2 m 비었다" 고 답한다.** 한 프레임만 믿으면 우회를
    풀고 벽으로 직진한다. 실제로 그렇게 벽에 붙어 20초를 밀었다 (2026-09-28,
    eval_goto 0/4).

    그래서 밑변이 화면 안에 보일 때 찍은 위치를 기억한다. 지우는 조건은 하나 —
    **그 자리 바닥이 지금 보이는데 비어 있을 때**다. 사각에 들어가 안 보이는 것과
    치워진 것을 가를 방법이 이것뿐이다. 시간으로 지우면 벽 앞에 오래 서 있다가
    잊고 박는다.

    자세는 시뮬에서는 참값, 실기에서는 오도메트리(자이로 z + 명령 속도 적분)다.
    몇 초짜리 기억이라 드리프트는 문제가 안 된다 — 도착 판정도 같은 가정을 쓴다.
    """

    def __init__(self, keep_m=1.5, record_m=1.0, clear_margin_m=0.10, grid_m=0.02,
                 point_r_m=0.05):
        self.keep_m = keep_m              # 이보다 멀어진 점은 버린다 (지나왔거나 멀다)
        # 이보다 먼 밑변은 새로 넣지 않는다. 멀수록 거리 추정이 앞뒤로 번져서
        # (걷는 중 머리 까딱임), 기둥 하나가 시선 방향으로 40 cm 길쭉하게 기억되고
        # 그걸 크게 돌아가느라 시간을 다 썼다 (2026-09-28). 우회 결정은 1 m 안에서 한다.
        self.record_m = record_m
        self.clear_margin_m = clear_margin_m
        self.grid_m = grid_m              # 같은 칸에 여러 번 찍히면 하나로 친다
        # 점 하나가 가리는 폭. 멀리서 방위 칸마다 한 점씩 찍으면 점 간격이 몇 cm 인데,
        # 다가가면 같은 점들이 넓은 각에 흩어져 칸 사이에 구멍이 난다. 그 구멍으로
        # 벽 너머 바닥값이 새어 나온다. 반경을 줘서 가까울수록 넓은 각을 덮게 한다.
        self.point_r_m = point_r_m
        self.pts = np.zeros((0, 2))

    def reset(self):
        self.pts = np.zeros((0, 2))

    def update(self, cam_xy, heading_deg, bearings, free, blind_m):
        """이번 프레임을 반영하고, 기억을 합친 여유거리를 돌려준다.

        cam_xy      : 카메라의 월드 xy (바닥 투영).
        heading_deg : bearings 의 0 이 가리키는 월드 방위 (몸통 프레임이면 몸통 요).
        bearings, free : free_space 결과. bearings 는 heading 기준으로 옮겨 둔 것.
        blind_m     : min_visible_range. 이보다 가까운 free 값은 측정이 아니라
                      "화면 아래 끝까지 뭔가 있다" 는 하한일 뿐이다.
        """
        cam_xy = np.asarray(cam_xy, dtype=float)
        free = np.asarray(free, dtype=float)
        half = abs(bearings[1] - bearings[0]) / 2.0 if len(bearings) > 1 else 180.0

        def locate(pts):
            rel = pts - cam_xy
            r = np.hypot(rel[:, 0], rel[:, 1])
            b = np.degrees(np.arctan2(rel[:, 1], rel[:, 0])) - heading_deg
            b = (b + 180.0) % 360.0 - 180.0
            idx = np.argmin(np.abs(b[:, None] - bearings[None, :]), axis=1)
            in_fov = np.abs(b - bearings[idx]) <= half
            return r, idx, in_fov

        # 1) 지금 보이는 바닥 위에 있어야 할 점은 지운다. 그 방위의 바닥이
        #    free[idx] 까지 비어 보이는데 점이 그 안쪽(사각 바깥)에 있으면 틀린 기억이다.
        if len(self.pts):
            r, idx, in_fov = locate(self.pts)
            seen_empty = (in_fov & (r > blind_m + 0.02)
                          & (r < free[idx] - self.clear_margin_m))
            self.pts = self.pts[~seen_empty & (r < self.keep_m)]

        # 2) 밑변이 화면 안에 찍힌 것만 새로 넣는다. free 가 사각 경계에 붙어 있으면
        #    밑변은 화면 아래로 빠져 있다 — 실제로는 더 가까울 수 있어 위치를 모른다.
        th = np.radians(heading_deg + np.asarray(bearings))
        new = (free > blind_m + 0.02) & (free < min(self.record_m, MAX_RANGE_M))
        if new.any():
            p = cam_xy + free[new, None] * np.stack([np.cos(th[new]), np.sin(th[new])], 1)
            pts = np.vstack([self.pts, p])
            key = np.round(pts / self.grid_m).astype(np.int64)
            _, keep = np.unique(key, axis=0, return_index=True)
            self.pts = pts[np.sort(keep)]

        # 3) 기억을 방위별로 합친다. 가까운 쪽이 이긴다.
        out = free.copy()
        if len(self.pts):
            rel = self.pts - cam_xy
            r = np.hypot(rel[:, 0], rel[:, 1])
            b = np.degrees(np.arctan2(rel[:, 1], rel[:, 0])) - heading_deg
            b = (b + 180.0) % 360.0 - 180.0
            ext = np.degrees(np.arctan2(self.point_r_m, np.maximum(r, 1e-3)))
            for i, bi in enumerate(bearings):
                hit = np.abs(b - bi) <= half + ext
                if hit.any():
                    out[i] = min(out[i], float(r[hit].min()))
        return out


def detour_waypoint(pts, robot_xy, target_xy, side=0, half_width_m=0.12,
                    margin_m=0.10, target_clear_m=0.45, link_m=0.10):
    """표적까지 직선을 막는 장애물을 돌아갈 경유점을 **월드 좌표로** 고른다.

    `steer` 는 방향만 준다. 정책이 명령한 각도를 덜 따라가면 그 차이가 그대로
    쌓여 모서리를 못 비킨다 — 실제로 −35° 를 시키면 −12° 로 가서 벽 모서리를
    몇 cm 남기고 막혔다 (2026-09-28). 경유점을 월드에 박아 두면 매 스텝 실제
    위치에서 다시 조준하므로, 덜 간 만큼 저절로 더 꺾는다.

    경유점은 **막는 덩어리의 모서리를 접선으로 스치는 점**이다 (아래 주석).
    매 스텝 새 위치에서 다시 잡으므로 로봇은 모서리를 반경 w 로 감아 돈다.
    모서리를 지나 표적까지 직선이 비면 경유점이 저절로 풀린다. 벽의 두께(뒷면)는
    안 보이지만 여유 10 cm 가 그걸 먹는다.

    pts        : ObstacleMemory.pts — 본 장애물 밑변 (N,2)
    side       : 지난번에 고른 쪽 (+1 왼쪽, -1 오른쪽, 0 없음). 매번 새로 고르면
                 두 쪽 길이가 비슷할 때 좌우로 흔들린다.
    target_clear_m : 표적에서 이 반경 안의 점은 표적 자신(사람 발)으로 보고 아예 뺀다.
                 0.35 를 앞뒤 거리로만 재서 뺐더니, 걷는 중 머리 까딱임으로 앞뒤로
                 번진 발 점 몇 개가 "막는 점" 으로 남고, 연결을 타고 사람 발 덩어리
                 전체가 장애물이 되어 경유점이 사람 옆으로 튀었다 (2026-09-28).
                 어차피 0.55 m 에서 서므로 그 안의 장애물은 볼 일이 없다.

    반환: (경유점 xy 또는 None, 고른 쪽). None 이면 표적까지 직선이 비었다.
    """
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    P = np.asarray(robot_xy, dtype=float)
    T = np.asarray(target_xy, dtype=float)
    d = T - P
    L = float(np.hypot(d[0], d[1]))
    if L < 1e-6 or len(pts) == 0:
        return None, 0
    pts = pts[np.hypot(*(pts - T).T) > target_clear_m]
    if len(pts) == 0:
        return None, 0
    u = d / L
    n = np.array([-u[1], u[0]])            # 왼쪽 법선
    rel = pts - P
    s, l = rel @ u, rel @ n
    w = half_width_m + margin_m

    block = (s > 0.0) & (s < L) & (np.abs(l) < w)
    if not block.any():
        return None, 0

    # 막는 점에서 시작해 이어진 점을 모은다. 통로 밖으로 뻗은 벽의 나머지까지
    # 알아야 어디가 끝(모서리)인지 안다.
    member, frontier = block.copy(), block.copy()
    while frontier.any():
        gap = np.hypot(pts[:, None, 0] - pts[None, frontier, 0],
                       pts[:, None, 1] - pts[None, frontier, 1]).min(axis=1)
        new = (gap < link_m) & ~member
        member |= new
        frontier = new

    # 모서리 = 로봇에서 봤을 때 **각이 가장 바깥인** 점 (앞쪽 점만). 경유점은 그
    # 모서리에서 시선(로봇->모서리)에 수직으로 바깥에 찍는다. 그러면 로봇->경유점
    # 직선 전체가 모서리에서 w 만큼 떨어진다 (접선). 모서리와 같은 앞뒤 위치에
    # 옆으로만 띄우면 가는 길이 모서리를 13 cm 로 스쳐 걸렸다 (반폭 12 cm).
    idx = np.nonzero(member & (s > 0.0))[0]
    if len(idx) == 0:
        return None, 0
    ang = np.arctan2(l[idx], s[idx])       # 표적 방향 기준, 왼쪽 +
    cand = {}
    for sd in (+1, -1):
        k = idx[np.argmax(ang)] if sd > 0 else idx[np.argmin(ang)]
        r = pts[k] - P
        rn = float(np.hypot(r[0], r[1]))
        nc = sd * np.array([-r[1], r[0]]) / max(rn, 1e-6)
        # 직선과 모서리 사이 거리가 정확히 w 가 되는 오프셋. 이미 w 안쪽이면 w 로.
        off = w * rn / np.sqrt(rn * rn - w * w) if rn > w * 1.05 else w
        cand[sd] = pts[k] + off * nc

    def detour_len(sd):
        return float(np.hypot(*(cand[sd] - P)) + np.hypot(*(T - cand[sd])))

    order = sorted((+1, -1), key=detour_len)
    # 한 번 고른 쪽은 반대쪽이 확실히 짧을 때만 버린다. 0.15 로는 기억이 조금씩
    # 바뀔 때마다 1초 간격으로 좌우가 뒤집혀 제자리에서 흔들렸다.
    if side in (+1, -1) and detour_len(side) <= detour_len(-side) + 0.5:
        order = [side, -side]
    # 경유점 자리에 다른 장애물이 있으면 그쪽은 못 쓴다. 반폭 전체로 재면 모서리
    # 근처에 번져 찍힌 점 하나에도 "막혔다" 가 나서 좌우가 뒤집혔다. 반의반만 본다.
    for sd in order:
        if np.hypot(*(pts - cand[sd]).T).min() > 0.5 * half_width_m:
            return cand[sd], sd
    return cand[order[0]], order[0]


def _clearance_grid(pts, x0, y0, nx, ny, cell_m, reach_m):
    """칸마다 가장 가까운 장애물 점까지의 거리. reach_m 보다 먼 칸은 inf.

    점마다 주변 원판을 찍어 최솟값을 남긴다. 칸 수 × 점 수 전부를 재는 것보다
    훨씬 싸다 (Pi 에서 돌 파일이라 scipy 거리변환은 안 쓴다).
    """
    dist = np.full(nx * ny, np.inf, dtype=np.float32)
    if len(pts) == 0:
        return dist.reshape(ny, nx)
    k = int(np.ceil(reach_m / cell_m))
    oy, ox = np.mgrid[-k:k + 1, -k:k + 1]
    ox, oy = ox.ravel(), oy.ravel()
    ci = np.floor((pts[:, 0] - x0) / cell_m).astype(int)
    cj = np.floor((pts[:, 1] - y0) / cell_m).astype(int)
    gi = ci[:, None] + ox[None, :]
    gj = cj[:, None] + oy[None, :]
    d = np.hypot(x0 + (gi + 0.5) * cell_m - pts[:, 0:1],
                 y0 + (gj + 0.5) * cell_m - pts[:, 1:2])
    ok = (gi >= 0) & (gi < nx) & (gj >= 0) & (gj < ny) & (d <= reach_m)
    np.minimum.at(dist, (gj[ok] * nx + gi[ok]), d[ok].astype(np.float32))
    return dist.reshape(ny, nx)


def plan_path(pts, robot_xy, target_xy, cell_m=0.04, hard_m=0.16, soft_m=0.26,
              soft_w=1.0, goal_r_m=0.45, escape_m=0.30, pad_m=0.6):
    """기억한 장애물 점 위에 격자를 깔고 A* 로 표적 근처까지 경로를 찾는다.

    `detour_waypoint` 는 막는 덩어리 하나의 모서리만 본다. 벽을 돌자마자 기둥이나
    턱에 막히면 그때 다시 판단하는데, 좁은 씬에서 이게 막다른 곳을 만들었다 —
    벽과 턱 사이(22 cm, 몸통 24 cm)로 경유점을 찍고 거기서 20초를 서 있었다
    (2026-09-28, 출발 +90°/−90°). 경로 전체를 한 번에 찾으면 그 틈이 애초에 안 뚫린다.

    hard_m  : 장애물 점에서 이보다 가까운 칸은 못 간다. 몸통 반폭 12 cm + 걸음
              오차 4 cm. 벽–기둥 틈(0.39 m)은 지나가고 벽–턱 틈(0.22 m)은 막힌다.
              예전 경유점 방식은 12+10 cm 를 써서 벽–기둥 틈도 막혀, 기둥 바깥이나
              턱 너머로 크게 돌았다 (도착 24.8 / 37.8초).
    soft_m  : 이 안쪽은 가까울수록 비싸게 친다. 틈이 있으면 가운데로 지나간다.
    goal_r_m: 표적에서 이 반경 안에 닿으면 끝. 어차피 0.55 m 에서 선다.
    escape_m: 로봇이 이미 hard_m 안에 들어와 있으면 시작 칸부터 막힌다. 그때는 이
              반경 안에서 **장애물에서 멀어지는 칸**만 열어 빠져나오게 한다.

    **모르는 곳은 비었다고 본다.** 카메라는 1 m 안 밑변만 기억에 넣으므로
    (ObstacleMemory.record_m) 멀리 있는 건 가까이 가서야 나타난다. 그래서 매번
    다시 푼다.

    반환: 월드 좌표 경로 (K,2) — 로봇 칸에서 표적 근처까지. 못 찾으면 None.
    """
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    P = np.asarray(robot_xy, dtype=float)
    T = np.asarray(target_xy, dtype=float)
    allp = np.vstack([P[None], T[None], pts]) if len(pts) else np.vstack([P, T])
    x0, y0 = allp.min(axis=0) - pad_m
    x1, y1 = allp.max(axis=0) + pad_m
    nx = int(np.ceil((x1 - x0) / cell_m))
    ny = int(np.ceil((y1 - y0) / cell_m))
    dist = _clearance_grid(pts, x0, y0, nx, ny, cell_m, soft_m)

    cx = x0 + (np.arange(nx) + 0.5) * cell_m
    cy = y0 + (np.arange(ny) + 0.5) * cell_m
    si = min(max(int((P[0] - x0) / cell_m), 0), nx - 1)
    sj = min(max(int((P[1] - y0) / cell_m), 0), ny - 1)
    free = dist >= hard_m
    if not free[sj, si]:
        near = np.hypot(cx[None, :] - P[0], cy[:, None] - P[1]) < escape_m
        free |= near & (dist >= dist[sj, si] - 1e-4)
    cost = 1.0 + soft_w * np.clip((soft_m - dist) / (soft_m - hard_m), 0.0, 1.0)
    to_t = np.hypot(cx[None, :] - T[0], cy[:, None] - T[1])
    goal = free & (to_t <= goal_r_m)
    if not goal.any():
        return None
    hcell = np.maximum(to_t - goal_r_m, 0.0) / cell_m   # 칸 단위, 비용 >= 길이라 허용적

    free_f, cost_f, goal_f, h_f = free.ravel(), cost.ravel(), goal.ravel(), hcell.ravel()
    start = sj * nx + si
    g = np.full(nx * ny, np.inf)
    g[start] = 0.0
    prev = np.full(nx * ny, -1, dtype=np.int64)
    heap = [(h_f[start], start)]
    steps = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
             (1, 1, 1.4142), (1, -1, 1.4142), (-1, 1, 1.4142), (-1, -1, 1.4142)]
    end = -1
    while heap:
        f, u = heapq.heappop(heap)
        if goal_f[u]:
            end = u
            break
        gu = g[u]
        if f - h_f[u] > gu + 1e-9:
            continue
        uj, ui = divmod(u, nx)
        for di, dj, ln in steps:
            vi, vj = ui + di, uj + dj
            if vi < 0 or vi >= nx or vj < 0 or vj >= ny:
                continue
            v = vj * nx + vi
            if not free_f[v]:
                continue
            # 대각선은 양옆 두 칸이 다 비어야 간다. 안 그러면 모서리를 비스듬히 뚫는다.
            if di and dj and not (free_f[uj * nx + vi] and free_f[vj * nx + ui]):
                continue
            gv = gu + ln * 0.5 * (cost_f[u] + cost_f[v])
            if gv < g[v]:
                g[v] = gv
                prev[v] = u
                heapq.heappush(heap, (gv + h_f[v], v))
    if end < 0:
        return None
    path = []
    while end >= 0:
        j, i = divmod(end, nx)
        path.append((cx[i], cy[j]))
        end = prev[end]
    path.reverse()
    path[0] = (P[0], P[1])
    return np.array(path)


def path_carrot(path, robot_xy, look_m=0.30):
    """경로 위에서 로봇보다 look_m 앞의 점. 이걸 경유점으로 쫓는다 (pure pursuit).

    경로 칸을 하나씩 밟게 하면 4 cm 마다 방위가 튀어 요 명령이 떨린다.
    """
    P = np.asarray(robot_xy, dtype=float)
    k = int(np.argmin(np.hypot(*(path - P).T)))
    acc = 0.0
    for i in range(k + 1, len(path)):
        acc += float(np.hypot(*(path[i] - path[i - 1])))
        if acc >= look_m:
            return path[i].copy()
    return path[-1].copy()


def plan_detour(pts, robot_xy, target_xy, direct_m=0.22, target_clear_m=0.45,
                look_m=0.30, **kw):
    """`detour_waypoint` 자리에 쓰는 A* 판. 반환: (쫓을 점, 경로, 막혔나).

    표적까지 직선이 장애물에서 direct_m 이상 떨어져 있으면 (None, None, False) —
    우회가 필요 없다. 막혔는데 경로가 없으면 (None, None, True) 이고, 그때 부르는
    쪽은 예전 모서리 방식(`detour_waypoint`)으로 물러난다.
    (예전 방식의 통로 반폭 12+10 cm 와 같은 값이라 직진 판정은 바뀌지 않는다.)
    표적 반경 target_clear_m 안의 점은 사람 발로 보고 뺀다 (detour_waypoint 와 같은 이유).
    """
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    P = np.asarray(robot_xy, dtype=float)
    T = np.asarray(target_xy, dtype=float)
    pts = pts[np.hypot(*(pts - T).T) > target_clear_m] if len(pts) else pts
    if len(pts) == 0:
        return None, None, False
    d = T - P
    L = float(np.hypot(d[0], d[1]))
    if L < 1e-6:
        return None, None, False
    s = np.clip((pts - P) @ d / (L * L), 0.0, 1.0)
    gap = np.hypot(*(pts - (P + s[:, None] * d)).T)
    if gap.min() >= direct_m:
        return None, None, False
    path = plan_path(pts, P, T, **kw)
    if path is None:
        return None, None, True
    return path_carrot(path, P, look_m), path, True
