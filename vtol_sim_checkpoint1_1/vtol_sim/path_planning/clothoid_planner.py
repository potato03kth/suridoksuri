"""
Clothoid (Euler Spiral) 기반 경로 생성기
==========================================
κ(s) = s / A²  (A² = R_min × L_s)
최대 곡률 = 1/R_min = a_max / v²  (구조적으로 보장)
"""
from __future__ import annotations
import numpy as np
from .base_planner import BasePlanner, Path, PathPoint


class ClothoidPlanner(BasePlanner):
    """
    각 WP 코너를: 직선 → 진입 나선 → 원호 → 이탈 나선 → 직선
    으로 이어 최대 곡률 제한을 구조적으로 만족.
    """

    def __init__(
        self,
        ds: float = 1.0,
        accel_tol: float = 0.9,
        spiral_fraction: float = 0.4,
        min_turn_deg: float = 2.0,
        end_extension: float = 15.0,
    ):
        self.ds = ds
        self.accel_tol = accel_tol
        self.spiral_fraction = spiral_fraction
        self.min_turn_rad = np.deg2rad(min_turn_deg)
        self.end_extension = end_extension

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def plan(self, waypoints_ned: np.ndarray, aircraft_params: dict,
             initial_state: dict | None = None) -> Path:
        import time
        t0 = time.time()

        v = aircraft_params["v_cruise"]
        g = aircraft_params.get("gravity", 9.81)
        a_max_g = aircraft_params["a_max_g"]
        a_max = a_max_g * g * self.accel_tol
        R_min = v * v / a_max

        wps = np.asarray(waypoints_ned, dtype=float)
        if wps.ndim != 2 or wps.shape[1] != 3:
            raise ValueError("waypoints_ned must be (N,3)")

        pts_2d, s_arr, kappa_arr, wp_marks = self._build_2d_path(wps[:, :2], R_min)

        N = len(pts_2d)
        h_arr = self._headings_from_pts(pts_2d)
        kappa_smooth = kappa_arr

        # altitude: linear interpolation between WPs by arc length
        wp_s = [0.0]
        for i in range(1, len(wps)):
            seg = np.linalg.norm(wps[i, :2] - wps[i - 1, :2])
            wp_s.append(wp_s[-1] + seg)
        wp_s = np.array(wp_s)
        alt_arr = np.interp(s_arr, wp_s, wps[:, 2])

        # gamma
        ds_arr = np.diff(s_arr)
        dalt = np.diff(alt_arr)
        gamma_arr = np.zeros(N)
        with np.errstate(invalid="ignore", divide="ignore"):
            g_mid = np.where(ds_arr > 1e-9, np.arctan2(dalt, ds_arr), 0.0)
        gamma_arr[:-1] = g_mid
        gamma_arr[-1] = gamma_arr[-2]

        points = []
        for k in range(N):
            pp = PathPoint(
                pos=np.array([pts_2d[k, 0], pts_2d[k, 1], alt_arr[k]]),
                v_ref=v,
                chi_ref=h_arr[k],
                gamma_ref=float(gamma_arr[k]),
                curvature=float(kappa_smooth[k]),
                s=float(s_arr[k]),
                wp_index=wp_marks.get(k),
            )
            points.append(pp)

        path = Path(
            points=points,
            waypoints_ned=wps,
            total_length=float(s_arr[-1]),
            planning_time=time.time() - t0,
        )
        return path

    # ------------------------------------------------------------------
    # 2-D path builder
    # ------------------------------------------------------------------

    def _build_2d_path(self, wps_2d: np.ndarray, R_min: float):
        """Return (pts, s_arr, kappa_arr, wp_marks dict{pt_idx: wp_idx})."""
        N_wp = len(wps_2d)

        # Pre-compute corner info for interior WPs
        corners: list[dict | None] = [None] * N_wp
        for i in range(1, N_wp - 1):
            d_in = _unit(wps_2d[i] - wps_2d[i - 1])
            d_out = _unit(wps_2d[i + 1] - wps_2d[i])
            alpha = _signed_turn(d_in, d_out)

            if abs(alpha) < self.min_turn_rad:
                corners[i] = None
                continue

            theta_s = abs(alpha) * self.spiral_fraction
            L_s = 2.0 * R_min * theta_s

            # Tangent length
            T_L = self._tangent_length(L_s, R_min, alpha)

            # Cap T_L so corners don't overlap
            seg_in = np.linalg.norm(wps_2d[i] - wps_2d[i - 1])
            seg_out = np.linalg.norm(wps_2d[i + 1] - wps_2d[i])
            # Keep 10 m safety margin
            T_max = 0.45 * min(seg_in, seg_out)
            if T_L > T_max and T_max > 1.0:
                scale = T_max / T_L
                L_s *= scale
                T_L = T_max

            P_entry = wps_2d[i] - d_in * T_L
            P_exit = wps_2d[i] + d_out * T_L
            corners[i] = dict(
                P_entry=P_entry, P_exit=P_exit,
                L_s=L_s, alpha=alpha, wp=wps_2d[i],
                d_in=d_in, d_out=d_out,
            )

        # Assemble segments
        all_pts: list[np.ndarray] = []
        all_kappa: list[float] = []
        wp_marks: dict[int, int] = {}

        # Track "pen position" along each straight
        seg_start = wps_2d[0].copy()

        for i in range(N_wp):
            c = corners[i] if i < N_wp else None

            if c is None:
                # Straight to next WP or end
                if i == N_wp - 1:
                    # Last WP: extend a bit past
                    d_last = _unit(wps_2d[-1] - wps_2d[-2]) if N_wp >= 2 else np.array([1.0, 0.0])
                    seg_end = wps_2d[-1] + d_last * self.end_extension
                else:
                    seg_end = corners[i + 1]["P_entry"] if corners[i + 1] is not None else wps_2d[i + 1]
                    if i + 1 == N_wp - 1:
                        seg_end = wps_2d[i + 1]

                seg_pts, seg_k = self._straight_segment(seg_start, seg_end)
                # Mark WP i (if interior, mark at closest point)
                if i > 0 and i < N_wp - 1 and corners[i] is None:
                    dists = np.linalg.norm(seg_pts - wps_2d[i], axis=1)
                    mark_local = int(np.argmin(dists))
                    wp_marks[len(all_pts) + mark_local] = i
                all_pts.extend(seg_pts.tolist())
                all_kappa.extend(seg_k)

                if c is None and i < N_wp - 1:
                    seg_start = seg_end
            else:
                # Straight from seg_start → P_entry
                seg_pts, seg_k = self._straight_segment(seg_start, c["P_entry"])
                all_pts.extend(seg_pts.tolist())
                all_kappa.extend(seg_k)

                # Corner (spiral-arc-spiral)
                c_pts, c_k, wp_local = self._build_corner(
                    c["P_entry"], c["P_exit"], c["wp"],
                    c["L_s"], R_min, c["alpha"]
                )
                wp_marks[len(all_pts) + wp_local] = i
                all_pts.extend(c_pts.tolist())
                all_kappa.extend(c_k)

                seg_start = c["P_exit"]

        pts = np.array(all_pts, dtype=float)
        kappa = np.array(all_kappa, dtype=float)

        # Build arc lengths
        diffs = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(diffs)])

        return pts, s, kappa, wp_marks

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def _straight_segment(self, P0: np.ndarray, P1: np.ndarray):
        dist = np.linalg.norm(P1 - P0)
        if dist < 1e-6:
            return np.array([P0]), [0.0]
        n = max(2, int(np.ceil(dist / self.ds)) + 1)
        t = np.linspace(0.0, 1.0, n)
        pts = P0[None, :] + t[:, None] * (P1 - P0)[None, :]
        kappa = [0.0] * n
        return pts, kappa

    def _tangent_length(self, L_s: float, R: float, alpha: float) -> float:
        """Compute the tangent length T_L for a clothoid corner."""
        abs_a = abs(alpha)
        # Generate corner pts in local frame (sign=+1)
        pts = self._corner_pts_local(L_s, R, abs_a, sign=1.0)
        # Last point of the corner in local frame
        ex, ey = pts[-1, 0], pts[-1, 1]
        # The exit leg direction in local frame: (cos(abs_a), sin(abs_a))
        # T_L = distance along incoming (+x) direction to intersection with outgoing leg
        # intersection: (T_L, 0) to (ex, ey), along direction (cos a, sin a)
        # T_L - ex + t*cos(a)=0, -ey + t*sin(a)=0 → t=ey/sin(a)
        sa = np.sin(abs_a)
        if abs(sa) < 1e-9:
            return L_s
        t_param = ey / sa
        T_L = ex - t_param * np.cos(abs_a)
        return max(T_L, L_s * 0.1)

    def _corner_pts_local(self, L_s: float, R: float, abs_alpha: float,
                          sign: float = 1.0) -> np.ndarray:
        """
        Generate clothoid corner in local frame.
        Entry at origin, incoming direction = +x.
        Returns (M, 2) array of (x, y).
        """
        theta_s = abs_alpha * self.spiral_fraction
        arc_angle = abs_alpha - 2.0 * theta_s  # remaining angle for circular arc

        n_spiral = max(2, int(np.ceil(L_s / self.ds)))
        s_sp = np.linspace(0.0, L_s, n_spiral)

        # Entry spiral: heading h(s) = sign * s² / (2*R*L_s)
        h_entry = sign * s_sp ** 2 / (2.0 * R * L_s)
        x_e, y_e = _integrate_heading(0.0, 0.0, h_entry, s_sp)

        all_pts = list(zip(x_e, y_e))
        x_cur, y_cur = x_e[-1], y_e[-1]
        h_cur = h_entry[-1]

        # Circular arc (if any)
        if arc_angle > 1e-4:
            arc_len = R * arc_angle
            n_arc = max(2, int(np.ceil(arc_len / self.ds)))
            s_arc = np.linspace(0.0, arc_len, n_arc)
            h_arc = h_cur + sign * s_arc / R
            x_a, y_a = _integrate_heading(x_cur, y_cur, h_arc, s_arc)
            all_pts.extend(zip(x_a[1:], y_a[1:]))
            x_cur, y_cur = x_a[-1], y_a[-1]
            h_cur = h_arc[-1]

        # Exit spiral: heading h(s) = h_cur_start + sign*(2*theta_s*s/L_s - s²/(2*R*L_s))
        # Equivalently: chi_out - sign*(L_s-s)²/(2*R*L_s)
        chi_out = sign * abs_alpha
        s_sx = np.linspace(0.0, L_s, n_spiral)
        h_exit = chi_out - sign * (L_s - s_sx) ** 2 / (2.0 * R * L_s)
        x_x, y_x = _integrate_heading(x_cur, y_cur, h_exit, s_sx)
        all_pts.extend(zip(x_x[1:], y_x[1:]))

        return np.array(all_pts, dtype=float)

    def _build_corner(self, P_entry: np.ndarray, P_exit: np.ndarray,
                      WP: np.ndarray, L_s: float, R: float, alpha: float):
        """
        Build clothoid corner (spiral-arc-spiral) in world NED frame.
        Returns (pts (M,2), kappa list, wp_local_idx int).
        """
        sign = np.sign(alpha)
        abs_a = abs(alpha)
        theta_s = abs_a * self.spiral_fraction
        arc_angle = abs_a - 2.0 * theta_s

        d_in = _unit(WP - P_entry)
        chi_in = np.arctan2(d_in[1], d_in[0])

        n_spiral = max(2, int(np.ceil(L_s / self.ds)))
        s_sp = np.linspace(0.0, L_s, n_spiral)

        # Entry spiral
        h_entry = chi_in + sign * s_sp ** 2 / (2.0 * R * L_s)
        kappa_entry = sign * s_sp / (R * L_s)
        x_e, y_e = _integrate_heading(P_entry[0], P_entry[1], h_entry, s_sp)

        all_pts = list(zip(x_e, y_e))
        all_k = kappa_entry.tolist()
        x_cur, y_cur = x_e[-1], y_e[-1]
        h_cur = h_entry[-1]

        # Circular arc
        if arc_angle > 1e-4:
            arc_len = R * arc_angle
            n_arc = max(2, int(np.ceil(arc_len / self.ds)))
            s_arc = np.linspace(0.0, arc_len, n_arc)
            h_arc = h_cur + sign * s_arc / R
            kappa_arc = [sign / R] * n_arc
            x_a, y_a = _integrate_heading(x_cur, y_cur, h_arc, s_arc)
            all_pts.extend(zip(x_a[1:], y_a[1:]))
            all_k.extend(kappa_arc[1:])
            x_cur, y_cur = x_a[-1], y_a[-1]
            h_cur = h_arc[-1]

        # Exit spiral
        chi_out = chi_in + sign * abs_a
        s_sx = np.linspace(0.0, L_s, n_spiral)
        h_exit = chi_out - sign * (L_s - s_sx) ** 2 / (2.0 * R * L_s)
        kappa_exit = sign * (L_s - s_sx) / (R * L_s)
        x_x, y_x = _integrate_heading(x_cur, y_cur, h_exit, s_sx)
        all_pts.extend(zip(x_x[1:], y_x[1:]))
        all_k.extend(kappa_exit[1:].tolist())

        pts = np.array(all_pts, dtype=float)
        # Mark closest point to WP corner vertex
        dists = np.linalg.norm(pts - WP[:2], axis=1)
        wp_local = int(np.argmin(dists))

        return pts, all_k, wp_local

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    def _headings_from_pts(self, pts: np.ndarray) -> np.ndarray:
        N = len(pts)
        h = np.zeros(N)
        dx = np.diff(pts[:, 0])
        dy = np.diff(pts[:, 1])
        h_mid = np.arctan2(dy, dx)
        h[:-1] = h_mid
        h[-1] = h_mid[-1]
        return h


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def _signed_turn(d_in: np.ndarray, d_out: np.ndarray) -> float:
    """Signed turn angle: left (CCW in NE plane) → negative per base_planner convention."""
    cross = d_in[0] * d_out[1] - d_in[1] * d_out[0]  # z-component
    dot = float(np.clip(np.dot(d_in, d_out), -1.0, 1.0))
    angle = np.arccos(dot)
    # cross > 0: left turn in NE plane → negative curvature
    return -np.sign(cross) * angle if abs(cross) > 1e-9 else 0.0


def _integrate_heading(x0: float, y0: float,
                       h_arr: np.ndarray, s_arr: np.ndarray):
    """Trapezoidal integration of heading profile → (x_arr, y_arr)."""
    dx = np.cos(h_arr)
    dy = np.sin(h_arr)
    ds = np.diff(s_arr)
    x = np.empty(len(s_arr))
    y = np.empty(len(s_arr))
    x[0], y[0] = x0, y0
    x[1:] = x0 + np.cumsum(0.5 * (dx[:-1] + dx[1:]) * ds)
    y[1:] = y0 + np.cumsum(0.5 * (dy[:-1] + dy[1:]) * ds)
    return x, y
