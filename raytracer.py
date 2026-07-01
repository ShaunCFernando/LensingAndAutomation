import numpy as np
from scipy.integrate import solve_ivp


# ----------------------------------
#  Schwarzschild Null Geodesic
# ----------------------------------

def _sch_rhs(lam, state, M, b):
    r, phi, pr, _ = state
    r = max(r, 2.0 * M * 1.001)   # stay outside horizon

    inv_r2 = 1.0 / r**2
    inv_r3 = inv_r2 / r
    inv_r4 = inv_r3 / r

    dVdr = b**2 * (-2.0 * inv_r3 + 6.0 * M * inv_r4)
    d_pr = -0.5 * dVdr

    dphi = b * inv_r2                   

    return [pr, dphi, d_pr, 0.0]


def _sch_horizon_event(lam, state, M, b):
    return state[0] - 2.0 * M * 1.005

_sch_horizon_event.terminal  = True
_sch_horizon_event.direction = -1


def trace_schwarzschild(M, b, r_start, n_steps=3000, lam_max=500.0):

    # Initial radial velocity: photon falling inward
    Veff0 = (1.0 - 2.0 * M / r_start) * (b**2 / r_start**2)
    pr0_sq = 1.0 - Veff0
    if pr0_sq < 0:
        pr0_sq = 0.0
    pr0 = -np.sqrt(pr0_sq)   # negative = moving inward

    state0 = [r_start, 0.0, pr0, b / r_start**2]

    sol = solve_ivp(
        fun     = _sch_rhs,
        t_span  = (0.0, lam_max),
        y0      = state0,
        args    = (M, b),
        method  = "RK45",
        t_eval  = np.linspace(0.0, lam_max, n_steps),
        rtol    = 1e-8, atol = 1e-8,
        events  = [_sch_horizon_event],
        dense_output = False
    )

    r   = sol.y[0]
    phi = sol.y[1]
    x   = r * np.cos(phi)
    y   = r * np.sin(phi)

    captured = len(sol.t_events[0]) > 0

    # Count approximate half-orbits
    dphi_total = abs(phi[-1] - phi[0])
    n_half_orbits = dphi_total / np.pi

    return {
        "r": r, "phi": phi, "x": x, "y": y,
        "captured": captured,
        "n_half_orbits": n_half_orbits,
        "b": b, "M": M
    }


def sch_shadow_image(M, D_obs, n_pixels=200, fov_rs=7.0):
    b_crit = 3.0 * np.sqrt(3.0) * M

    # Sky coordinate grid (in units of M)
    half = fov_rs * M
    alpha_arr = np.linspace(-half, half, n_pixels)
    beta_arr  = np.linspace(-half, half, n_pixels)
    A, B = np.meshgrid(alpha_arr, beta_arr)
    impact = np.sqrt(A**2 + B**2)

    image = np.where(impact < b_crit, 0.0, 1.0)

    # Photon ring
    ring_width = 0.35 * M
    photon_ring = (impact > b_crit - ring_width) & (impact < b_crit + ring_width * 0.5)
    image[photon_ring] = 0.85   # bright ring

    return image, alpha_arr / M, beta_arr / M, b_crit / M


# ----------------------------------
#  Kerr Null Geodesic — Carter Equations
# ----------------------------------

def _kerr_R(r, M, a, xi, eta):
    Delta = r**2 - 2.0 * M * r + a**2
    P     = r**2 + a**2 - a * xi
    return P**2 - Delta * (eta + (xi - a)**2)


def _kerr_Theta(theta, M, a, xi, eta):
    cos2 = np.cos(theta)**2
    sin2 = max(np.sin(theta)**2, 1e-10)
    return eta - cos2 * (xi**2 / sin2 - a**2)


def _kerr_rhs(lam, state, M, a, xi, eta, sign_r, sign_th):
    r, theta, phi, t = state

    Sigma = r**2 + (a * np.cos(theta))**2
    Delta = r**2 - 2.0 * M * r + a**2
    P     = r**2 + a**2 - a * xi

    R_val  = _kerr_R(r, M, a, xi, eta)
    Th_val = _kerr_Theta(theta, M, a, xi, eta)

    # Clamp to zero (numerical noise can make them slightly negative)
    dr_dlam    = sign_r  * np.sqrt(max(R_val,  0.0)) / max(Sigma, 1e-8)
    dth_dlam   = sign_th * np.sqrt(max(Th_val, 0.0)) / max(Sigma, 1e-8)

    sin2 = max(np.sin(theta)**2, 1e-10)
    dDelta = max(Delta, 1e-10)

    dphi_dlam  = (-(a - xi / sin2) + a * P / dDelta) / max(Sigma, 1e-8)
    dt_dlam    = (-a * (a * sin2 - xi) + (r**2 + a**2) * P / dDelta) / max(Sigma, 1e-8)

    return [dr_dlam, dth_dlam, dphi_dlam, dt_dlam]


def trace_kerr(M, a, xi, eta, r_start, theta_obs=np.pi/2,
               n_steps=4000, lam_max=200.0):
    r_outer, _ = kerr_horizons_local(M, a)

    # Initial theta velocity from Carter constant
    Th0 = _kerr_Theta(theta_obs, M, a, xi, eta)
    dth0 = -np.sqrt(max(Th0, 0.0)) / (r_start**2 + (a * np.cos(theta_obs))**2)

    # Initial r velocity (falling inward)
    R0   = _kerr_R(r_start, M, a, xi, eta)
    dr0  = -np.sqrt(max(R0, 0.0)) / (r_start**2 + (a * np.cos(theta_obs))**2)

    # Signs track direction (+1 increasing, -1 decreasing)
    sign_r  = [-1]   # mutable list for closure
    sign_th = [np.sign(dth0) if dth0 != 0 else -1]

    def rhs_adaptive(lam, state):
        return _kerr_rhs(lam, state, M, a, xi, eta, sign_r[0], sign_th[0])

    def horizon_event(lam, state):
        return state[0] - r_outer * 1.005

    horizon_event.terminal  = True
    horizon_event.direction = -1

    state0 = [r_start, theta_obs, 0.0, 0.0]

    # Integrate in short segments to flip signs at turning points
    r_path   = [r_start]
    th_path  = [theta_obs]
    phi_path = [0.0]
    t_path   = [0.0]
    captured = False

    lam     = 0.0
    state   = state0[:]
    dlam    = lam_max / n_steps
    prev_dr = dr0
    prev_dth = dth0

    for _ in range(n_steps):
        if lam >= lam_max:
            break

        dr_dlam  = sign_r[0]  * np.sqrt(max(_kerr_R(_clamp_r(state[0], r_outer), M, a, xi, eta), 0.0)) \
                   / max(state[0]**2 + (a*np.cos(state[1]))**2, 1e-8)
        dth_dlam = sign_th[0] * np.sqrt(max(_kerr_Theta(state[1], M, a, xi, eta), 0.0)) \
                   / max(state[0]**2 + (a*np.cos(state[1]))**2, 1e-8)

        # Detect sign flips at turning points
        if prev_dr * dr_dlam < 0:
            sign_r[0] *= -1
        if prev_dth * dth_dlam < 0:
            sign_th[0] *= -1

        prev_dr  = dr_dlam
        prev_dth = dth_dlam

        # Euler step (fast; RK4 below for accuracy)
        new_state = _rk4_step(rhs_adaptive, lam, state, dlam)

        lam   += dlam
        state  = new_state

        r_path.append(state[0])
        th_path.append(state[1])
        phi_path.append(state[2])
        t_path.append(state[3])

        if state[0] <= r_outer * 1.01:
            captured = True
            break
        if state[0] >= r_start * 2.0:
            break   # escaped far enough

    r_arr   = np.array(r_path)
    th_arr  = np.array(th_path)
    phi_arr = np.array(phi_path)

    # Convert to Cartesian (equatorial projection)
    x = r_arr * np.sin(th_arr) * np.cos(phi_arr)
    y = r_arr * np.sin(th_arr) * np.sin(phi_arr)
    z = r_arr * np.cos(th_arr)

    dphi = abs(phi_arr[-1] - phi_arr[0])
    n_winds = dphi / (2.0 * np.pi)

    return {
        "r": r_arr, "theta": th_arr, "phi": phi_arr,
        "x": x, "y": y, "z": z,
        "captured": captured,
        "n_windings": n_winds,
        "xi": xi, "eta": eta, "a": a, "M": M,
    }


def kerr_shadow_image(M, a, theta_obs=np.pi/2, n_pixels=220, fov_M=7.5):
    from metric import kerr_shadow_params

    r_outer, _ = kerr_horizons_local(M, a)

    half  = fov_M * M
    alpha = np.linspace(-half, half, n_pixels)
    beta  = np.linspace(-half, half, n_pixels)
    A, B  = np.meshgrid(alpha, beta)

    sin_obs = np.sin(theta_obs)
    cos_obs = np.cos(theta_obs)

    # Impact parameters for each pixel
    XI  = -A * sin_obs
    ETA = B**2 + (A**2 - a**2) * cos_obs**2   # at θ_obs = π/2 this simplifies

    # Check for photon capture

    # Get shadow boundary from the critical curve
    alpha_crit, beta_crit_upper = kerr_shadow_params(M, a, theta_obs)

    image = np.ones((n_pixels, n_pixels))   # 1 = sky, 0 = shadow

    # For each pixel determine if it is inside the shadow
    if len(alpha_crit) > 3:

        # Build closed curve: upper and lower beta
        a_all  = np.concatenate([alpha_crit, alpha_crit[::-1]])
        b_all  = np.concatenate([beta_crit_upper, -beta_crit_upper[::-1]])

        # Point-in-polygon test using winding number
        for iy in range(n_pixels):
            for ix in range(n_pixels):
                ax, bx = A[iy, ix], B[iy, ix]
                if _point_in_curve(ax, bx, a_all, b_all):
                    image[iy, ix] = 0.0
    else:

        # Fallback: simple circular shadow
        b_crit = 3.0 * np.sqrt(3.0) * M
        dist   = np.sqrt(A**2 + B**2)
        image[dist < b_crit] = 0.0

    # Photon ring
    if len(alpha_crit) > 3:

        # Dilate the shadow slightly for the photon ring
        for iy in range(n_pixels):
            for ix in range(n_pixels):
                if image[iy, ix] == 0.0:
                    continue
                ax, bx = A[iy, ix], B[iy, ix]

                # Distance to shadow boundary (approximate)
                dists = np.sqrt((alpha_crit - ax)**2 + (beta_crit_upper - abs(bx))**2)
                if dists.min() < 0.3 * M:
                    image[iy, ix] = 0.85  # photon ring bright

    return image, alpha / M, beta / M


# ----------------------------------
#  Helpers
# ----------------------------------

def kerr_horizons_local(M, a):

    ## Wrapper so raytracer doesn't import metric directly in every function.
    disc = max(M**2 - a**2, 0.0)
    return M + np.sqrt(disc), M - np.sqrt(disc)


def _clamp_r(r, r_outer):
    return max(r, r_outer * 1.002)


def _rk4_step(f, t, y, h):

    ## Classic 4th-order Runge-Kutta single step.
    y  = np.asarray(y, dtype=float)
    k1 = np.asarray(f(t,       y))
    k2 = np.asarray(f(t + h/2, y + h/2 * k1))
    k3 = np.asarray(f(t + h/2, y + h/2 * k2))
    k4 = np.asarray(f(t + h,   y + h   * k3))
    return list(y + h/6 * (k1 + 2*k2 + 2*k3 + k4))


def _point_in_curve(px, py, cx, cy):

    ## Ray-casting point-in-polygon test.
    n = len(cx)
    inside = False
    j = n - 1
    for i in range(n):
        xi_, yi_ = cx[i], cy[i]
        xj_, yj_ = cx[j], cy[j]
        if ((yi_ > py) != (yj_ > py)) and (px < (xj_ - xi_) * (py - yi_) / (yj_ - yi_ + 1e-12) + xi_):
            inside = not inside
        j = i
    return inside


def batch_trace_schwarzschild(M, b_array, r_start=50.0):

    ## Trace multiple photon rays through Schwarzschild spacetime efficiently.
    results = []
    for b in b_array:
        try:
            res = trace_schwarzschild(M, b, r_start, n_steps=2000, lam_max=300.0)
            results.append(res)
        except Exception:
            results.append(None)
    return results