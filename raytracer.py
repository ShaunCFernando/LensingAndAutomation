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


def _kerr_dR(r, M, a, xi, eta):
    # dR/dr, used for the second-order radial equation
    P = r**2 + a**2 - a * xi
    return 4.0 * r * P - (2.0 * r - 2.0 * M) * (eta + (xi - a)**2)


def _kerr_dTheta(theta, M, a, xi, eta):
    # dTheta/dtheta, used for the second-order polar equation
    s = np.sin(theta)
    if abs(s) < 1e-6:
        s = 1e-6 if s >= 0 else -1e-6
    c = np.cos(theta)
    return -2.0 * a**2 * c * s + 2.0 * xi**2 * c / s**3


def _kerr_rhs(tau, state, M, a, xi, eta):
    """
    Kerr null geodesic equations in Mino time tau, where d(tau) = d(lambda) / Sigma.

    First-order (Carter) form:   (dr/dtau)^2 = R(r),   (dtheta/dtau)^2 = Theta(theta)
    Differentiating once gives:   d2r/dtau2 = R'(r)/2,  d2theta/dtau2 = Theta'(theta)/2

    We integrate the second-order form. At a turning point (closest approach,
    or a maximum/minimum in theta) R or Theta hits zero but its derivative
    does not, so the photon turns around on its own. The first-order form
    r' = +/- sqrt(R) can't do this without tracking the sign by hand.

    state = [r, theta, phi, t, lambda, dr/dtau, dtheta/dtau]
    """
    r, theta, phi, t, lam, pr, pth = state

    Sigma = r**2 + (a * np.cos(theta))**2
    Delta = r**2 - 2.0 * M * r + a**2
    Delta = Delta if abs(Delta) > 1e-12 else 1e-12
    P     = r**2 + a**2 - a * xi
    sin2  = max(np.sin(theta)**2, 1e-12)

    dphi = -(a - xi / sin2) + a * P / Delta
    dt   = -a * (a * sin2 - xi) + (r**2 + a**2) * P / Delta

    return [pr,
            pth,
            dphi,
            dt,
            Sigma,                                   # d(lambda)/d(tau)
            0.5 * _kerr_dR(r, M, a, xi, eta),
            0.5 * _kerr_dTheta(theta, M, a, xi, eta)]


def trace_kerr(M, a, xi, eta, r_start, theta_obs=np.pi/2,
               n_steps=4000, lam_max=200.0, theta_dir=-1):
    """
    Trace a photon with impact parameters (xi, eta) inward from r_start.

    xi  = L_z / E   (sign sets prograde > 0 / retrograde < 0 for a > 0)
    eta = Carter constant / E^2   (eta = 0 keeps an equatorial photon equatorial)

    Integration stops when the photon crosses the outer horizon (captured),
    escapes back out past 2 * r_start, or has travelled lam_max in affine
    parameter (same meaning as before the Mino-time rewrite).
    """
    r_outer, _ = kerr_horizons_local(M, a)

    R0  = _kerr_R(r_start, M, a, xi, eta)
    Th0 = _kerr_Theta(theta_obs, M, a, xi, eta)
    if R0 < 0:
        raise ValueError(f"Photon with xi={xi}, eta={eta} cannot exist at r={r_start} (R(r) < 0).")

    pr0  = -np.sqrt(R0)                                 # moving inward
    pth0 = (theta_dir if theta_dir != 0 else -1) * np.sqrt(max(Th0, 0.0))

    state0 = [r_start, theta_obs, 0.0, 0.0, 0.0, pr0, pth0]

    def horizon_event(tau, s, *args):
        return s[0] - r_outer * 1.01
    horizon_event.terminal, horizon_event.direction = True, -1

    def escape_event(tau, s, *args):
        return s[0] - 2.0 * r_start
    escape_event.terminal, escape_event.direction = True, 1

    def affine_limit_event(tau, s, *args):
        return s[4] - lam_max
    affine_limit_event.terminal, affine_limit_event.direction = True, 1

    # Mino time is bounded by the affine-limit event; this is just a safety cap.
    tau_cap = 50.0 * lam_max / max(r_outer, 1e-3)**2

    sol = solve_ivp(
        fun          = _kerr_rhs,
        t_span       = (0.0, tau_cap),
        y0           = state0,
        args         = (M, a, xi, eta),
        method       = "DOP853",
        rtol         = 1e-10, atol = 1e-10,
        events       = [horizon_event, escape_event, affine_limit_event],
        dense_output = True,
    )

    tau = np.linspace(0.0, sol.t[-1], n_steps)
    y   = sol.sol(tau)
    r_arr, th_arr, phi_arr, t_arr, lam_arr = y[0], y[1], y[2], y[3], y[4]

    captured = len(sol.t_events[0]) > 0

    # Convert to Cartesian (Boyer-Lindquist treated as spherical for plotting)
    x = r_arr * np.sin(th_arr) * np.cos(phi_arr)
    y_ = r_arr * np.sin(th_arr) * np.sin(phi_arr)
    z = r_arr * np.cos(th_arr)

    dphi = abs(phi_arr[-1] - phi_arr[0])
    n_winds = dphi / (2.0 * np.pi)

    return {
        "r": r_arr, "theta": th_arr, "phi": phi_arr, "t": t_arr, "lam": lam_arr,
        "x": x, "y": y_, "z": z,
        "captured": captured,
        "escaped": len(sol.t_events[1]) > 0,
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