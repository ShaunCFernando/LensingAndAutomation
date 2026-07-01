import numpy as np


# ----------------------------------
#  Schwarzschild Black Hole
# ----------------------------------

def sch_rs(M):
    return 2.0 * M


def sch_metric(r, M):
    f = 1.0 - 2.0 * M / np.clip(r, 1e-8, None)
    g_tt   = -f
    g_rr   =  1.0 / np.clip(f, 1e-10, None)
    g_thth =  r**2
    g_phph =  r**2          # at θ = π/2, sin²θ = 1
    return g_tt, g_rr, g_thth, g_phph


def sch_lapse(r, M):
    return np.sqrt(np.clip(1.0 - 2.0 * M / r, 0.0, None))


def sch_Veff(r, M, L, kind="timelike"):
    eps = 1.0 if kind == "timelike" else 0.0
    f   = np.clip(1.0 - 2.0 * M / r, 0.0, None)
    return f * (eps + L**2 / r**2)


def sch_photon_sphere(M):
    return 3.0 * M


def sch_critical_b(M):
    """
    A photon with b < b_crit is captured; b > b_crit escapes.
    """
    return 3.0 * np.sqrt(3.0) * M


def sch_shadow_radius(M, D_obs):
    return sch_critical_b(M) / D_obs


# ----------------------------------
#  Kerr Black Hole
# ----------------------------------


def kerr_Sigma(r, theta, M, a):
    return r**2 + (a * np.cos(theta))**2


def kerr_Delta(r, M, a):
    return r**2 - 2.0 * M * r + a**2


def kerr_horizons(M, a):
    """
    Returns (r_outer, r_inner)
    """
    discriminant = np.clip(M**2 - a**2, 0.0, None)
    sqrt_disc = np.sqrt(discriminant)
    return M + sqrt_disc, M - sqrt_disc


def kerr_ergosphere(theta, M, a):
    cos2 = np.cos(theta)**2
    return M + np.sqrt(np.clip(M**2 - a**2 * cos2, 0.0, None))


def kerr_metric_components(r, theta, M, a):
    """
    Full Kerr metric tensor components in Boyer-Lindquist coordinates.

    The metric:
      g_tt   = -(1 - 2Mr/Σ)
      g_rr   = Σ/Δ
      g_thth = Σ
      g_phph = (r² + a² + 2Ma²r sin²θ/Σ) sin²θ
      g_tph  = -2Mar sin²θ / Σ        [frame dragging]
    """
    Sigma = kerr_Sigma(r, theta, M, a)
    Delta = kerr_Delta(r, M, a)
    sin2  = np.sin(theta)**2

    g_tt   = -(1.0 - 2.0 * M * r / Sigma)
    g_rr   = Sigma / np.clip(Delta, 1e-10, None)
    g_thth = Sigma
    g_phph = (r**2 + a**2 + 2.0 * M * a**2 * r * sin2 / Sigma) * sin2
    g_tph  = -2.0 * M * a * r * sin2 / Sigma    # = g_phi_t

    return {"g_tt": g_tt, 
            "g_rr": g_rr, 
            "g_thth": g_thth,
            "g_phph": g_phph, 
            "g_tph": g_tph}


def kerr_frame_drag_omega(r, theta, M, a):
    """
    Frame-dragging angular velocity Ω(r, θ) = -g_tφ / g_φφ.

    This is the angular velocity of Zero Angular Momentum Observers (ZAMOs):
    observers who fall freely from infinity with L = 0, yet are dragged
    around the black hole at angular velocity Ω.

    As r → r+, Ω → a / (r+² + a²) = Ω_H (horizon angular velocity).
    As r → ∞,  Ω → 0 (flat spacetime, no dragging).
    """
    g = kerr_metric_components(r, theta, M, a)
    return -g["g_tph"] / np.clip(g["g_phph"], 1e-10, None)


def kerr_ISCO(M, a, prograde=True):
    """
    Innermost Stable Circular Orbit (ISCO) for the Kerr metric.

    Uses the Bardeen-Press-Teukolsky formula (exact analytic result):

      r_ISCO = M { 3 + Z₂ ∓ √[(3 - Z₁)(3 + Z₁ + 2Z₂)] }

    where ∓ is - for prograde, + for retrograde orbits, and:
      Z₁ = 1 + (1 - a²/M²)^(1/3) [(1 + a/M)^(1/3) + (1 - a/M)^(1/3)]
      Z₂ = √(3a²/M² + Z₁²)

    Physical meaning:
    - Prograde orbits co-rotate with the black hole → frame dragging
      helps support the orbit → ISCO moves inward as spin increases.
    - Retrograde orbits counter-rotate → frame dragging fights the orbit
      → ISCO moves outward.
    """
    aM  = np.clip(a / M, -1.0 + 1e-6, 1.0 - 1e-6)   # spin parameter a/M ∈ (-1, 1)
    Z1  = 1.0 + (1.0 - aM**2)**(1/3) * ((1.0 + aM)**(1/3) + (1.0 - aM)**(1/3))
    Z2  = np.sqrt(3.0 * aM**2 + Z1**2)
    sign = -1.0 if prograde else 1.0
    return M * (3.0 + Z2 + sign * np.sqrt((3.0 - Z1) * (3.0 + Z1 + 2.0 * Z2)))


def kerr_photon_orbit(M, a, prograde=True):
    """
    Radius of the (unstable) circular photon orbit in the equatorial plane.

    Analytic formula:
      r_ph = 2M { 1 + cos[ (2/3) arccos(∓a/M) ] }

    where ∓ is - for prograde, + for retrograde.

    At a = 0: r_ph = 3M (Schwarzschild photon sphere).
    At a = M, prograde: r_ph = M (sits at the horizon in extremal limit).
    """
    aM   = np.clip(a / M, -1.0 + 1e-6, 1.0 - 1e-6)
    sign = -1.0 if prograde else 1.0
    return 2.0 * M * (1.0 + np.cos(2.0 / 3.0 * np.arccos(sign * aM)))


def kerr_shadow_params(M, a, theta_obs=np.pi/2):
    """
    Compute the Kerr black hole shadow boundary parameters.

    The shadow is defined by photons on unstable spherical orbits.
    In Boyer-Lindquist coords these photons have impact parameters (ξ, η):

      ξ = (r_ph² * (3M - r_ph) + a²(r_ph + M)) / (a(r_ph - M))
      η = r_ph³ * (4Ma² - r_ph(r_ph - 3M)²) / (a²(r_ph - M)²)

    These are the celestial coordinates on the observer's sky.
    Returns arrays of (alpha, beta) sky coordinates tracing the shadow edge.
    """
    r_ph_arr = np.linspace(
        kerr_photon_orbit(M, a, prograde=True)  + 0.001,
        kerr_photon_orbit(M, a, prograde=False) - 0.001,
        500
    )

    xi_arr  = []
    eta_arr = []

    for r_ph in r_ph_arr:
        D   = kerr_Delta(r_ph, M, a)
        num_xi  = (r_ph**2 * (3*M - r_ph) + a**2 * (r_ph + M))
        den_xi  = a * (r_ph - M) + 1e-10
        xi  = num_xi / den_xi

        num_eta = r_ph**3 * (4*M*a**2 - r_ph*(r_ph - 3*M)**2)
        den_eta = (a*(r_ph - M))**2 + 1e-10
        eta = num_eta / den_eta

        if eta >= 0:
            xi_arr.append(xi)
            eta_arr.append(eta)

    xi_arr  = np.array(xi_arr)
    eta_arr = np.array(eta_arr)

    # Celestial coords on observer's sky at inclination theta_obs
    sin_obs = np.sin(theta_obs)
    alpha   = -xi_arr / sin_obs
    beta_sq = np.clip(eta_arr + a**2 * np.cos(theta_obs)**2 - xi_arr**2 / np.tan(theta_obs)**2
                      if theta_obs != np.pi/2 else eta_arr, 0.0, None)
    beta    = np.sqrt(beta_sq)

    return alpha, beta