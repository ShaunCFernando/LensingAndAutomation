

import numpy as np


# ----------------------------------
#  Temperature Profile
# ----------------------------------

def disk_temperature(r, r_isco, T_max=1.0):

    # Novikov-Thorne temperature profile (normalized).
    r = np.asarray(r, dtype=float)
    T = np.zeros_like(r)
    mask = r >= r_isco
    rr = r[mask]
    ratio = r_isco / rr
    T[mask] = T_max * ratio**(3.0/4.0) * np.sqrt(np.clip(1.0 - np.sqrt(ratio), 0.0, None))
    return T


# ----------------------------------
#  Blackbody Color Approximation
# ----------------------------------

def blackbody_color(T_norm):
    
    """
    The mapping (from cool to hot):
      0.0  → black (no emission)
      0.05 → dark red
      0.2  → orange-red
      0.4  → orange-yellow
      0.6  → yellow-white
      0.8  → white
      1.0  → blue-white
    """
    T = np.clip(T_norm, 0.0, 1.0)

    # Piecewise color ramp along blackbody locus
    # Colors: black → red → orange → yellow → white → blue-white
    r_stops = np.array([0.00, 0.10, 0.25, 0.45, 0.65, 0.85, 1.00])
    R_stops = np.array([0.00, 0.55, 0.90, 1.00, 1.00, 1.00, 0.85])
    G_stops = np.array([0.00, 0.05, 0.20, 0.55, 0.85, 0.95, 0.95])
    B_stops = np.array([0.00, 0.00, 0.00, 0.00, 0.40, 0.90, 1.00])

    R = np.interp(T, r_stops, R_stops)
    G = np.interp(T, r_stops, G_stops)
    B = np.interp(T, r_stops, B_stops)

    return R, G, B


# ----------------------------------
#  Doppler / Gravitational Redshift Factor
# ----------------------------------


def doppler_redshift_sch(r, phi_obs, M):
  
    # Combined Doppler + gravitational redshift factor for the Schwarzschild disk.
    r = np.asarray(r, dtype=float)
    f = np.clip(1.0 - 2.0 * M / r, 0.01, None)

    Omega_K = np.sqrt(M / r**3)          # Keplerian angular velocity

    # Velocity of emitter as seen by ZAMO
    v = r * Omega_K / np.sqrt(f)
    v = np.clip(v, 0.0, 0.999)

    gamma = 1.0 / np.sqrt(1.0 - v**2)   # Lorentz factor

    # Doppler shift depends on viewing angle
    cos_angle = np.sin(phi_obs)          # approaching when cos_angle = +1

    g_dop   = 1.0 / (gamma * (1.0 - v * cos_angle))
    g_grav  = np.sqrt(f)

    return g_dop * g_grav


def doppler_redshift_kerr(r, phi_obs, M, a):
    
    # Energy redshift factor for the Kerr disk (equatorial observer).
    r   = np.asarray(r, dtype=float)
    sqM = np.sqrt(M)
    Omega = sqM / (r**1.5 + a * sqM)    # prograde Keplerian frequency in Kerr

    denom = np.sqrt(np.clip(1.0 - 3.0 * M / r + 2.0 * a * np.sqrt(M / r**3), 0.01, None))

    # Doppler contribution from line-of-sight projection
    cos_angle = np.sin(phi_obs)
    v_los = r * Omega * cos_angle / np.sqrt(np.clip(1.0 - 2.0 * M / r + a**2 / r**2, 0.01, None))

    g = denom / np.clip(1.0 - v_los, 0.01, None)
    return np.clip(g, 0.1, 3.0)


# ----------------------------------
#  Build Disk Mesh for 3D Plotting
# ----------------------------------

def build_disk_mesh(r_isco, r_outer, n_r=80, n_phi=180,
                    M=1.0, a=0.0, include_redshift=True):
  
    # Builds a 2D polar mesh of the accretion disk for 3D visualization.
    r   = np.linspace(r_isco, r_outer, n_r)
    phi = np.linspace(0, 2 * np.pi, n_phi)
    R, PHI = np.meshgrid(r, phi)

    X = R * np.cos(PHI)
    Y = R * np.sin(PHI)
    Z = np.zeros_like(X)

    # Temperature profile
    T = disk_temperature(R, r_isco)

    # Redshift factor
    if include_redshift:
        if a == 0.0:
            g = doppler_redshift_sch(R, PHI, M)
        else:
            g = doppler_redshift_kerr(R, PHI, M, a)
        T_obs = np.clip(T * g, 0.0, 1.0)
    else:
        T_obs = T

    # Convert temperature to RGBA
    Rc, Gc, Bc = blackbody_color(T_obs)
    alpha_map   = np.clip(T_obs * 1.5, 0.0, 0.95)   # transparent where cold

    RGBA = np.stack([Rc, Gc, Bc, alpha_map], axis=-1)

    return X, Y, Z, RGBA, T_obs