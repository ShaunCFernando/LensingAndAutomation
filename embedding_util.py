"""
embedding_util.py — Shared Flamm's paraboloid surface builder.
Used by visualize_bh.py for 3D plots.
"""
import numpy as np


def build_embedding_surface_full(M, r_max_factor=8.0, n_r=100, n_phi=150):
    rs = 2.0 * M
    r_inner = rs * 1.005
    r_outer = rs * r_max_factor
    # Non-uniform grid: denser near the horizon
    u = np.linspace(np.sqrt(r_inner - rs), np.sqrt(r_outer - rs), n_r)
    r = u**2 + rs
    phi = np.linspace(0, 2*np.pi, n_phi)
    R, PHI = np.meshgrid(r, phi)
    X = R * np.cos(PHI)
    Y = R * np.sin(PHI)

    def flamm_z(r_arr):
        return 2.0 * np.sqrt(np.clip(rs * (r_arr - rs), 0.0, None))

    Z_upper = flamm_z(R)
    Z_lower = -Z_upper
    return {"X": X, "Y": Y, "Z_upper": Z_upper, "Z_lower": Z_lower, "rs": rs}