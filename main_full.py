"""
main_full.py — Full Black Hole Simulation: Schwarzschild + Kerr + Ray Tracing
==============================================================================

Produces 6 output figures:
  1. schwarzschild_shadow.png      — Schwarzschild shadow + photon ring
  2. photon_orbits.png             — Rays color-coded by winding number
  3. kerr_shadow.png               — Kerr shadow (a=0.9M), D-shape asymmetry
  4. spin_comparison.png           — Shadow vs spin: a=0, 0.5, 0.9, 0.998
  5. 3d_combined.png               — 3D funnel + accretion disk + ergosphere
  6. kerr_frame_dragging.png       — Prograde vs retrograde photon orbits
"""

import os
import sys
import numpy as np
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))

from metric import (sch_rs, sch_critical_b, kerr_horizons,
                    kerr_ISCO, kerr_photon_orbit, kerr_ergosphere)
from visualize_bh import (fig_schwarzschild_shadow, fig_photon_orbits,
                           fig_kerr_shadow, fig_comparison,
                           fig_3d_combined, fig_kerr_photon_orbits)

os.makedirs("outputs", exist_ok=True)

M = 1.0   # black hole mass (geometrized units)

print("=" * 65)
print("  Full Black Hole Simulation: Schwarzschild + Kerr Ray Tracing")
print("=" * 65)

# ── Key radii summary ──
rs = sch_rs(M)
print(f"\n  Mass M = {M}")
print(f"  Schwarzschild radius   rs = 2M = {rs:.3f}")
print(f"  Shadow radius          b_c = 3√3 M = {sch_critical_b(M):.3f}")

for a_val, label in [(0.5, "a=0.5M"), (0.9, "a=0.9M"), (0.998, "a=0.998M")]:
    r_out, r_in = kerr_horizons(M, a_val)
    r_isco_pro  = kerr_ISCO(M, a_val, prograde=True)
    r_isco_ret  = kerr_ISCO(M, a_val, prograde=False)
    r_ph_pro    = kerr_photon_orbit(M, a_val, prograde=True)
    r_ph_ret    = kerr_photon_orbit(M, a_val, prograde=False)
    r_ergo      = kerr_ergosphere(np.pi/2, M, a_val)
    print(f"\n  Kerr {label}:")
    print(f"    Outer horizon r+ = {r_out:.3f}M")
    print(f"    Inner horizon r- = {r_in:.3f}M")
    print(f"    Ergosphere (equator) = {r_ergo:.3f}M")
    print(f"    Photon orbit (pro/ret) = {r_ph_pro:.3f}M / {r_ph_ret:.3f}M")
    print(f"    ISCO (pro/ret)         = {r_isco_pro:.3f}M / {r_isco_ret:.3f}M")

print()

# ── Run figures ──
TASKS = [
    ("Figure 1: Schwarzschild shadow",
     lambda: fig_schwarzschild_shadow(M, save="outputs/1_schwarzschild_shadow.png")),
    ("Figure 2: Photon orbit diagram",
     lambda: fig_photon_orbits(M, save="outputs/2_photon_orbits.png")),
    ("Figure 3: Kerr shadow (a=0.9M)",
     lambda: fig_kerr_shadow(M, a=0.9*M, save="outputs/3_kerr_shadow.png")),
    ("Figure 4: Spin comparison grid",
     lambda: fig_comparison(M, a_vals=[0.0, 0.5, 0.9, 0.998],
                            save="outputs/4_spin_comparison.png")),
    ("Figure 5: 3D combined (a=0.7M)",
     lambda: fig_3d_combined(M, a=0.7*M, save="outputs/5_3d_combined.png")),
    ("Figure 6: Kerr frame dragging (a=0.9M)",
     lambda: fig_kerr_photon_orbits(M, a=0.9*M,
                                    save="outputs/6_kerr_frame_dragging.png")),
]

for name, task in TASKS:
    print(f"\n{name}")
    try:
        task()
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback; traceback.print_exc()

print("\n" + "=" * 65)
print("  Done. All outputs in outputs/")
print("=" * 65)