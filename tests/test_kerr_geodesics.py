"""
Physics checks for the Kerr photon integrator in raytracer.trace_kerr.

Run with:  python -m pytest tests/      (or)      python tests/test_kerr_geodesics.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from metric import kerr_photon_orbit, kerr_horizons          # noqa: E402
from raytracer import trace_kerr                              # noqa: E402

M = 1.0
R_START = 30.0


def _closest_approach_analytic(a, xi):
    """Largest real root of R(r) = 0 for an equatorial photon (eta = 0)."""
    # R(r) = (r^2 + a^2 - a xi)^2 - (r^2 - 2Mr + a^2)(xi - a)^2, expanded in r
    L2 = (xi - a)**2
    c = a**2 - a * xi
    coeffs = [1.0, 0.0, 2.0 * c - L2, 2.0 * M * L2, c**2 - a**2 * L2]
    roots = np.roots(coeffs)
    real = roots[np.abs(roots.imag) < 1e-9].real
    return real.max()


def _critical_xi(a, prograde):
    """Impact parameter of the unstable circular photon orbit (capture boundary)."""
    # Independent of metric.py on purpose: Bardeen (1973) in its original form
    r = kerr_photon_orbit(M, a, prograde=prograde)
    Delta = r**2 - 2 * M * r + a**2
    return (M * (r**2 - a**2) - r * Delta) / (a * (r - M))


def test_wide_ray_turns_around_and_escapes():
    """The original bug: rays froze at closest approach instead of escaping."""
    res = trace_kerr(M, 0.9, xi=8.0, eta=0.0, r_start=R_START)
    assert not res["captured"]
    assert res["escaped"]
    assert res["r"][-1] > R_START


def test_small_impact_parameter_is_captured():
    res = trace_kerr(M, 0.9, xi=2.0, eta=0.0, r_start=R_START)
    assert res["captured"]
    r_outer, _ = kerr_horizons(M, 0.9)
    assert res["r"][-1] < 1.02 * r_outer


def test_closest_approach_matches_analytic_root():
    for a, xi in [(0.0, 8.0), (0.5, 7.0), (0.9, 6.0), (0.9, -9.0)]:
        res = trace_kerr(M, a if a != 0 else 1e-9, xi=xi, eta=0.0, r_start=R_START)
        expected = _closest_approach_analytic(a, xi)
        assert abs(res["r"].min() - expected) < 1e-3, (a, xi, res["r"].min(), expected)


def test_schwarzschild_limit_closest_approach():
    """a -> 0: closest approach solves r^3 - b^2 r + 2 M b^2 = 0."""
    b = 8.0
    roots = np.roots([1.0, 0.0, -b**2, 2.0 * M * b**2])
    expected = roots[np.abs(roots.imag) < 1e-9].real.max()
    res = trace_kerr(M, 1e-9, xi=b, eta=0.0, r_start=R_START)
    assert abs(res["r"].min() - expected) < 1e-3


def test_capture_boundary_shifts_with_spin():
    """Frame dragging: prograde photons can pass closer than retrograde ones."""
    a = 0.9
    for prograde in (True, False):
        xi_c = _critical_xi(a, prograde)
        just_outside = trace_kerr(M, a, xi=1.01 * xi_c, eta=0.0, r_start=R_START)
        just_inside = trace_kerr(M, a, xi=0.99 * xi_c, eta=0.0, r_start=R_START)
        assert just_outside["escaped"] and not just_outside["captured"], prograde
        assert just_inside["captured"], prograde
    assert abs(_critical_xi(a, True)) < abs(_critical_xi(a, False))


def test_shadow_edge_matches_integrator():
    """metric.kerr_shadow_params must agree with where the integrator captures photons."""
    from metric import kerr_shadow_params
    a = 0.9
    alpha, _ = kerr_shadow_params(M, a)          # equatorial observer: alpha = -xi
    for xi_edge in (-alpha.max(), -alpha.min()):
        assert trace_kerr(M, a, xi=0.99 * xi_edge, eta=0.0, r_start=R_START)["captured"]
        assert trace_kerr(M, a, xi=1.01 * xi_edge, eta=0.0, r_start=R_START)["escaped"]


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
