# LensingAndAutomation
Modeling Gravitational Lensing and Photon Trajectories in Schwarzschild and Kerr Black Hole Spacetimes

## Running

```
pip install numpy scipy matplotlib
python main_full.py          # writes 6 figures to outputs/
python tests/test_kerr_geodesics.py
```

The tests check the Kerr photon integrator against analytic results: closest-approach
radii (roots of R(r) = 0), the Schwarzschild limit, and the Bardeen (1973) capture
boundary for prograde and retrograde photons.
