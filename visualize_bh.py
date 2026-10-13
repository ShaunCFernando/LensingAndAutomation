import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D   # noqa

from metric       import (sch_rs, sch_critical_b, sch_photon_sphere,
                           kerr_horizons, kerr_ISCO, kerr_photon_orbit,
                           kerr_ergosphere, kerr_shadow_params)
from raytracer    import (sch_shadow_image, kerr_shadow_image,
                           batch_trace_schwarzschild, trace_kerr)
from accretion_disk import build_disk_mesh, disk_temperature, blackbody_color


# ──────────────────────────────────────────────────────────────────────────────
#  Color themes
# ──────────────────────────────────────────────────────────────────────────────

BG      = "#04060f"
TEXT    = "#c8d8f0"
TEXT2   = "#6880a8"
HORIZON = "#e53935"
PHOTON  = "#fb8c00"
ISCO    = "#43a047"
ERGO    = "#9c27b0"
RING    = "#fdd835"

SPACE_CMAP = mcolors.LinearSegmentedColormap.from_list("space", [
    (0.00, "#04060f"), (0.20, "#0d1a3a"), (0.45, "#0d4a8a"),
    (0.65, "#1976d2"), (0.80, "#4fc3f7"), (1.00, "#ffffff")])

ORBIT_COLORS = ["#29b6f6", "#4caf50", "#ff7043", "#e040fb",
                "#fdd835", "#26c6da", "#ef5350"]


# ----------------------------------
#  Figure 1 — Schwarzschild Shadow
# ----------------------------------

def fig_schwarzschild_shadow(M=1.0, save=None):
    
    # Renders the Schwarzschild black hole shadow.

    print("  Rendering Schwarzschild shadow...")
    rs    = sch_rs(M)
    b_crit = sch_critical_b(M)

    image, alpha_arr, beta_arr, bc_M = sch_shadow_image(M, D_obs=50*M,
                                                         n_pixels=400, fov_rs=6.0)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8), facecolor=BG)
    ax.set_facecolor(BG)

    # Background starfield (randomized faint dots)
    rng = np.random.default_rng(42)
    n_stars = 600
    sx = rng.uniform(-6, 6, n_stars)
    sy = rng.uniform(-6, 6, n_stars)
    sm = rng.exponential(0.3, n_stars)
    ax.scatter(sx, sy, s=sm, c="white", alpha=0.4, zorder=1, linewidths=0)

    # Main shadow image (semi-transparent overlay)
    # 0 = shadow (black), 0.85 = ring (bright), 1.0 = sky
    cmap_shadow = mcolors.LinearSegmentedColormap.from_list("shadow", [
        (0.00, "#000000"), (0.45, "#000000"),
        (0.70, "#ff8f00"), (0.88, "#fffde7"),
        (1.00, "#ffffff00")])
    ax.imshow(image, extent=[alpha_arr[0], alpha_arr[-1],
                              beta_arr[0],  beta_arr[-1]],
              cmap=cmap_shadow, origin="lower", vmin=0, vmax=1,
              zorder=2, alpha=0.95, interpolation="bilinear")

    # Annotate key radii
    theta = np.linspace(0, 2*np.pi, 360)
    for r_label, color, ls, label in [
        (2*M/M, HORIZON, "-",  f"Event horizon  r = 2M"),
        (sch_photon_sphere(M)/M, PHOTON, "--", f"Photon sphere  r = 3M"),
        (b_crit/M, RING, ":",  f"Shadow edge  b_c = 3√3·M ≈ {b_crit/M:.2f}M"),
    ]:
        ax.plot(r_label*np.cos(theta), r_label*np.sin(theta),
                color=color, lw=1.2, ls=ls, label=label, zorder=5)

    # Labels
    ax.set_xlabel("α / M  (sky x-coordinate)", color=TEXT2, fontsize=11)
    ax.set_ylabel("β / M  (sky y-coordinate)", color=TEXT2, fontsize=11)
    ax.set_title("Schwarzschild Black Hole Shadow\n"
                 "Critical impact parameter b_c = 3√3 M",
                 color=TEXT, fontsize=12, fontweight="bold", pad=12)
    ax.tick_params(colors=TEXT2)
    ax.set_xlim(-6, 6); ax.set_ylim(-6, 6)
    ax.set_aspect("equal")

    leg = ax.legend(loc="upper right", fontsize=9,
                    facecolor="#0a0a25", edgecolor="#253060", labelcolor=TEXT)
    for spine in ax.spines.values():
        spine.set_edgecolor("#1a2a4a")

    _save_or_show(fig, save)
    return fig


# ----------------------------------
#  Figure 2 — Photon Orbit Diagram
# ----------------------------------

def fig_photon_orbits(M=1.0, save=None):
    
    """
    Photon orbit diagram for Schwarzschild spacetime.
    Color code:
      Blue   → 0 half-orbits (direct rays — slightly deflected)
      Green  → ~1 half-orbit (photon ring n=1)
      Orange → ~2 half-orbits (photon ring n=2)
      Red    → captured (fell into black hole)
    """

    print("  Rendering photon orbit diagram...")
    rs     = sch_rs(M)
    b_crit = sch_critical_b(M)
    r_start = 30.0 * M

    # Sample impact parameters: linear + log-spaced near b_crit
    b_direct  = np.linspace(b_crit * 1.8, b_crit * 5.0, 12)
    b_ring1   = np.linspace(b_crit * 1.005, b_crit * 1.5, 10)
    b_ring2   = np.linspace(b_crit * 1.0002, b_crit * 1.003, 6)
    b_capture = np.linspace(0.5 * M, b_crit * 0.98, 8)
    b_all     = np.concatenate([b_direct, b_ring1, b_ring2, b_capture])

    results   = batch_trace_schwarzschild(M, b_all, r_start=r_start)

    fig, ax   = plt.subplots(1, 1, figsize=(10, 10), facecolor=BG)
    ax.set_facecolor(BG)

    # Draw black hole
    hole = Circle((0, 0), rs, color="black", zorder=10)
    ax.add_patch(hole)

    # Photon sphere
    ax.add_patch(Circle((0, 0), sch_photon_sphere(M),
                         fill=False, edgecolor=PHOTON, lw=1.0,
                         linestyle="--", zorder=8, alpha=0.7))

    # Draw rays
    for res in results:
        if res is None:
            continue
        n = res["n_half_orbits"]
        b = res["b"]
        cap = res["captured"]

        if cap:
            col = "#e53935"
            alpha, lw = 0.5, 0.9
        elif n < 0.5:
            t = min((b - b_crit) / (b_crit * 3), 1.0)
            col = ORBIT_COLORS[0]
            alpha, lw = 0.6 + 0.3*t, 0.9
        elif n < 1.5:
            col = ORBIT_COLORS[1]
            alpha, lw = 0.85, 1.3
        elif n < 2.5:
            col = ORBIT_COLORS[2]
            alpha, lw = 0.9, 1.5
        else:
            col = ORBIT_COLORS[3]
            alpha, lw = 0.95, 1.8

        x, y = res["x"], res["y"]
        ax.plot(x, y, color=col, lw=lw, alpha=alpha, zorder=5)
        # Mirror (symmetry): also plot -y version
        ax.plot(x, -y, color=col, lw=lw, alpha=alpha * 0.5, zorder=4)

    # Legend patches
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0],[0], color=ORBIT_COLORS[0], lw=1.5, label="Direct (0 half-orbits)"),
        Line2D([0],[0], color=ORBIT_COLORS[1], lw=1.5, label="1st photon ring (~1)"),
        Line2D([0],[0], color=ORBIT_COLORS[2], lw=1.5, label="2nd photon ring (~2)"),
        Line2D([0],[0], color="#e53935",        lw=1.5, label="Captured (shadow)"),
        Line2D([0],[0], color=PHOTON, lw=1.0, ls="--", label="Photon sphere r=3M"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=9,
              facecolor="#0a0a25", edgecolor="#253060", labelcolor=TEXT)

    lim = r_start * 0.55
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("x / M", color=TEXT2, fontsize=11)
    ax.set_ylabel("y / M", color=TEXT2, fontsize=11)
    ax.set_title("Schwarzschild Photon Orbits\n"
                 "Rays color-coded by number of orbits around the photon sphere",
                 color=TEXT, fontsize=12, fontweight="bold", pad=12)
    ax.tick_params(colors=TEXT2)
    for spine in ax.spines.values():
        spine.set_edgecolor("#1a2a4a")

    _save_or_show(fig, save)
    return fig


# ----------------------------------
#  Figure 3 — Kerr Shadow
# ----------------------------------


def fig_kerr_shadow(M=1.0, a=0.9, save=None):

    # Render the Kerr black hole shadow for spin a.

    print(f"  Rendering Kerr shadow (a/M = {a/M:.2f})...")
    r_outer, _ = kerr_horizons(M, a)
    r_isco_pro  = kerr_ISCO(M, a, prograde=True)

    image, alpha_arr, beta_arr = kerr_shadow_image(
        M, a, theta_obs=np.pi/2, n_pixels=300, fov_M=7.5)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8), facecolor=BG)
    ax.set_facecolor(BG)

    # Starfield
    rng = np.random.default_rng(99)
    ax.scatter(rng.uniform(-7.5, 7.5, 700), rng.uniform(-7.5, 7.5, 700),
               s=rng.exponential(0.3, 700), c="white", alpha=0.35, zorder=1)

    cmap_shadow = mcolors.LinearSegmentedColormap.from_list("shadow", [
        (0.00, "#000000"), (0.45, "#000000"),
        (0.70, "#ff8f00"), (0.88, "#fffde7"),
        (1.00, "#ffffff00")])
    ax.imshow(image, extent=[alpha_arr[0], alpha_arr[-1],
                              beta_arr[0],  beta_arr[-1]],
              cmap=cmap_shadow, origin="lower", vmin=0, vmax=1,
              zorder=2, alpha=0.95, interpolation="bilinear")

    # Shadow boundary from critical curve
    alpha_crit, beta_crit = kerr_shadow_params(M, a)
    if len(alpha_crit) > 3:
        ax.plot(alpha_crit/M,  beta_crit/M, color=RING, lw=1.5, ls=":", zorder=6, label="Shadow boundary")
        ax.plot(alpha_crit/M, -beta_crit/M, color=RING, lw=1.5, ls=":", zorder=6)

    # Ergosphere at equator (θ = π/2)
    r_ergo = kerr_ergosphere(np.pi/2, M, a)
    theta  = np.linspace(0, 2*np.pi, 360)
    ax.plot(r_ergo/M * np.cos(theta), r_ergo/M * np.sin(theta),
            color=ERGO, lw=1.0, ls="--", label=f"Ergosphere r_e = {r_ergo/M:.2f}M", zorder=5)

    ax.set_xlabel("α / M", color=TEXT2, fontsize=11)
    ax.set_ylabel("β / M", color=TEXT2, fontsize=11)
    ax.set_title(f"Kerr Black Hole Shadow  (a/M = {a/M:.2f})\n"
                 f"Frame dragging shifts shadow: D-shaped asymmetry",
                 color=TEXT, fontsize=12, fontweight="bold", pad=12)
    ax.tick_params(colors=TEXT2)
    ax.set_xlim(-7.5, 7.5); ax.set_ylim(-7.5, 7.5)
    ax.set_aspect("equal")

    ax.annotate("Prograde\n(inward shift)", xy=(-5.0, 0), color=PHOTON,
                fontsize=8, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.3", fc="#0a0020", ec=PHOTON, lw=0.8))
    ax.annotate("Retrograde\n(outward shift)", xy=(4.5, 0), color="#29b6f6",
                fontsize=8, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.3", fc="#0a0020", ec="#29b6f6", lw=0.8))
    ax.annotate(f"Spin  a = {a:.2f}M\n→ (CCW)", xy=(0, 6.5), color=ERGO,
                fontsize=9, ha="center")

    ax.legend(loc="lower right", fontsize=9,
              facecolor="#0a0a25", edgecolor="#253060", labelcolor=TEXT)
    for spine in ax.spines.values():
        spine.set_edgecolor("#1a2a4a")

    _save_or_show(fig, save)
    return fig


# ----------------------------------
#  Figure 4 — Side-by-side Comparison
# ----------------------------------

def fig_comparison(M=1.0, a_vals=[0.0, 0.5, 0.9, 0.998], save=None):

    # Side-by-side comparison of shadows for increasing spin parameter a/M.

    print("  Rendering spin comparison grid...")
    fig, axes = plt.subplots(1, len(a_vals), figsize=(4*len(a_vals), 5),
                              facecolor=BG)

    for ax, a in zip(axes, a_vals):
        ax.set_facecolor(BG)
        rng = np.random.default_rng(int(a*100))
        ax.scatter(rng.uniform(-7, 7, 300), rng.uniform(-7, 7, 300),
                   s=rng.exponential(0.2, 300), c="white", alpha=0.3, zorder=1)

        if a == 0.0:
            image, alp, bet, _ = sch_shadow_image(M, D_obs=50*M, n_pixels=180, fov_rs=5.5)
        else:
            image, alp, bet = kerr_shadow_image(M, a, n_pixels=180, fov_M=7.0)

        cmap_s = mcolors.LinearSegmentedColormap.from_list("s", [
            (0.0, "#000000"), (0.45, "#000000"), (0.7, "#ff8f00"),
            (0.9, "#fffde7"), (1.0, "#ffffff00")])
        ax.imshow(image, extent=[alp[0], alp[-1], bet[0], bet[-1]],
                  cmap=cmap_s, origin="lower", vmin=0, vmax=1,
                  zorder=2, alpha=0.95, interpolation="bilinear")

        r_outer, _ = kerr_horizons(M, a) if a > 0 else (sch_rs(M), 0)
        theta = np.linspace(0, 2*np.pi, 360)
        ax.plot(r_outer/M*np.cos(theta), r_outer/M*np.sin(theta),
                color=HORIZON, lw=1.0, alpha=0.8)

        ax.set_xlim(-7, 7); ax.set_ylim(-7, 7)
        ax.set_aspect("equal")
        ax.tick_params(colors=TEXT2, labelsize=8)
        ax.set_title(f"a/M = {a:.3f}" + (" (Schwarzschild)" if a == 0 else ""),
                     color=TEXT, fontsize=10, pad=6)
        ax.set_xlabel("α / M", color=TEXT2, fontsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor("#1a2a4a")

    axes[0].set_ylabel("β / M", color=TEXT2, fontsize=9)
    fig.suptitle("Black Hole Shadow: Evolution with Spin Parameter a/M",
                 color=TEXT, fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    _save_or_show(fig, save)
    return fig


# ----------------------------------
#  Figure 5 — 3D Combined: Embedding + Disk + Photon Ring
# ----------------------------------

def fig_3d_combined(M=1.0, a=0.7, save=None):
    """
    3D visualization combining:
      - Schwarzschild/Kerr embedding funnel
      - Accretion disk (colored by Novikov-Thorne temperature + Doppler)
      - Photon ring / shadow disk
      - Horizon and ergosphere markers
    """
    print("  Rendering 3D combined visualization...")
    from embedding_util import build_embedding_surface_full

    rs_val   = sch_rs(M)
    r_outer, _ = kerr_horizons(M, a)
    r_isco   = kerr_ISCO(M, a, prograde=True)
    r_ergo   = kerr_ergosphere(np.pi/2, M, a)

    fig = plt.figure(figsize=(14, 10), facecolor=BG)
    ax  = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(BG)

    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = True
        pane.set_facecolor(BG)
        pane.set_edgecolor("#151530")
    ax.grid(color="#151530", lw=0.4, alpha=0.5)
    ax.tick_params(colors=TEXT2, labelsize=8)

    # ---- Embedding funnel ----
    surf = build_embedding_surface_full(M, r_max_factor=8.0, n_r=100, n_phi=150)
    X, Y = surf["X"], surf["Y"]
    Z_up, Z_dn = surf["Z_upper"], surf["Z_lower"]
    R_surf = np.sqrt(X**2 + Y**2)

    # Lapse coloring
    lapse_val = np.sqrt(np.clip(1.0 - rs_val / R_surf, 0.0, None))
    color_arr = SPACE_CMAP(0.3 * lapse_val + 0.1)

    ax.plot_surface(X, Y, Z_up, facecolors=color_arr,
                    rstride=2, cstride=2, linewidth=0, alpha=0.55, zorder=1)
    ax.plot_surface(X, Y, Z_dn, facecolors=color_arr,
                    rstride=2, cstride=2, linewidth=0, alpha=0.35, zorder=1)

    # ---- Accretion disk ----
    disk_r_outer = 7.5 * M
    Xd, Yd, Zd, RGBA, _ = build_disk_mesh(r_isco, disk_r_outer,
                                            n_r=60, n_phi=150,
                                            M=M, a=a, include_redshift=True)
    ax.plot_surface(Xd, Yd, Zd + 0.01,  # tiny z-offset so disk floats above z=0
                    facecolors=RGBA,
                    rstride=1, cstride=2, linewidth=0,
                    shade=False, alpha=0.95, zorder=5)

    # ---- Horizon sphere ----
    u_h = np.linspace(0, 2*np.pi, 60)
    v_h = np.linspace(0, np.pi,   30)
    Xh  = r_outer * np.outer(np.cos(u_h), np.sin(v_h))
    Yh  = r_outer * np.outer(np.sin(u_h), np.sin(v_h))
    Zh  = r_outer * np.outer(np.ones_like(u_h), np.cos(v_h))
    ax.plot_surface(Xh, Yh, Zh, color="black", alpha=0.98, zorder=10, linewidth=0)
    ax.plot_surface(Xh, Yh, Zh, color=HORIZON, alpha=0.15, zorder=11, linewidth=0)

    # ---- Ergosphere ring (equatorial) ----
    theta_ring = np.linspace(0, 2*np.pi, 360)
    ax.plot(r_ergo*np.cos(theta_ring), r_ergo*np.sin(theta_ring), [0]*360,
            color=ERGO, lw=1.5, alpha=0.8, zorder=20)

    # ---- ISCO ring ----
    ax.plot(r_isco*np.cos(theta_ring), r_isco*np.sin(theta_ring), [0]*360,
            color=ISCO, lw=1.2, ls="--", alpha=0.9, zorder=20)

    # ---- Spin axis arrow ----
    z_max = 2.5 * np.sqrt(rs_val * (7.0*rs_val - rs_val))
    ax.quiver(0, 0, -z_max*0.5, 0, 0, z_max, color=ERGO,
              arrow_length_ratio=0.12, lw=2.0, alpha=0.7)
    ax.text(0, 0, z_max*0.6, "Spin axis  J", color=ERGO, fontsize=9, ha="center")

    ax.set_xlabel("x / M", color=TEXT2, fontsize=9, labelpad=5)
    ax.set_ylabel("y / M", color=TEXT2, fontsize=9, labelpad=5)
    ax.set_zlabel("z (embedding)", color=TEXT2, fontsize=9, labelpad=5)
    ax.set_title(
        f"Kerr Black Hole 3D  |  a/M = {a/M:.2f}  |  r+ = {r_outer:.2f}M  |  r_ISCO = {r_isco:.2f}M",
        color=TEXT, fontsize=11, fontweight="bold", pad=12)

    ax.view_init(elev=28, azim=-55)

    _save_or_show(fig, save)
    return fig


# ----------------------------------
#  Figure 6 — Kerr Photon Orbits (equatorial)
# ----------------------------------

def fig_kerr_photon_orbits(M=1.0, a=0.7, save=None):

    # Trace photon rays in the equatorial plane of a Kerr black hole.
    
    print(f"  Rendering Kerr photon orbits (a/M={a/M:.2f})...")
    r_outer, _ = kerr_horizons(M, a)
    r_ph_pro  = kerr_photon_orbit(M, a, prograde=True)
    r_ph_ret  = kerr_photon_orbit(M, a, prograde=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), facecolor=BG)

    labels_data = [
        ("Prograde photons  (co-rotate with spin)",  +1),
        ("Retrograde photons  (counter-rotate)", -1),
    ]

    for ax, (title, sign) in zip(axes, labels_data):
        ax.set_facecolor(BG)
        hole = Circle((0, 0), r_outer, color="black", zorder=10)
        ax.add_patch(hole)
        ax.add_patch(Circle((0, 0), r_outer, fill=False,
                             edgecolor=HORIZON, lw=1.2, zorder=11))

        r_ph = r_ph_pro if sign > 0 else r_ph_ret
        ax.add_patch(Circle((0, 0), r_ph, fill=False,
                             edgecolor=PHOTON, lw=1.0, ls="--",
                             label=f"Photon orbit r={r_ph:.2f}M", zorder=8))

        # Ergosphere
        r_ergo = kerr_ergosphere(np.pi/2, M, a)
        th = np.linspace(0, 2*np.pi, 360)
        ax.plot(r_ergo*np.cos(th), r_ergo*np.sin(th),
                color=ERGO, lw=0.8, ls=":", alpha=0.7, label=f"Ergosphere r={r_ergo:.2f}M")

        # Trace rays
        r_start = 25.0 * M
        b_crit_local = r_ph * (1.0 - 2*M/r_ph + a**2/r_ph**2)**(-0.5) if r_ph > r_outer else r_ph

        b_vals = np.linspace(r_ph * 0.7, r_ph * 3.0, 14)
        colors_cycle = plt.cm.plasma(np.linspace(0.2, 0.9, len(b_vals)))

        for b_raw, col in zip(b_vals, colors_cycle):
            xi  = sign * abs(b_raw)    # sign determines prograde/retrograde
            eta = 0.0                  # equatorial: η = 0

            try:
                res = trace_kerr(M, a, xi=xi, eta=eta,
                                 r_start=r_start, theta_obs=np.pi/2,
                                 n_steps=2000, lam_max=120.0)
                x2d = res["r"] * np.cos(res["phi"])
                y2d = res["r"] * np.sin(res["phi"])
                alpha_line = 0.85 if not res["captured"] else 0.5
                ax.plot(x2d, y2d, color=col, lw=1.2, alpha=alpha_line, zorder=5)
            except Exception:
                pass

        lim = r_start * 0.45
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_title(title, color=TEXT, fontsize=10, fontweight="bold", pad=8)
        ax.set_xlabel("x / M", color=TEXT2, fontsize=9)
        ax.set_ylabel("y / M", color=TEXT2, fontsize=9)
        ax.tick_params(colors=TEXT2)
        ax.legend(fontsize=8, facecolor="#0a0a25",
                  edgecolor="#253060", labelcolor=TEXT)
        for spine in ax.spines.values():
            spine.set_edgecolor("#1a2a4a")

    fig.suptitle(f"Frame Dragging: Prograde vs Retrograde Photons  (a/M = {a/M:.2f})",
                 color=TEXT, fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save_or_show(fig, save)
    return fig


# ──────────────────────────────────────────────────────────────────────────────
#  Utilities
# ──────────────────────────────────────────────────────────────────────────────

def _save_or_show(fig, save):
    plt.tight_layout()
    if save:
        fig.savefig(save, dpi=150, facecolor=fig.get_facecolor(),
                    bbox_inches="tight")
        print(f"    → saved: {save}")
    else:
        plt.show()