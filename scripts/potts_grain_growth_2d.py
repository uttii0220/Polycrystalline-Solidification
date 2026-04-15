#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal 2D Q-state Potts model / Monte Carlo grain growth simulation.

Model overview
--------------
- 2D lattice with periodic boundary conditions.
- Each lattice site holds a grain id (integer state 0..Q-1).
- Each grain id is assigned a fixed crystallographic orientation angle (radians).
- Grain boundary energy depends on misorientation Δθ between adjacent grains:
    (A) isotropic:   J(Δθ) = J0
    (B) anisotropic: J(Δθ) = J0 * (1 + a * sin²(2Δθ))
- Metropolis Monte Carlo is used to evolve the system.

Misorientation convention (used in this code)
----------------------------------------------
Orientation angle θ has period π (180° rotational symmetry — a common
simplification for 2D grain models).

  raw_diff = |θ₁ - θ₂|  mod  π
  Δθ = min(raw_diff, π - raw_diff)   →  Δθ ∈ [0, π/2]

This is intentionally simple; real crystal symmetry groups are more complex.

Neighborhood
------------
Default: 8-neighbor (Moore) with periodic boundary conditions.
Can be changed to 4-neighbor (von Neumann) via --neighbors 4.

Extension point for inclination dependence
------------------------------------------
See pair_boundary_energy(): it currently takes only the two grain ids and
angles.  To add inclination (interface normal) dependence, pass the lattice
offset direction (dy, dx) from local_energy() into pair_boundary_energy()
and implement J = J(Δθ, inclination) there.

Dependencies: numpy, matplotlib (no other packages required)

Usage examples
--------------
  python3 scripts/potts_grain_growth_2d.py
  python3 scripts/potts_grain_growth_2d.py --sweeps 300 --T 0.05
  python3 scripts/potts_grain_growth_2d.py --compare --sweeps 200
  python3 scripts/potts_grain_growth_2d.py --model isotropic --outdir outputs/iso_run
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


# ──────────────────────────────────────────────
# Default parameters
# ──────────────────────────────────────────────

def default_params() -> dict:
    """Return a dictionary of default simulation parameters."""
    return dict(
        nx=100,            # lattice width  (columns)
        ny=100,            # lattice height (rows)
        n_grains=300,      # number of distinct grain ids (Q)
        neighbor_mode="8", # "4" (von Neumann) or "8" (Moore)
        # ── Monte Carlo ──
        n_sweeps=200,      # sweeps to run; 1 sweep = nx*ny attempted flips
        T=0.10,            # MC temperature; low → faster grain growth
        # ── Grain boundary energy ──
        model="anisotropic",  # "isotropic" or "anisotropic"
        J0=1.0,            # baseline grain boundary energy
        a=0.5,             # anisotropy strength (only used in anisotropic model)
        # ── Orientation ──
        angle_period=np.pi,  # periodicity of orientation angle (π = 180° symmetry)
        # ── I/O ──
        seed=0,            # random seed for reproducibility
        outdir=os.path.join("outputs", "potts_2d"),
        plot_interval=10,  # save/show maps every N sweeps
        save_images=True,  # write PNG files to outdir
        show_live=False,   # open interactive matplotlib windows (avoid on headless)
        compare=False,     # if True, run both models and produce a comparison plot
    )


# ──────────────────────────────────────────────
# Initialization
# ──────────────────────────────────────────────

def initialize_lattice(nx: int, ny: int, n_grains: int,
                       angle_period: float,
                       rng: np.random.Generator):
    """
    Create the initial grain id lattice and per-grain orientation table.

    Returns
    -------
    state  : ndarray shape (ny, nx) int32   — grain id per site
    angles : ndarray shape (n_grains,) float64 — orientation angle per grain id
    """
    # Each site gets a random grain id in [0, n_grains).
    # This gives many tiny "grains" that immediately start coarsening.
    state = rng.integers(0, n_grains, size=(ny, nx), dtype=np.int32)
    # Each grain id gets a random orientation in [0, angle_period).
    angles = rng.random(n_grains) * angle_period
    return state, angles


def ensure_outdir(path: str) -> None:
    """Create output directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


# ──────────────────────────────────────────────
# Neighborhood helpers
# ──────────────────────────────────────────────

def neighbor_offsets(mode: str):
    """
    Return list of (dy, dx) offset tuples for the chosen neighborhood.

    "4" → von Neumann (N, S, E, W)
    "8" → Moore (also NE, NW, SE, SW)
    """
    if mode == "4":
        return [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if mode == "8":
        return [(-1, 0), (1, 0), (0, -1), (0, 1),
                (-1, -1), (-1, 1), (1, -1), (1, 1)]
    raise ValueError(f"neighbor_mode must be '4' or '8', got '{mode}'")


# ──────────────────────────────────────────────
# Misorientation and boundary energy
# ──────────────────────────────────────────────

def misorientation(theta1: float, theta2: float,
                   angle_period: float = np.pi) -> float:
    """
    Compute misorientation Δθ between two orientation angles.

    Uses the π-periodic symmetry convention described in the module docstring.
    Result is in [0, angle_period / 2].
    """
    d = abs(theta1 - theta2) % angle_period
    return min(d, angle_period - d)


def J_of_dtheta(dtheta: float, model: str, J0: float, a: float) -> float:
    """
    Grain boundary energy as a function of misorientation dtheta.

    isotropic   : J = J0               (constant, orientation-independent)
    anisotropic : J = J0*(1 + a*sin²(2Δθ))  (Read–Shockley-inspired toy form)
    """
    if model == "isotropic":
        return J0
    if model == "anisotropic":
        return J0 * (1.0 + a * np.sin(2.0 * dtheta) ** 2)
    raise ValueError(f"model must be 'isotropic' or 'anisotropic', got '{model}'")


def pair_boundary_energy(id_i: int, id_j: int,
                         angles: np.ndarray,
                         model: str, J0: float, a: float,
                         angle_period: float) -> float:
    """
    Boundary energy contribution for one neighboring pair of sites.

    Returns 0 when both sites share the same grain id (no boundary).
    Otherwise returns J(Δθ) evaluated for the grains' orientations.

    ── Extension point for inclination dependence ──
    Add a `boundary_dir=(dy, dx)` argument here, compute the interface
    normal from it, and replace J_of_dtheta() with J(Δθ, inclination).
    local_energy() would need to pass (dy, dx) along as well.
    """
    if id_i == id_j:
        return 0.0
    dth = misorientation(angles[id_i], angles[id_j], angle_period=angle_period)
    return J_of_dtheta(dth, model=model, J0=J0, a=a)


# ──────────────────────────────────────────────
# Local energy
# ──────────────────────────────────────────────

def local_energy(state: np.ndarray, y: int, x: int,
                 angles: np.ndarray, offsets: list,
                 model: str, J0: float, a: float,
                 angle_period: float) -> float:
    """
    Sum of boundary energies between site (y, x) and all its neighbors.

    Periodic boundary conditions are applied via modulo indexing.
    """
    ny, nx = state.shape
    id0 = state[y, x]
    energy = 0.0
    for dy, dx in offsets:
        yy = (y + dy) % ny
        xx = (x + dx) % nx
        energy += pair_boundary_energy(
            id0, state[yy, xx], angles,
            model=model, J0=J0, a=a, angle_period=angle_period,
        )
    return energy


# ──────────────────────────────────────────────
# Monte Carlo
# ──────────────────────────────────────────────

def metropolis_accept(dE: float, T: float,
                      rng: np.random.Generator) -> bool:
    """
    Metropolis acceptance criterion.

    Accepts downhill moves unconditionally.
    Accepts uphill moves with probability exp(-dE/T).
    When T ≤ 0, only downhill moves are accepted (zero-temperature limit).
    """
    if dE <= 0.0:
        return True
    if T <= 0.0:
        return False
    return bool(rng.random() < np.exp(-dE / T))


def mc_sweep(state: np.ndarray, angles: np.ndarray,
             offsets: list, model: str,
             J0: float, a: float, angle_period: float,
             T: float, rng: np.random.Generator) -> None:
    """
    Perform one Monte Carlo sweep (nx * ny attempted site updates).

    Update rule (standard Potts grain growth):
      1. Pick a random site (y, x).
      2. Pick a random neighbor; use its grain id as the candidate state.
      3. Compute local ΔE = E_after − E_before.
      4. Accept with Metropolis probability.

    The state array is modified in-place.
    """
    ny, nx = state.shape
    n_offsets = len(offsets)

    for _ in range(nx * ny):
        y = int(rng.integers(0, ny))
        x = int(rng.integers(0, nx))

        # Pick a random neighbor as the proposed new grain id.
        oi = int(rng.integers(0, n_offsets))
        dy, dx = offsets[oi]
        yy = (y + dy) % ny
        xx = (x + dx) % nx

        candidate = int(state[yy, xx])
        current = int(state[y, x])
        if candidate == current:
            continue  # no change → skip energy evaluation

        e_before = local_energy(state, y, x, angles, offsets,
                                model=model, J0=J0, a=a,
                                angle_period=angle_period)
        state[y, x] = candidate
        e_after = local_energy(state, y, x, angles, offsets,
                               model=model, J0=J0, a=a,
                               angle_period=angle_period)

        if not metropolis_accept(e_after - e_before, T, rng):
            state[y, x] = current  # revert


# ──────────────────────────────────────────────
# Analysis metrics
# ──────────────────────────────────────────────

def boundary_length_indicator(state: np.ndarray) -> int:
    """
    Simple proxy for total grain boundary length.

    Counts mismatching bonds in the right (+x) and down (+y) directions
    (avoids double-counting).  Periodic boundary conditions included.

    Returns an integer count of mismatching bonds.
    """
    gb = 0
    gb += int(np.count_nonzero(state != np.roll(state, -1, axis=1)))  # right
    gb += int(np.count_nonzero(state != np.roll(state, -1, axis=0)))  # down
    return gb


def mean_grain_area(state: np.ndarray) -> float:
    """
    Estimate mean grain area (cells per grain) over grains present in the lattice.

    Uses simple cell-count per grain id; not rigorous but fast and useful
    for tracking coarsening over time.
    """
    counts = np.bincount(state.ravel())
    active = counts[counts > 0]
    return float(np.mean(active)) if len(active) > 0 else 0.0


# ──────────────────────────────────────────────
# Visualization helpers
# ──────────────────────────────────────────────

# Fixed visualization RNG so colors are consistent across frames.
_VIS_RNG = np.random.default_rng(12345)


def _make_color_table(n_grains: int) -> np.ndarray:
    """Generate a fixed random RGB color for each grain id."""
    return _VIS_RNG.random((n_grains, 3))


def plot_maps(state: np.ndarray, angles: np.ndarray,
              title: str, outpath: str = None,
              show: bool = False) -> None:
    """
    Create a two-panel figure:
      Left  — grain id map (random distinct color per grain id)
      Right — orientation angle map (HSV colormap)

    Parameters
    ----------
    outpath : file path to save the figure, or None to skip saving.
    show    : open an interactive window (avoid on headless systems).
    """
    n_grains = len(angles)
    color_table = _make_color_table(n_grains)
    rgb = color_table[state]          # (ny, nx, 3)
    angle_map = angles[state]         # (ny, nx) — per-site orientation

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    axes[0].imshow(rgb, interpolation="nearest")
    axes[0].set_title("grain id map (random colors)")
    axes[0].axis("off")

    im = axes[1].imshow(angle_map, cmap="hsv",
                        vmin=0, vmax=np.pi,
                        interpolation="nearest")
    axes[1].set_title("orientation angle (rad)")
    axes[1].axis("off")
    cbar = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label("θ [rad]")

    fig.suptitle(title)

    if outpath is not None:
        fig.savefig(outpath, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_history(gb_hist, area_hist, title: str,
                 outpath: str = None, show: bool = False) -> None:
    """
    Plot grain boundary length indicator and mean grain area over sweeps.
    Dual y-axis: GB indicator (left) and mean grain area (right).
    """
    fig, ax1 = plt.subplots(figsize=(7, 4), constrained_layout=True)

    ax1.plot(gb_hist, color="tab:blue", label="GB indicator")
    ax1.set_xlabel("sweep")
    ax1.set_ylabel("GB indicator (mismatching bonds)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(area_hist, color="tab:orange", label="mean grain area")
    ax2.set_ylabel("mean grain area (cells)", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    fig.suptitle(title)

    if outpath is not None:
        fig.savefig(outpath, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


# ──────────────────────────────────────────────
# Single simulation run
# ──────────────────────────────────────────────

def run_simulation(params: dict) -> dict:
    """
    Run one full simulation according to params and return results.

    Saves PNG snapshots of grain maps at every plot_interval sweeps,
    plus a history plot at the end.

    Returns a dict with keys:
      final_state, angles, gb_hist, area_hist,
      final_gb, final_mean_area, outdir
    """
    nx = params["nx"]
    ny = params["ny"]
    n_grains = params["n_grains"]
    neighbor_mode = params["neighbor_mode"]
    n_sweeps = params["n_sweeps"]
    T = params["T"]
    model = params["model"]
    J0 = params["J0"]
    a = params["a"]
    angle_period = params["angle_period"]
    seed = params["seed"]
    outdir = params["outdir"]
    plot_interval = params["plot_interval"]
    save_images = params["save_images"]
    show_live = params["show_live"]

    ensure_outdir(outdir)

    rng = np.random.default_rng(seed)
    state, angles = initialize_lattice(nx, ny, n_grains, angle_period, rng)
    offsets = neighbor_offsets(neighbor_mode)

    gb_hist = []
    area_hist = []

    def _record():
        gb_hist.append(boundary_length_indicator(state))
        area_hist.append(mean_grain_area(state))

    def _maybe_plot(sweep: int):
        if not (save_images or show_live):
            return
        title = f"{model} | sweep={sweep} | T={T}"
        path = (os.path.join(outdir, f"{model}_maps_{sweep:04d}.png")
                if save_images else None)
        plot_maps(state, angles, title, outpath=path, show=show_live)

    # sweep 0 — record initial state
    _record()
    _maybe_plot(0)

    for sweep in range(1, n_sweeps + 1):
        mc_sweep(state, angles, offsets,
                 model=model, J0=J0, a=a, angle_period=angle_period,
                 T=T, rng=rng)
        _record()

        if sweep % plot_interval == 0:
            _maybe_plot(sweep)

    # History plot
    hist_path = (os.path.join(outdir, f"{model}_history.png")
                 if save_images else None)
    plot_history(gb_hist, area_hist,
                 title=f"{model} – grain growth history",
                 outpath=hist_path, show=show_live)

    return dict(
        final_state=state.copy(),
        angles=angles.copy(),
        gb_hist=np.array(gb_hist, dtype=float),
        area_hist=np.array(area_hist, dtype=float),
        final_gb=gb_hist[-1],
        final_mean_area=area_hist[-1],
        outdir=outdir,
    )


# ──────────────────────────────────────────────
# CLI argument parsing
# ──────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = default_params()
    ap = argparse.ArgumentParser(
        description=(
            "Minimal 2D Q-state Potts / Monte Carlo grain growth simulation.\n"
            "Outputs PNG snapshots and a history plot to --outdir."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--nx", type=int, default=p["nx"],
                    help="Lattice width (columns)")
    ap.add_argument("--ny", type=int, default=p["ny"],
                    help="Lattice height (rows)")
    ap.add_argument("--n-grains", type=int, default=p["n_grains"],
                    help="Number of distinct grain ids (Q)")
    ap.add_argument("--neighbors", type=str, default=p["neighbor_mode"],
                    choices=["4", "8"],
                    help="Neighborhood: 4=von Neumann, 8=Moore")
    ap.add_argument("--sweeps", type=int, default=p["n_sweeps"],
                    help="Number of MC sweeps (1 sweep = nx*ny attempts)")
    ap.add_argument("--T", type=float, default=p["T"],
                    help="Monte Carlo temperature (lower → faster growth)")
    ap.add_argument("--model", type=str, default=p["model"],
                    choices=["isotropic", "anisotropic"],
                    help="Grain boundary energy model")
    ap.add_argument("--J0", type=float, default=p["J0"],
                    help="Baseline grain boundary energy")
    ap.add_argument("--a", type=float, default=p["a"],
                    help="Anisotropy strength (anisotropic model only)")
    ap.add_argument("--seed", type=int, default=p["seed"],
                    help="Random seed for reproducibility")
    ap.add_argument("--outdir", type=str, default=p["outdir"],
                    help="Directory for output files")
    ap.add_argument("--plot-interval", type=int, default=p["plot_interval"],
                    help="Save/show maps every N sweeps")
    ap.add_argument("--no-save", action="store_true",
                    help="Do not write PNG files (useful for quick tests)")
    ap.add_argument("--show", action="store_true",
                    help="Open interactive matplotlib windows (not for headless)")
    ap.add_argument("--compare", action="store_true",
                    help="Run both isotropic and anisotropic models and compare")
    return ap.parse_args()


# ──────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    params = default_params()
    params["nx"] = args.nx
    params["ny"] = args.ny
    params["n_grains"] = args.n_grains
    params["neighbor_mode"] = args.neighbors
    params["n_sweeps"] = args.sweeps
    params["T"] = args.T
    params["model"] = args.model
    params["J0"] = args.J0
    params["a"] = args.a
    params["seed"] = args.seed
    params["outdir"] = args.outdir
    params["plot_interval"] = args.plot_interval
    params["save_images"] = not args.no_save
    params["show_live"] = args.show
    params["compare"] = args.compare

    if not params["compare"]:
        # ── Single run ──
        res = run_simulation(params)
        print("=== Simulation complete ===")
        print(f"  model          : {params['model']}")
        print(f"  grid           : {params['nx']} × {params['ny']}")
        print(f"  neighbors      : {params['neighbor_mode']}-neighbor, periodic BC")
        print(f"  sweeps         : {params['n_sweeps']}")
        print(f"  T              : {params['T']}")
        print(f"  J0             : {params['J0']},  a = {params['a']}")
        print(f"  final GB ind.  : {res['final_gb']}")
        print(f"  final mean area: {res['final_mean_area']:.2f} cells")
        print(f"  outputs        : {res['outdir']}/")
    else:
        # ── Comparison run: isotropic vs anisotropic (same seed) ──
        base_seed = params["seed"]
        base_outdir = params["outdir"]

        params_iso = params.copy()
        params_iso["model"] = "isotropic"
        params_iso["seed"] = base_seed
        params_iso["outdir"] = os.path.join(base_outdir, "isotropic")

        params_aniso = params.copy()
        params_aniso["model"] = "anisotropic"
        params_aniso["seed"] = base_seed
        params_aniso["outdir"] = os.path.join(base_outdir, "anisotropic")

        print("Running isotropic model …")
        res_iso = run_simulation(params_iso)
        print("Running anisotropic model …")
        res_aniso = run_simulation(params_aniso)

        # Combined comparison plot
        ensure_outdir(base_outdir)
        fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
        ax.plot(res_iso["gb_hist"], label="isotropic")
        ax.plot(res_aniso["gb_hist"], label="anisotropic")
        ax.set_xlabel("sweep")
        ax.set_ylabel("GB indicator (mismatching bonds)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.suptitle("GB indicator comparison: isotropic vs anisotropic")
        cmp_path = os.path.join(base_outdir, "compare_gb_indicator.png")
        fig.savefig(cmp_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        print("=== Comparison complete ===")
        print(f"  isotropic  outputs : {params_iso['outdir']}/")
        print(f"  anisotropic outputs: {params_aniso['outdir']}/")
        print(f"  comparison plot    : {cmp_path}")


if __name__ == "__main__":
    main()
