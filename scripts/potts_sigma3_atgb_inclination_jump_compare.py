#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2D Potts model simulation for Σ3 Asymmetric Tilt Grain Boundary (ATGB) growth.

Purpose
-------
Compare two grain-boundary-energy models for Si Σ3 boundaries:

  'no_jump' : Energy follows the continuous base theory curve for all
              inclination angles.
  'jump'    : Energy follows the base theory curve below 70.53°, then
              transitions (cusp/jump) using empirical data above 70.53°.

Both models are seeded identically so the comparison is fair.
Results are saved to outputs/sigma3_atgb_compare/.

Physical background
-------------------
- Only two grain states (ids 0 and 1) exist, representing two crystal
  orientations with an implicit Σ3 (60° misorientation).
- Grain-boundary energy depends on the inclination angle φ — the angle
  between the boundary plane normal and the reference direction — computed
  from the lattice-bond vector (dy, dx) between neighbouring sites.
- The 'jump' model introduces a cusp/discontinuity around 70–90°,
  mimicking faceting transitions observed in Si Σ3 boundaries.

Usage
-----
  python3 scripts/potts_sigma3_atgb_inclination_jump_compare.py

Dependencies: numpy, matplotlib only.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

# ── Output directory ──────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join("outputs", "sigma3_atgb_compare")

# ── Simulation parameters ─────────────────────────────────────────────────────
NX = 200          # lattice width  (sites)
NY = 200          # lattice height (sites)
N_SWEEPS = 300    # MC sweeps (1 sweep = NX*NY single-site attempts)
TEMPERATURE = 0.1 # Metropolis temperature  T
SEED = 42         # global RNG seed (same initial state for both models)

# 8-neighbour (Moore) offsets (dy, dx)
NEIGHBORS = [
    (-1, -1), (-1, 0), (-1, 1),
    ( 0, -1),           ( 0, 1),
    ( 1, -1), ( 1, 0), ( 1, 1),
]

# Empirical data for the 'jump' model (degrees → energy)
_JUMP_DEG = np.array([70.53, 76.74, 79.98, 90.00])
_JUMP_VAL = np.array([0.441529, 0.481381, 0.457609, 0.408454])
_JUMP_THRESHOLD_DEG = 70.53  # below this → use theory; above → interpolate


# ── Grain-boundary energy functions ──────────────────────────────────────────

def inclination_angle(dy: int, dx: int) -> float:
    """Return inclination angle φ in radians from a bond offset (dy, dx).

    φ = arctan2(|dy|, |dx|), which lies in [0, π/2] (first quadrant).
    The absolute values fold the angle into the first quadrant, exploiting
    the 4-fold symmetry of the square lattice bond directions.
    """
    return np.arctan2(abs(dy), abs(dx))


def gb_energy_theory(phi: float) -> float:
    """Base theory energy (Theory2) for Σ3 boundary at inclination φ (rad).

    E = 0.026187599 * cos(φ) + 0.642051791 * sin(φ)
    """
    return 0.026187599 * np.cos(phi) + 0.642051791 * np.sin(phi)


def gb_energy(phi: float, model: str) -> float:
    """Return grain-boundary energy for inclination angle φ (rad).

    Parameters
    ----------
    phi   : inclination angle in radians, in [0, π/2].
    model : 'no_jump' or 'jump'.

    'no_jump'
        Returns E_theory for all φ.
    'jump'
        Below 70.53°  → E_theory.
        At/above 70.53° → linear interpolation over empirical (deg, val) data.
    """
    e_theory = gb_energy_theory(phi)

    if model == "no_jump":
        return e_theory

    # 'jump' model: check threshold in degrees
    phi_deg = np.degrees(phi)
    if phi_deg < _JUMP_THRESHOLD_DEG:
        return e_theory
    # linear interpolation over the empirical data points
    return float(np.interp(phi_deg, _JUMP_DEG, _JUMP_VAL))


# ── Lattice initialisation ────────────────────────────────────────────────────

def init_lattice(ny: int, nx: int, rng: np.random.Generator) -> np.ndarray:
    """Initialise the lattice with random 0/1 ids (Σ3: two states only)."""
    return rng.integers(0, 2, size=(ny, nx), dtype=np.int8)


# ── Local energy calculation ──────────────────────────────────────────────────

def local_energy(lattice: np.ndarray, y: int, x: int,
                 model: str, ny: int, nx: int) -> float:
    """Compute the total energy of bonds touching site (y, x).

    For each of the 8 neighbours:
      - bond energy = 0                       if ids are equal (same grain)
      - bond energy = gb_energy(phi, model)   if ids differ   (grain boundary)
    """
    site_id = lattice[y, x]
    energy = 0.0
    for dy, dx in NEIGHBORS:
        ny_ = (y + dy) % ny
        nx_ = (x + dx) % nx
        if lattice[ny_, nx_] != site_id:
            phi = inclination_angle(dy, dx)
            energy += gb_energy(phi, model)
    return energy


# ── Grain-boundary length proxy ───────────────────────────────────────────────

def gb_length(lattice: np.ndarray) -> int:
    """Count mismatched bonds (right + down only to avoid double-counting).

    Returns the number of right-bonds (i, j)–(i, j+1) plus
    down-bonds (i, j)–(i+1, j) where the two sites differ.
    This is proportional to the total grain-boundary length.
    """
    right = int(np.sum(lattice != np.roll(lattice, -1, axis=1)))
    down  = int(np.sum(lattice != np.roll(lattice, -1, axis=0)))
    return right + down


# ── Metropolis Monte Carlo ────────────────────────────────────────────────────

def mc_sweep(lattice: np.ndarray, model: str,
             ny: int, nx: int, temperature: float,
             rng: np.random.Generator) -> np.ndarray:
    """Perform one MC sweep (NX*NY single-site Metropolis attempts).

    For each attempt:
      1. Pick a random site (y, x).
      2. Pick a random neighbour and propose copying its id.
      3. Compute ΔE = E_after − E_before (using bonds touching (y, x) only).
      4. Accept if ΔE ≤ 0; else accept with probability exp(−ΔE / T).

    Returns the updated lattice (modified in-place; also returned for clarity).
    """
    n_attempts = ny * nx

    # Pre-draw all random numbers for the sweep (faster than per-step calls)
    ys    = rng.integers(0, ny, size=n_attempts)
    xs    = rng.integers(0, nx, size=n_attempts)
    nb_idx = rng.integers(0, len(NEIGHBORS), size=n_attempts)
    u      = rng.random(size=n_attempts)  # uniform for Metropolis acceptance

    for k in range(n_attempts):
        y, x = int(ys[k]), int(xs[k])
        dy, dx = NEIGHBORS[nb_idx[k]]

        # Proposed new id = neighbour's current id
        ny_ = (y + dy) % ny
        nx_ = (x + dx) % nx
        new_id = lattice[ny_, nx_]

        old_id = lattice[y, x]
        if new_id == old_id:
            continue  # no change → skip

        # Energy before and after the proposed flip
        e_before = local_energy(lattice, y, x, model, ny, nx)
        lattice[y, x] = new_id
        e_after = local_energy(lattice, y, x, model, ny, nx)

        # Metropolis acceptance criterion
        delta_e = e_after - e_before
        if delta_e > 0 and u[k] >= np.exp(-delta_e / temperature):
            lattice[y, x] = old_id  # reject: revert

    return lattice


# ── Run a full simulation ─────────────────────────────────────────────────────

def run_simulation(model: str, initial_lattice: np.ndarray,
                   n_sweeps: int = N_SWEEPS,
                   temperature: float = TEMPERATURE,
                   seed: int = SEED) -> tuple:
    """Run the Potts MC simulation for the given energy model.

    Parameters
    ----------
    model           : 'no_jump' or 'jump'
    initial_lattice : starting grain-id array (copied so original is unchanged)
    n_sweeps        : number of MC sweeps
    temperature     : Metropolis temperature
    seed            : RNG seed for the MC random draws

    Returns
    -------
    lattice   : final lattice after n_sweeps
    gb_series : list of GB-length proxy values recorded after each sweep
    """
    ny, nx = initial_lattice.shape
    lattice = initial_lattice.copy()

    # Separate RNG for the MC dynamics (same seed → same thermal noise for
    # both models, making the comparison as controlled as possible)
    rng = np.random.default_rng(seed)

    gb_series = [gb_length(lattice)]  # record initial GB length

    for sweep in range(1, n_sweeps + 1):
        lattice = mc_sweep(lattice, model, ny, nx, temperature, rng)
        gb_series.append(gb_length(lattice))

        if sweep % 50 == 0:
            print(f"  [{model}] sweep {sweep}/{n_sweeps}, "
                  f"GB indicator = {gb_series[-1]}")

    return lattice, gb_series


# ── Plotting helpers ──────────────────────────────────────────────────────────

def save_comparison_png(lattice_nj: np.ndarray, lattice_j: np.ndarray,
                        initial: np.ndarray, out_dir: str) -> None:
    """Save a side-by-side comparison of initial and final microstructures."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    cmap = plt.cm.bwr  # blue = state 0, red = state 1

    axes[0].imshow(initial, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    axes[0].set_title("Initial state (both models)")
    axes[0].axis("off")

    axes[1].imshow(lattice_nj, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    axes[1].set_title(f"Final: no_jump (T={TEMPERATURE}, {N_SWEEPS} sweeps)")
    axes[1].axis("off")

    axes[2].imshow(lattice_j, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    axes[2].set_title(f"Final: jump (T={TEMPERATURE}, {N_SWEEPS} sweeps)")
    axes[2].axis("off")

    fig.suptitle(
        "Σ3 ATGB Potts model – inclination-dependent GB energy comparison",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()

    path = os.path.join(out_dir, "microstructure_comparison.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def save_gb_timeseries_png(gb_nj: list, gb_j: list, out_dir: str) -> None:
    """Save a time-series plot of the GB-length proxy for both models."""
    sweeps = list(range(len(gb_nj)))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sweeps, gb_nj, label="no_jump", color="steelblue", linewidth=1.5)
    ax.plot(sweeps, gb_j,  label="jump",    color="tomato",    linewidth=1.5)
    ax.set_xlabel("Sweep")
    ax.set_ylabel("GB length proxy (mismatched bonds)")
    ax.set_title(
        "Σ3 ATGB: grain-boundary length vs. MC sweep\n"
        f"(T={TEMPERATURE}, {NX}×{NY} lattice, seed={SEED})"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = os.path.join(out_dir, "gb_timeseries.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # Create output directory if it does not exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Identical initial state for both models ──────────────────────────────
    init_rng = np.random.default_rng(SEED)
    initial = init_lattice(NY, NX, init_rng)

    print(f"Σ3 ATGB Potts simulation  ({NY}×{NX}, T={TEMPERATURE}, "
          f"{N_SWEEPS} sweeps, seed={SEED})")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    # ── Run no_jump model ────────────────────────────────────────────────────
    print("Running model: no_jump …")
    lattice_nj, gb_nj = run_simulation(
        model="no_jump",
        initial_lattice=initial,
        n_sweeps=N_SWEEPS,
        temperature=TEMPERATURE,
        seed=SEED,
    )
    print(f"  → Final GB indicator (no_jump): {gb_nj[-1]}\n")

    # ── Run jump model ───────────────────────────────────────────────────────
    print("Running model: jump …")
    lattice_j, gb_j = run_simulation(
        model="jump",
        initial_lattice=initial,
        n_sweeps=N_SWEEPS,
        temperature=TEMPERATURE,
        seed=SEED,
    )
    print(f"  → Final GB indicator (jump): {gb_j[-1]}\n")

    # ── Save outputs ─────────────────────────────────────────────────────────
    save_comparison_png(lattice_nj, lattice_j, initial, OUTPUT_DIR)
    save_gb_timeseries_png(gb_nj, gb_j, OUTPUT_DIR)

    print("\nDone.  Results saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
