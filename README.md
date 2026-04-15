# Polycrystalline Solidification

Research scripts for 2D polycrystalline grain growth simulation using the
Q-state Potts model and Monte Carlo methods.

---

## Purpose

Explore how grain boundaries evolve during solidification in a 2D polycrystalline
material. The model captures:

- Many grains, each with a random crystallographic orientation angle.
- Grain boundary energy that depends on the misorientation between adjacent grains.
- Time evolution driven by Metropolis Monte Carlo.
- Switchable isotropic vs. anisotropic boundary energy models.

---

## Directory Layout

```
scripts/   ← runnable experiment scripts (one file per experiment)
outputs/   ← generated images, plots, data (not committed to git)
```

---

## Python Dependencies

Only `numpy` and `matplotlib` are required.

```bash
python3 -m pip install numpy matplotlib
```

---

## Running the Potts Grain Growth Simulation

The main script is `scripts/potts_grain_growth_2d.py`.

### Quick start (default settings)

```bash
python3 scripts/potts_grain_growth_2d.py
```

Outputs go to `outputs/potts_2d/` by default:
- `anisotropic_maps_0000.png` … grain map snapshots at each saved sweep
- `anisotropic_history.png`  … grain boundary indicator + mean grain area over time

### Common options

```bash
# Change model, sweeps, temperature
python3 scripts/potts_grain_growth_2d.py --model isotropic --sweeps 500 --T 0.05

# Compare isotropic vs anisotropic side-by-side (same random seed)
python3 scripts/potts_grain_growth_2d.py --compare --sweeps 300

# Change grid size and neighborhood
python3 scripts/potts_grain_growth_2d.py --nx 200 --ny 200 --neighbors 4

# Custom output directory
python3 scripts/potts_grain_growth_2d.py --outdir outputs/my_run

# Quick test without saving images
python3 scripts/potts_grain_growth_2d.py --sweeps 10 --no-save
```

### All CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--nx`, `--ny` | 100 | Lattice dimensions |
| `--n-grains` | 300 | Number of grain ids (Q states) |
| `--neighbors` | `8` | Neighborhood: `4` = von Neumann, `8` = Moore |
| `--sweeps` | 200 | Monte Carlo sweeps (1 sweep = nx×ny attempts) |
| `--T` | 0.10 | MC temperature (lower → faster grain growth) |
| `--model` | `anisotropic` | `isotropic` or `anisotropic` |
| `--J0` | 1.0 | Baseline grain boundary energy |
| `--a` | 0.5 | Anisotropy strength (anisotropic model) |
| `--seed` | 0 | Random seed for reproducibility |
| `--outdir` | `outputs/potts_2d` | Directory for output files |
| `--plot-interval` | 10 | Save map snapshot every N sweeps |
| `--no-save` | — | Disable PNG output |
| `--show` | — | Open interactive matplotlib windows |
| `--compare` | — | Run both models and produce comparison plot |

---

## Outputs

All generated files are written to the directory specified by `--outdir`
(default: `outputs/potts_2d/`).

| File | Description |
|------|-------------|
| `{model}_maps_{sweep:04d}.png` | Grain id map + orientation map at each saved sweep |
| `{model}_history.png` | GB indicator and mean grain area over time |
| `compare_gb_indicator.png` | GB comparison plot (only with `--compare`) |

The `outputs/` directory is excluded from version control (see `.gitignore`).

---

## Model Notes

- **Misorientation**: angle difference with π-periodicity → Δθ ∈ [0, π/2].
- **Isotropic**: J(Δθ) = J0 (constant boundary energy).
- **Anisotropic**: J(Δθ) = J0 × (1 + a × sin²(2Δθ)).
- **Neighborhood**: 8-neighbor Moore by default; 4-neighbor von Neumann available.
- **Boundary conditions**: periodic in both directions.

### Extending to inclination dependence

The function `pair_boundary_energy()` in the script is the key extension point.
Pass the lattice offset direction `(dy, dx)` from `local_energy()` into it and
implement `J = J(Δθ, inclination)` to add interface-normal dependence.
