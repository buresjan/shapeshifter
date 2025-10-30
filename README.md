# Shapeshifter

Shapeshifter is a script-oriented playground for rapid experimentation with optimization-driven lattice Boltzmann method (LBM) simulations. The repository wires together four internal submodules:

- [`submodules/optilb/`](submodules/optilb/) – optimization problems and solvers used to drive geometry updates.
- [`submodules/lb2dgeom/`](submodules/lb2dgeom/) – parametric 2D geometry generation utilities for LBM domains.
- [`submodules/meshgen/`](submodules/meshgen/) – 3D geometry templating and voxelization pipelines for simulation-ready lattices.
- [`submodules/tnl-lbm/`](submodules/tnl-lbm/) – C++/CUDA LBM solver that consumes generated geometry and reports objective values.

The typical workflow is:

1. Build a parameterized geometry with `lb2dgeom` and write it to disk.
2. Launch the `tnl-lbm` executable to simulate that geometry and extract a scalar objective.
3. Feed the result back into an `optilb` optimization loop to generate the next set of parameters.

## Repository setup

1. Clone the repository and pull submodules:
   ```bash
   git clone git@github.com:buresjan/shapeshifter.git
   cd shapeshifter
   git submodule update --init --recursive
   ```
2. (Optional) Build the LBM solver following the instructions in `submodules/tnl-lbm/` if you do not already have a compiled binary.

## Python environment

1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. Install the local packages and any top-level requirements:
   ```bash
   pip install -e submodules/optilb -e submodules/lb2dgeom -e submodules/meshgen
   pip install -r requirements.txt
   ```

## Keeping everything up to date

- Update the main repository:
  ```bash
  git pull
  ```
- Refresh submodules to their tracked commits:
  ```bash
  git submodule update --remote
  ```
  or, for a specific module:
  ```bash
  git submodule update --remote submodules/optilb
  ```
- If you need to modify a submodule, switch into its directory, check out the desired branch, and manage commits there. Push changes from within the submodule, then record the updated commit in the parent repository with `git add submodules/<name>`.

When new commits land upstream, repeat the sequence above to ensure both the top-level scripts and the submodules are in sync.
