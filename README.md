# Shapeshifter

Shapeshifter is a script-oriented playground for optimization-driven lattice Boltzmann method (LBM) studies. It wires together three core submodules for the default 2D workflow and an optional 3D mesh generator:

- [`submodules/optilb/`](submodules/optilb/) – optimization problems and solvers that drive geometry updates.
- [`submodules/lb2dgeom/`](submodules/lb2dgeom/) – parametric 2D geometry generation utilities for LBM domains.
- [`submodules/meshgen/`](submodules/meshgen/) – 3D geometry templating and voxelization pipelines for simulation-ready lattices (used when exploring volumetric cases).
- [`submodules/tnl-lbm/`](submodules/tnl-lbm/) – C++/CUDA LBM solver that consumes generated geometry and reports objective values.

The typical workflow is:

1. Build a parameterized geometry with `lb2dgeom` (or `meshgen` for voxelized 3D studies) and write it to disk.
2. Launch the `tnl-lbm` executable to simulate that geometry and extract a scalar objective.
3. Feed the result back into an `optilb` optimization loop to generate the next set of parameters.

## Repository setup

1. Clone the repository and pull submodules:
   ```bash
   git clone git@github.com:buresjan/shapeshifter.git
   cd shapeshifter
   git submodule update --init --recursive
   ```
2. Build the LBM solver following the instructions in `submodules/tnl-lbm/` if you do not already have a compiled binary.

Check `playground/` for runnable experiments. For example, `playground/run_junction_tcpc.py` generates a mesh with `meshgen`, stages the artefacts in `tnl-lbm`, runs the `sim_tcpc` binary, and prints the objective returned by the solver.

## Python environment

1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. Install the editable submodules and Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   The requirements file installs `optilb`, `lb2dgeom`, and `meshgen` in editable mode along with the shared scientific stack (`numpy`, `scipy`, `matplotlib`). `meshgen` brings extra optional tooling (`gmsh`, `trimesh`, `mayavi`, `tqdm`) and may require system packages:
   - Ensure the Gmsh binary is available on your `PATH` (or install it via your package manager).
   - Mayavi relies on Qt/OpenGL; install matching system libraries (e.g., `qtbase5-dev`, `mesa`, vendor GPU drivers) before running the pip install.
   If you only plan to work with 2D cases, remove or comment out the `-e submodules/meshgen` line in `requirements.txt` before installing.

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
