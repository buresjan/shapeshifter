TCPC config-driven workflow for optimizations and extra-point evaluations.

Overview
- Optimizers run through `scripts/tcpc_optimize.py` using a config file in `configs/tcpc/`.
- Slurm submission uses `scripts/tcpc_submit.py` (no per-job shell scripts needed).
- Extra-point evaluations use `scripts/tcpc_extra_point.py` (single) or `scripts/tcpc_collect_points.py` (batch).
- Predefined point lists live in `configs/tcpc/points/`.

Parallel execution policy
- All TCPC optimizations are intended to run in parallel by default.
- New configs should always set `optimizer.parallel: True` explicitly (even though the code defaults to true).
- Set `optimizer.n_workers` to match `submit.cpus` for maximum concurrent evaluations.
- For Nelder-Mead, set `optimizer.force_thread_pool: True` (or `OPTILB_FORCE_THREAD_POOL=1`) to guarantee parallel evals with the current objective wiring.
- For MADS, parallelism is controlled by PyNomad threads (`optimizer.n_workers`) and `optimizer.parallel: True`.

Prerequisites
- Build the tnl-lbm solver (binaries under `submodules/tnl-lbm/build/sim_NSE/`).
- Use the repo venv (`pip install -r requirements.txt`) so submodules are importable.
- For MADS runs, ensure PyNomad is available (`optilb` uses PyNomadBBO).

Quick start (local)
```
python3 scripts/tcpc_optimize.py --config configs/tcpc/nm_tcpc.py
```

Quick start (Slurm)
```
python3 scripts/tcpc_submit.py --config configs/tcpc/nm_tcpc.py
```

Running optimizations (local vs Slurm)
Local run:
```
python3 scripts/tcpc_optimize.py --config configs/tcpc/mads_tke_resume2.py
```

Slurm submission (uses config submit defaults):
```
python3 scripts/tcpc_submit.py --config configs/tcpc/mads_tke_resume2.py
```

Override job resources at submit time:
```
python3 scripts/tcpc_submit.py --config configs/tcpc/mads_tke_resume2.py \
  --job-name=mads-opt-tcpc-tke-resume2 --time=168:00:00 --cpus=10 --gpus=1 --mem=32G
```

Config reference (common fields)
- `label`: Short name for the run.
- `algorithm_label`: Label embedded into evaluation logs and case names.
- `objective_kind`: `tcpc` or `tke` (selects solver binary and defaults).
- `resolution`: Voxel/solver resolution.
- `max_evals`: Evaluation budget.
- `geometry_penalty`: Value used when geometry/solver fails.
- `eval_log_root`: Override per-run evaluation log directory (default `tmp/junction_tcpc_logs/`).
- `eval_log_shared`: Enable shared evaluation CSV logging (default `true`).
- `eval_log_shared_root`: Directory for shared evaluation CSVs (default `data/junction_tcpc_logs/shared/`).
- `eval_log_shared_path`: Explicit shared evaluation CSV path (overrides `eval_log_shared_root`).
- `eval_log_shared_run`: When true, append the run tag as a subdirectory under the shared log root.
- `space`:
  - `names`: Parameter names.
  - `x0`: Starting point.
  - `lower` / `upper`: Bounds.
- `optimizer`:
  - `type`: `nelder_mead` or `mads`.
  - `n_workers`: Worker count (use `null` to defer to env vars).
  - `parallel`: Enable parallel evaluations (default true; keep it on for new configs).
  - `force_thread_pool`: Force threaded evaluation for Nelder-Mead (avoids pickling issues).
  - `memoize`, `parallel_poll_points`, `log_simplex`: NM controls.
  - `no_improve_thr`, `no_improv_break`, `tol`, `penalty`: NM stop/penalty settings.
  - `initial_simplex`, `initial_simplex_values`: NM resume in-config.
  - `simplex_npz_env`: NM resume from NPZ via env var.
  - `cache_round`, `cache_seed`, `subprocess`, `print_bounds`: MADS memoization/subprocess options.
- `solver`:
  - `binary_name` or `binary_path`.
  - `slurm`: default Slurm settings for solver jobs.
- `split`:
  - `time_index`, `min_fraction`, `write_vtp`, `write_debug_points`.
- `submit`:
  - Default Slurm settings for optimizer submissions.
- `extra_points`:
  - `csv_default`: default CSV output.
  - `algorithm_label`: label for extra-point runs.
  - `submit`: Slurm defaults for extra-point jobs.

Environment overrides (optional)
- `OPT_NM_WORKERS`, `OPT_MADS_WORKERS` to override worker counts.
- `OPTILB_FORCE_THREAD_POOL=1` to force threaded evaluation for Nelder-Mead.
- `TCPC_RUN_TAG` to group run artifacts.
- `TCPC_EVAL_LOG_SHARED_RUN` to scope shared evaluation logs per run tag.
- `TCPC_SLURM_*` to override solver-job resources in `run_tcpc_simulation.py`:
  `TCPC_SLURM_PARTITION`, `TCPC_SLURM_GPUS`, `TCPC_SLURM_CPUS`, `TCPC_SLURM_MEM`,
  `TCPC_SLURM_WALLTIME`, `TCPC_SLURM_POLL_INTERVAL`, `TCPC_SLURM_AVG_WINDOW`.

Extra-point evaluation (single point)
```
python3 scripts/tcpc_extra_point.py \
  --config configs/tcpc/nm_tcpc.py \
  --direction offset --sign plus \
  --offset -0.002352 --lower-angle -1.78876 --upper-angle 6.15250 \
  --lower-flare 0.00196229 --upper-flare 0.00142384
```

Extra-point batch from CSV
CSV columns required:
`direction, sign, offset, lower_angle, upper_angle, lower_flare, upper_flare`
Optional: `case_label`

Submit via Slurm:
```
python3 scripts/tcpc_collect_points.py \
  --config configs/tcpc/nm_tcpc.py \
  --points-csv configs/tcpc/points/extra_points_round1.csv
```

Run locally:
```
python3 scripts/tcpc_collect_points.py \
  --config configs/tcpc/nm_tcpc.py \
  --points-csv configs/tcpc/points/extra_points_round1.csv \
  --mode local
```

Parametric extra-point batch (base + step)
```
python3 scripts/tcpc_collect_points.py \
  --config configs/tcpc/mads_tcpc.py \
  --base "-0.00335,-1.79,6.15,0.00196,0.00142" \
  --step "0.0010,1.5,1.5,0.00012,0.00012"
```

Where outputs go
- Evaluation CSV logs: `tmp/junction_tcpc_logs/` (auto-named by label + timestamp).
- Shared evaluation CSVs: `data/junction_tcpc_logs/shared/` (auto-named by algorithm label).
- Extra-point CSVs: `tmp/extra_points/` (see config `extra_points.csv_default`).
- Solver artifacts: `tmp/junction_tcpc_runs/`.

Mapped legacy scripts
- See `playground/archive/2025-12/MANIFEST.md` for old -> new command mapping.
