CONFIG = {
    "label": "mads_fontan_sr_resume2",
    "algorithm_label": "mads_fontan_sr",
    "max_evals": 60,
    "geometry_penalty": 1.0e9,
    "space": {
        "names": (
            "bump1_amp",
            "bump2_amp",
            "size_scale",
            "straighten_strength",
            "offset_x",
        ),
        # Best-so-far point from the prior MADS SR run (eval 43).
        "x0": [0.0036, 2.46, 0.972, 0.4, -0.324],
        # Tightened bounds around the best region from the previous sweep.
        "lower": [0.0, 2.1, 0.95, 0.34, -0.5],
        "upper": [0.01, 3.0, 0.99, 0.4, -0.15],
    },
    "optimizer": {
        "type": "mads",
        "n_workers": 8,
        "parallel": True,
        "cache_round": None,
        "subprocess": True,
        "print_bounds": True,
    },
    "objective": {
        # pitch = Z_EXTENT / z_voxels
        "vef_config_path": "configs/vef/vef_fontan_sr_mads_opt.json",
        "output_root": "data/vef_fontan_opt/vef_outputs",
        "runs_root": "data/vef_fontan_opt/runs",
        "z_voxels": 250,
        "sim_resolution": 250,
        "keep_temp_files": False,
        "maximize": False,
    },
    "solver": {
        "binary_name": "sim_fontan_sr",
        # "binary_path": "submodules/tnl-lbm/build/sim_NSE/sim_fontan_sr",
        "slurm": {
            "partition": None,
            "gpus": 1,
            "cpus": 4,
            "mem": "8G",
            "walltime": "22:00:00",
            "poll_interval": 60.0,
            "avg_window": 1.0,
            "verbose": False,
        },
    },
    "submit": {
        "job_name": "mads-opt-fontan-sr-resume2",
        "time": "168:00:00",
        "cpus": 8,
        "gpus": None,
        "mem": "16G",
        "partition": None,
        "output": "opt-fontan-sr-mads-resume2-%j.out",
        "open_mode": "append",
    },
}
