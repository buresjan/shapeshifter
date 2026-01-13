import os

CONFIG = {
    "label": "mads_tcpc_tke_resume4",
    "algorithm_label": "mads_tke_resume4",
    "objective_kind": "tke",
    "resolution": 5,
    "max_evals": int(os.environ.get("TCPC_MADS_TKE_RESUME4_MAX_EVALS", "30")),
    "geometry_penalty": 1.0e9,
    "eval_log_shared_path": "data/junction_tcpc_logs/shared/mads_tke_resume4.csv",
    "space": {
        "names": ("offset", "lower_angle", "upper_angle", "lower_flare", "upper_flare"),
        "x0": [-0.002212, -1.572, -5.4, 0.0024673, 0.0009612],
        "lower": [-0.002800, -2.000, -6.000, 0.0022500, 0.0008500],
        "upper": [-0.001600, -0.800, -4.200, 0.0025000, 0.0010500],
    },
    "optimizer": {
        "type": "mads",
        "n_workers": 8,
        "parallel": True,
        "cache_round": 12,
        "subprocess": True,
        "print_bounds": True,
    },
    "solver": {
        "binary_name": "sim_tcpc_tke",
        "slurm": {
            "partition": None,
            "gpus": 1,
            "cpus": 4,
            "mem": "16G",
            "walltime": "22:00:00",
            "poll_interval": 60.0,
            "avg_window": 1.0,
            "verbose": False,
        },
    },
    "split": {
        "time_index": -1,
        "min_fraction": 0.25,
        "write_vtp": False,
        "write_debug_points": False,
    },
    "submit": {
        "job_name": "mads-opt-tcpc-tke-resume4",
        "time": "168:00:00",
        "cpus": 8,
        "gpus": None,
        "mem": "16G",
        "output": "opt-tcpc-tke-mads-resume4-%j.out",
        "open_mode": "append",
    },
}
