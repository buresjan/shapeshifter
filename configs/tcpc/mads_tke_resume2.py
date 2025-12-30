import os

CONFIG = {
    "label": "mads_tcpc_tke_resume2",
    "algorithm_label": "mads_tke_resume2",
    "objective_kind": "tke",
    "resolution": 5,
    "max_evals": int(os.environ.get("TCPC_MADS_TKE_RESUME2_MAX_EVALS", "70")),
    "geometry_penalty": 1.0e9,
    "space": {
        "names": ("offset", "lower_angle", "upper_angle", "lower_flare", "upper_flare"),
        "x0": [-0.001, -0.97, -5.04, 0.00225, 0.00102],
        "lower": [-0.004, -3.5, -7.2, 0.00175, 0.0006],
        "upper": [0.002, 0.8, -3.6, 0.00250, 0.00130],
    },
    "optimizer": {
        "type": "mads",
        "n_workers": 4,
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
        "job_name": "mads-opt-tcpc-tke-resume2",
        "time": "168:00:00",
        "cpus": 10,
        "gpus": 1,
        "mem": "32G",
        "output": "opt-tcpc-tke-mads-resume2-%j.out",
        "open_mode": "append",
    },
}
