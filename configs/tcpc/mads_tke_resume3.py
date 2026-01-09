import os

CONFIG = {
    "label": "mads_tcpc_tke_resume3",
    "algorithm_label": "mads_tke_resume3",
    "objective_kind": "tke",
    "resolution": 5,
    "max_evals": int(os.environ.get("TCPC_MADS_TKE_RESUME3_MAX_EVALS", "70")),
    "geometry_penalty": 1.0e9,
    "space": {
        "names": ("offset", "lower_angle", "upper_angle", "lower_flare", "upper_flare"),
        "x0": [-0.0022, -1.4, -5.4, 0.002475, 0.00095],
        "lower": [-0.0034, -2.26, -6.12, 0.002325, 0.00081],
        "upper": [-0.0010, -0.54, -4.68, 0.00250, 0.00109],
    },
    "optimizer": {
        "type": "mads",
        "n_workers": 8,
        "parallel": True,
        "cache_round": 12,
        "subprocess": True,
        "cache_seed": [
            {
                "x": [-0.0022, -1.4, -5.4, 0.002475, 0.00095],
                "value": 181.83933953,
            }
        ],
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
        "job_name": "mads-opt-tcpc-tke-resume3",
        "time": "168:00:00",
        "cpus": 8,
        "gpus": None,
        "mem": "4G",
        "output": "opt-tcpc-tke-mads-resume3-%j.out",
        "open_mode": "append",
    },
}
