Archived playground scripts (2025-12). Suggested replacements:

Optimization runners
- optimize_junction_tcpc.py -> scripts/tcpc_optimize.py --config configs/tcpc/nm_tcpc.py
- optimize_junction_tcpc_tke.py -> scripts/tcpc_optimize.py --config configs/tcpc/nm_tke.py
- optimize_junction_tcpc_nm_resume.py -> scripts/tcpc_optimize.py --config configs/tcpc/nm_resume_tcpc.py
- optimize_junction_tcpc_tke_resume.py -> scripts/tcpc_optimize.py --config configs/tcpc/nm_resume_tke.py
- optimize_junction_tcpc_mads.py -> scripts/tcpc_optimize.py --config configs/tcpc/mads_tcpc.py
- optimize_junction_tcpc_mads_resume.py -> scripts/tcpc_optimize.py --config configs/tcpc/mads_resume_tcpc.py
- optimize_junction_tcpc_mads_tke.py -> scripts/tcpc_optimize.py --config configs/tcpc/mads_tke.py
- optimize_junction_tcpc_mads_tke_resume.py -> scripts/tcpc_optimize.py --config configs/tcpc/mads_tke_resume.py
- optimize_junction_tcpc_mads_tke_resume2.py -> scripts/tcpc_optimize.py --config configs/tcpc/mads_tke_resume2.py

Slurm launchers
- *.sh in this folder -> scripts/tcpc_submit.py --config <config>

Extra point evaluation
- evaluate_tcpc_extra_point.py / evaluate_tcpc_tke_extra_point.py -> scripts/tcpc_extra_point.py --config <config>
- run_extra_tcpc_point.sh / run_extra_tcpc_tke_point.sh -> scripts/tcpc_extra_point.py (use --config and env overrides)
- collect_extra_points.sh -> scripts/tcpc_collect_points.py --config configs/tcpc/nm_tcpc.py --points-csv configs/tcpc/points/extra_points_round1.csv
- collect_extra_points_round2.sh -> scripts/tcpc_collect_points.py --config configs/tcpc/nm_tcpc.py --points-csv configs/tcpc/points/extra_points_round2.csv
- collect_extra_points_mads_1pct.sh -> scripts/tcpc_collect_points.py --config configs/tcpc/mads_tcpc.py --points-csv configs/tcpc/points/extra_points_mads_1pct.csv
- collect_extra_points_tke_1pct.sh -> scripts/tcpc_collect_points.py --config configs/tcpc/nm_tke.py --points-csv configs/tcpc/points/extra_points_tke_1pct.csv
- collect_extra_points_parametric.sh -> scripts/tcpc_collect_points.py --config <config> --base <vals> --step <vals>

Single-case runner
- run_junction_tcpc.py -> scripts/tcpc_run_single.py

Split wrapper
- tcpc_split_wrapper.py -> scripts/tcpc_split_wrapper.py
