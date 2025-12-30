# Junction TCPC CSV merge map

Merged outputs are sorted by `timestamp_iso8601` and have `eval_id` renumbered
to consecutive values starting at 1. Duplicates were removed when all numeric
fields (objective + parameters) matched after rounding to 12 decimals; the
earliest timestamp was kept.

## Merged outputs
- `data/junction_tcpc_logs/merged/mads_tke_merged.csv`
  - Sources:
    - `data/junction_tcpc_logs/run_a4f79f0d_mads_tke_20251205-202008.csv`
    - `data/junction_tcpc_logs/run_c3a3c343_mads_tke_resume_20251214-225300.csv`
    - `data/junction_tcpc_logs/run_b6487a62_mads_tke_resume2_20251222-121026.csv`
  - Rows: 59 -> 58 (1 duplicate removed)
- `data/junction_tcpc_logs/merged/nelder_mead_tke_merged.csv`
  - Sources:
    - `data/junction_tcpc_logs/run_eec59623_nelder_mead_tke_20251202-105326.csv`
    - `data/junction_tcpc_logs/run_5f3fce47_nelder_mead_tke_resume_20251210-233836.csv`
  - Rows: 70 -> 70 (0 duplicates removed)
- `data/junction_tcpc_logs/merged/nelder_mead_merged.csv`
  - Sources:
    - `data/junction_tcpc_logs/run_6df0829c_nelder_mead_20251119-162839.csv`
    - `data/junction_tcpc_logs/run_276186ab_nelder_mead_resume_20251125-075917.csv`
  - Rows: 72 -> 72 (0 duplicates removed)
- `data/junction_tcpc_logs/merged/mads_merged.csv`
  - Sources:
    - `data/junction_tcpc_logs/run_5390b29d_mads_20251120-173542.csv`
    - `data/junction_tcpc_logs/run_b037b6ef_mads_20251128-155751.csv`
  - Rows: 63 -> 63 (0 duplicates removed)
- `data/junction_tcpc_logs/merged/extra_points_mads_1pct_merged.csv`
  - Sources:
    - `data/junction_tcpc_logs/extra_points_mads_1pct_extra_points_mads_1pct_20251217-162208.csv`
    - `data/junction_tcpc_logs/extra_points_mads_1pct_extra_points_mads_1pct_20251217-162210.csv`
    - `data/junction_tcpc_logs/extra_points_mads_1pct_extra_points_mads_1pct_20251217-162214.csv`
    - `data/junction_tcpc_logs/extra_points_mads_1pct_extra_points_mads_1pct_20251218-020311.csv`
    - `data/junction_tcpc_logs/extra_points_mads_1pct_extra_points_mads_1pct_20251218-073212.csv`
    - `data/junction_tcpc_logs/extra_points_mads_1pct_extra_points_mads_1pct_20251218-073312.csv`
    - `data/junction_tcpc_logs/extra_points_mads_1pct_extra_points_mads_1pct_20251218-093912.csv`
    - `data/junction_tcpc_logs/extra_points_mads_1pct_extra_points_mads_1pct_20251218-114612.csv`
    - `data/junction_tcpc_logs/extra_points_mads_1pct_extra_points_mads_1pct_20251218-114617.csv`
  - Rows: 10 -> 10 (0 duplicates removed)
- `data/junction_tcpc_logs/merged/extra_points_tke_1pct_merged.csv`
  - Sources:
    - `data/junction_tcpc_logs/extra_points_tke_1pct_extra_points_tke_1pct_20251215-112926.csv`
    - `data/junction_tcpc_logs/extra_points_tke_1pct_extra_points_tke_1pct_20251217-162142.csv`
    - `data/junction_tcpc_logs/extra_points_tke_1pct_extra_points_tke_1pct_20251217-162144.csv`
    - `data/junction_tcpc_logs/extra_points_tke_1pct_extra_points_tke_1pct_20251217-162145.csv`
    - `data/junction_tcpc_logs/extra_points_tke_1pct_extra_points_tke_1pct_20251217-162146.csv`
  - Rows: 14 -> 10 (4 duplicates removed)

## Other CSVs (already consolidated, different schema)
- `data/extra_points/extra_points_objectives.csv`
- `data/extra_points/extra_points_objectives_round2.csv`
- `data/extra_points/extra_points_mads_1pct.csv`
- `data/extra_points/extra_points_tke_1pct.csv`
