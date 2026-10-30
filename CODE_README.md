# EnMOGKSO: GNN-guided DQN local search for overlap-aware WSN deployment
Code, dataset and results for the CSE423 Deep Learning project (Srinivasan S R, 127184013).

## 1. Pipeline

```
Problem input (250 m x 300 m field, N = 50, Rs = 25 m)
   -> 01 Dataset generation (search-trace dataset, 65,000 rows)
   -> 02 Final test (30 paired seeds) + scalability + 5 metaheuristics
   -> 03 Component ablation          -> 04 Learned-controller tests (GNN / DQN)
   -> 05 Lloyd-CVT and VFA            -> 06 Connectivity + GNN test
   -> 07 CNN-DQN, Q-learning, GKSO, WOA-LFGA (main setting)
   -> 08-12 Robustness Studies A, B, C (same 18 methods)
   -> 13 Minimum number of sensors
   -> analysis/ merge + statistics + figures for Studies A-C
```

All scripts are self-contained Python files. Each one contains the full EnMOGKSO
code (Part A: original algorithm, Part B: learned ARLS with GNN + Move-DQN) followed
by the experiment it runs. They were run as single Kaggle notebook cells (CPU only).
Every run is saved immediately; re-running a cell resumes after the last saved run.

## 2. Code (folder `code/`)

| File | Report section | Main outputs |
|---|---|---|
| 01_dataset_generation.py | 3.2 Dataset | wsn_full_dataset_v10.csv, v10_final_results.csv, v10_raw_trace.csv |
| 02_final_test_scalability_metaheuristics.py | 4.1, 4.4, 4.6 | exp1_final_runs.csv, exp2_scalability.csv, exp3_baseline_runs.csv |
| 03_component_ablation.py | 4.2 | ablation50_runs.csv |
| 04_learned_controller_tests.py | 4.3 | p1_test_runs.csv, p1_pretrain_log.csv |
| 05_lloyd_cvt_vfa_scalability.py | 4.5, 4.6 | v8_wsn_baseline_runs.csv, v8_scalability_runs.csv |
| 06_connectivity_gnn_test.py | 4.3, 4.8 | v9_gnn_test_runs.csv, v9_connectivity_all_layouts.csv |
| 07_cnn_qlearning_gkso_woalfga.py | 4.3, 4.4 | p5_runs.csv |
| 08_study_A_fine_grid_margins.py | 4.9 Study A | p2_runs.csv, p2_layout_metrics.csv |
| 09_study_B_outside_penalty.py | 4.9 Study B | p3_runs.csv, p3_layout_metrics.csv |
| 10_study_C_irregular_field.py | 4.9 Study C | p4_runs.csv, p4_layout_metrics.csv |
| 11_study_A_common_methods.py | 4.9 (18 methods) | p9a_runs.csv, p9a_layout_metrics.csv |
| 12_study_BC_common_methods.py | 4.9 (18 methods) | p9bc_runs.csv, p9bc_layout_metrics.csv |
| 13_minimum_number_of_sensors.py | 4.10 | p6_minN_runs.csv, p6_prune_runs.csv |

Requirements: Python 3.10+, numpy, pandas, scipy, matplotlib, torch (CPU).

## 3. Analysis (folder `analysis/`)
Run from inside `analysis/`, in this order:
1. `python merge_studies_ABC.py` - merges all runs of Studies A-C into `merged_18.csv`
2. `python statistics_studies_ABC.py` - Wilcoxon + Holm (17 comparisons) and Friedman test -> `Studies_ABC_18methods_results.xlsx`
3. `python figures_studies_ABC.py` - 600 dpi figures of Studies A-C -> `figs/`

## 4. Dataset (folder `dataset/`)
No external labelled dataset exists; the optimiser generates its own data and the
GNN and DQNs are trained online inside every run.

| File | Rows | Content |
|---|---|---|
| wsn_full_dataset_v10.csv | 65,000 | 13 configurations x 5 seeds x 1,000 best-so-far checkpoints |
| wsn_full_dataset_v10_seed0.csv | 13,000 | 13 configurations x 1,000 checkpoints (seed 0) |
| v10_final_results.csv | 60 | final result of every run |
| v10_raw_trace.csv | 60,000 | raw search trace |

## 5. Results (folder `results/`)
- `main_setting/` - final test, scalability, baselines, ablation, controller tests, connectivity
- `robustness_studies/` - Studies A, B and C, including the merged 18-method table and statistics
- `min_sensors/` - minimum number of sensors and pruning runs
- `layouts/` - saved sensor positions (.npy) of the runs

Targets used throughout: coverage C > 98 %, overlap O < 10 %, EE > 97 %,
with EE = 0.82 C + 0.18 (1 - O).
