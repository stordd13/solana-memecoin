You are a Grok4-powered data scientist worker optimizing our Solana memecoin analysis pipeline for a personal trading bot. You have access to the existing codebase.

Today's Task: Build new unified_archetype_classifier.py in <1 hour on 30k: Copy from two_stage_sprint_classifier.py, remove stages for single 6-class pred (unified labels from clustering_comparison_results_30519.json), window=5 default, add class_weights='balanced', retrain with weighted scorer (0.7 recall + 0.3 precision for top pump clusters). Use 30k for train/test 80/20 stratified. Report precision/recall/F1 per class (focus top 2 pumps >60%). Aim F1 >0.6 overall, recall >0.65 on top clusters. Run vol on 30k with unified clusters (pumps % post-min 5 per cluster).

Features: 33 for w=5 (27 + 5 + acf + logs if not—add if missing).

Step-by-Step:

Create New Script (10 mins): Copy two_stage to unified_archetype_classifier.py, remove stage logic, predict 6 unified clusters (y = cluster 0-5).
Prep Data (5 mins): Load 30k, y = unified_cluster. Extract 33 features w=5. 80/20 stratified.
Re-Train (15 mins): XGB, Optuna (n_trials=10). Scorer weighted on recall for top 2. Add class_weights='balanced'. Eval precision/recall/F1 per class.
Vol on Unified Clusters (10 mins): Group by 6 clusters, high-vol pumps % >50% post-min 5, avg time_to_1.5x per cluster.
Outputs: Table precision/recall/F1 per class, vol dict + numbers per cluster. Saved model.