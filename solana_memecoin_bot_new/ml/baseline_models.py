import polars as pl
from xgboost import XGBClassifier
from sklearn.metrics import f1_score

def train_baseline(df: pl.DataFrame, use_archetypes: bool = True, archetype_filter: str = 'Sprint') -> dict:
    """
    Baselines with optional no-archetype mode (direct on all data to challenge clustering).
    Filter to 'Sprint' for small start; class weights for imbalance. Targets >35% F1.
    """
    models = {}
    metrics = {}
    features = ["scaled_returns", "ma_5", "rsi_14", "imbalance_ratio", "initial_dump_flag", "acf_lag_1"]
    
    if use_archetypes:
        archetypes = df["archetype"].unique()
        if archetype_filter:
            archetypes = [archetype_filter]
        for arch in archetypes:
            subset = df.filter(pl.col("archetype") == arch)
            train = subset.filter(pl.col("split") == "train")
            test = subset.filter(pl.col("split") == "test")
            X_train, y_train = train.select(features).to_numpy(), train["pump_label"].to_numpy()
            X_test, y_test = test.select(features).to_numpy(), test["pump_label"].to_numpy()
            pos_weight = len(y_train) / (sum(y_train) + 1e-6)
            xgb = XGBClassifier(scale_pos_weight=pos_weight, random_state=42)
            xgb.fit(X_train, y_train)
            preds = xgb.predict(X_test)
            f1 = f1_score(y_test, preds)
            metrics[arch] = {"f1": f1}
            models[arch] = xgb
            print(f"Archetype {arch} F1: {f1:.2%}")
    else:
        # Direct mode
        train = df.filter(pl.col("split") == "train")
        test = df.filter(pl.col("split") == "test")
        X_train, y_train = train.select(features).to_numpy(), train["pump_label"].to_numpy()
        X_test, y_test = test.select(features).to_numpy(), test["pump_label"].to_numpy()
        pos_weight = len(y_train) / (sum(y_train) + 1e-6)
        xgb = XGBClassifier(scale_pos_weight=pos_weight, random_state=42)
        xgb.fit(X_train, y_train)
        preds = xgb.predict(X_test)
        f1 = f1_score(y_test, preds)
        metrics['direct'] = {"f1": f1}
        models['direct'] = xgb
        print(f"Direct (No Archetypes) F1: {f1:.2%}")

    return {"models": models, "metrics": metrics}