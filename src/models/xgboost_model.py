import xgboost as xgb


def build_xgboost_model(scale_pos_weight: float):
    """
    Membuat model XGBoost untuk binary fire prediction.
    Konfigurasi ini dipisah supaya train_xgboost.py lebih bersih.
    """

    model = xgb.XGBClassifier(
        n_estimators=800,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=5,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=2.0,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
    )