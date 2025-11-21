import numpy as np
import pandas as pd
from sklearn.utils import resample

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegressionCV
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import nhl_scraper.games as ga

shots = pd.read_csv("tests/shots.csv")
shots["season"] = shots["game"].apply(lambda x: int(str(x)[0:4]))
shots = shots[shots["season"] >= 2023]
shots = shots.dropna(
    subset=["xCoord", "yCoord", "shotType", "isHome", "timeRemaining", "periodType"]
)
shots["situation"] = shots.apply(
    lambda row: ga.getSituation(row["situationCode"], row["isHome"]), axis=1
)

shots = shots[
    ~shots["situation"].isin(
        [
            "0v1",
            "1v0",
            "1v4",
            "1v5",
            "3v1",
            "4v1",
            "3v6",
            "6v3",
            "5v1",
        ]
    )
]

X = pd.get_dummies(
    shots[
        [
            "xCoord",
            "yCoord",
            "shotType",
            "isHome",
            "timeRemaining",
            "periodType",
            "situation",
        ]
    ],
    columns=["periodType", "shotType", "situation"],
)
print(X.columns)
y = (shots["typeDescKey"] == "goal").values.astype(int)


print("training model")
model = GradientBoostingClassifier(n_estimators=300)
model.fit(X, y)

print("model trained")
import joblib

joblib.dump(model, "nhl_scraper/xg_model.joblib")

importances = model.feature_importances_
df_importances = pd.DataFrame({"feature": X.columns, "importance": importances})
df_importances = df_importances.sort_values(by="importance", ascending=False)
print(df_importances)
