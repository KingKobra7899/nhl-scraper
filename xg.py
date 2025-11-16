import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from nhl_scraper.games import draw_rink_features
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt

shots = pd.read_csv("shots.csv")
shots = shots.dropna(
    subset=["xCoord", "yCoord", "shotType", "isHome", "timeRemaining", "periodType"]
)

shots["score"] = np.where(
    shots["isHome"],
    shots["homeScore"] - shots["awayScore"],
    shots["awayScore"] - shots["homeScore"],
)

X = pd.get_dummies(
    shots[["xCoord", "yCoord", "shotType", "isHome", "timeRemaining", "periodType"]],
    columns=["periodType", "shotType"],
)

y = (shots["typeDescKey"] == "goal").values.astype(int)

print("training model")
model = GradientBoostingClassifier(n_estimators=200)
model.fit(X, y)
print("model trained")

shot_types = shots["shotType"].dropna().unique()
n_types = len(shot_types)
cols = 3
rows = (n_types + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))


cmap = "bwr_r"
shots["xG"] = model.predict_proba(X)[:, 1]
xg_available = "xG" in shots.columns
for ax, st in zip(axes.ravel(), shot_types):
    draw_rink_features(ax, color="black")
    subset = shots[shots["shotType"] == st]
    colors = subset["xG"]
    sc = ax.scatter(
        subset["xCoord"], subset["yCoord"], c=colors, cmap=cmap, s=20, alpha=0.1
    )
    ax.set_title(st)


for ax in axes.ravel()[n_types:]:
    ax.axis("off")

plt.tight_layout()
if xg_available:
    cbar = fig.colorbar(
        sc, ax=axes.ravel(), orientation="vertical", fraction=0.02, pad=0.02
    )
    cbar.set_label("xG")
fig.savefig("shots.png")
plt.show()
