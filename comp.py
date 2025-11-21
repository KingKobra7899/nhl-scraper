import pandas as pd
import joblib
import nhl_scraper.games as ga
from sklearn.metrics import *
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
import numpy as np

mp = pd.read_csv("evolving-hockey/shots_2025.csv")
mp = mp[["event", "xGoal"]]
mp["isGoal"] = mp["event"] == "GOAL"
mp["xG"] = mp["xGoal"]
mp = mp[["isGoal", "xG"]]
eh = pd.read_csv("evolving-hockey/EH_pbp_query_20252026_2025-11-15.csv")
eh = eh[["event_type", "pred_goal"]]
eh["isGoal"] = eh["event_type"] == "GOAL"
eh["xG"] = eh["pred_goal"]
eh = eh[["isGoal", "xG"]]
eh = eh[~eh["xG"].isna()]
shts = pd.read_csv("tests/current_season.csv")

shts["situation"] = shts.apply(
    lambda row: ga.getSituation(row["situationCode"], row["isHome"]), axis=1
)
shts = shts[
    ~shts["situation"].isin(
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
    shts[
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
cols = [
    "xCoord",
    "yCoord",
    "isHome",
    "timeRemaining",
    "periodType_OT",
    "periodType_REG",
    "shotType_backhand",
    "shotType_bat",
    "shotType_between-legs",
    "shotType_cradle",
    "shotType_deflected",
    "shotType_poke",
    "shotType_slap",
    "shotType_snap",
    "shotType_tip-in",
    "shotType_wrap-around",
    "shotType_wrist",
    "situation_3v3",
    "situation_3v4",
    "situation_3v5",
    "situation_4v3",
    "situation_4v4",
    "situation_4v5",
    "situation_4v6",
    "situation_5v3",
    "situation_5v4",
    "situation_5v5",
    "situation_5v6",
    "situation_6v4",
    "situation_6v5",
]
x = X.reindex(columns=cols, fill_value=0)

xg_model = joblib.load("nhl_scraper/xg_model.joblib")

ucn = pd.DataFrame(
    {"isGoal": shts["typeDescKey"] == "goal", "xG": xg_model.predict_proba(x)[:, 1]}
)


def ECE(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bins[i]
        bin_upper = bins[i + 1]

        in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)
        bin_size = np.sum(in_bin)

        if bin_size == 0:
            continue

        avg_pred = np.mean(y_prob[in_bin])
        avg_true = np.mean(y_true[in_bin])

        ece += (bin_size / len(y_prob)) * abs(avg_pred - avg_true)

    return ece


def grade_xg(name: str, df: pd.DataFrame) -> pd.DataFrame:
    brier = brier_score_loss(df["isGoal"], df["xG"])
    ll = log_loss(df["isGoal"], df["xG"])
    auc = roc_auc_score(df["isGoal"], df["xG"])
    ece = ECE(df["isGoal"], df["xG"], n_bins=10)
    return pd.DataFrame(
        [
            {
                "model": name,
                "brier": brier / 2,
                "log_loss": ll,
                "ece": ece,
                "roc_auc": auc,
            }
        ]
    )


final_df = pd.concat(
    [
        grade_xg("MoneyPuck", mp),
        grade_xg("EvolvingHockey", eh),
        grade_xg("Mine", ucn),
    ]
)

final_df["brier"] = final_df["brier"].apply(lambda x: f"{x:.3f}")
final_df["log_loss"] = final_df["log_loss"].apply(lambda x: f"{x:.3f}")
final_df["ece"] = final_df["ece"].apply(lambda x: f"{x:.3f}")
final_df["roc_auc"] = final_df["roc_auc"].apply(lambda x: f"{x:.3f}")


final_df = final_df.rename(
    columns={
        "model": "     ",
        "brier": "Brier",
        "log_loss": "Log Loss",
        "roc_auc": "ROC AUC",
        "ece": "ECE",
    }
)

fig, ax = plt.subplots(figsize=(10, 4))
ax.axis("off")

cell_text = final_df.values.tolist()
for col_idx in range(1, final_df.shape[1]):
    col = final_df.iloc[:, col_idx]
    if col_idx > 3:
        max_val = col.max()
    else:
        max_val = col.min()
    for row_idx, val in enumerate(col):
        if val == max_val:
            cell_text[row_idx][col_idx] = r"$\bf{" + str(val) + "}$"

table = ax.table(
    cellText=cell_text,
    colLabels=final_df.columns,
    cellLoc="center",
    loc="center",
)

table.auto_set_font_size(False)
table.set_fontsize(12)
table.auto_set_column_width(col=list(range(final_df.shape[1])))

plt.tight_layout()
plt.show()
fig.set_dpi(300)
fig.savefig("img/sigm_table.png")


def plot_calibration_curve(dfs, names, n_bins=10):
    fig = plt.figure(figsize=(8, 8))

    for df, name in zip(dfs, names):
        y_true = df["isGoal"].values
        y_prob = df["xG"].values

        # Compute calibration curve
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)

        # Estimate ECE
        bin_sizes = np.histogram(y_prob, bins=n_bins)[0]
        try:
            ece = np.sum(np.abs(prob_true - prob_pred) * bin_sizes) / np.sum(bin_sizes)
        except:
            print(prob_true, prob_pred)

        # Plot curve
        plt.plot(prob_pred, prob_true, marker="o", label=f"{name} (ECE={ece:.3f})")

    # Perfectly calibrated line
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")

    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration Curves with ECE")
    plt.legend()
    plt.grid(True)
    plt.show()
    fig.set_dpi(300)
    fig.savefig("img/sigm_cal.png")


def plot_roc_curves(dfs, names):
    fig = plt.figure(figsize=(8, 8))

    for df, name in zip(dfs, names):
        y_true = df["isGoal"].values
        y_score = df["xG"].values

        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")

    # Diagonal line for random classifier
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.grid(True)
    plt.show()
    fig.set_dpi(300)
    fig.savefig("img/sigm_auc.png")


plot_calibration_curve([mp, eh, ucn], ["MoneyPuck", "EvolvingHockey", "Mine"])
plot_roc_curves([mp, eh, ucn], ["MoneyPuck", "EvolvingHockey", "Mine"])
