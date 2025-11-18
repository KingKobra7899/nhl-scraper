import pandas as pd
import joblib
import nhl_scraper.games as ga
from sklearn.metrics import *
import matplotlib.pyplot as plt

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
            "venue",
        ]
    ],
    columns=["periodType", "shotType", "situation", "venue"],
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
    "situation_6v3",
    "situation_6v4",
    "situation_6v5",
    "venue_Amalie Arena",
    "venue_Amerant Bank Arena",
    "venue_American Airlines Center",
    "venue_Avicii Arena",
    "venue_BB&T Center",
    "venue_Ball Arena",
    "venue_Bell MTS Place",
    "venue_Bridgestone Arena",
    "venue_Canada Life Centre",
    "venue_Canadian Tire Centre",
    "venue_Capital One Arena",
    "venue_Carter-Finley Stadium",
    "venue_Centre Bell",
    "venue_Climate Pledge Arena",
    "venue_Commonwealth Stadium, Edmonton",
    "venue_Crypto.com Arena",
    "venue_Delta Center",
    "venue_Edgewood Tahoe Resort",
    "venue_Enterprise Center",
    "venue_FLA Live Arena",
    "venue_Fenway Park",
    "venue_Gila River Arena",
    "venue_Honda Center",
    "venue_KeyBank Center",
    "venue_Lenovo Center",
    "venue_Little Caesars Arena",
    "venue_Madison Square Garden",
    "venue_MetLife Stadium",
    "venue_Mullett Arena",
    "venue_Nassau Veterans Memorial Coliseum",
    "venue_Nationwide Arena",
    "venue_Nissan Stadium",
    "venue_Nokia Arena",
    "venue_O2 Czech Republic",
    "venue_Ohio Stadium",
    "venue_PNC Arena",
    "venue_PPG Paints Arena",
    "venue_Prudential Center",
    "venue_Rogers Arena",
    "venue_Rogers Place",
    "venue_SAP Center at San Jose",
    "venue_STAPLES Center",
    "venue_Scotiabank Arena",
    "venue_Scotiabank Saddledome",
    "venue_T-Mobile Arena",
    "venue_T-Mobile Park",
    "venue_TD Garden",
    "venue_Target Field",
    "venue_Tim Hortons Field",
    "venue_UBS Arena",
    "venue_United Center",
    "venue_Wells Fargo Center",
    "venue_Wrigley Field",
    "venue_Xcel Energy Center",
]
x = X.reindex(columns=cols, fill_value=0)

xg_model = joblib.load("nhl_scraper/xg_model.joblib")

ucn = pd.DataFrame(
    {"isGoal": shts["typeDescKey"] == "goal", "xG": xg_model.predict_proba(x)[:, 1]}
)


def grade_xg(name: str, df: pd.DataFrame) -> pd.DataFrame:
    brier = brier_score_loss(df["isGoal"], df["xG"])
    ll = log_loss(df["isGoal"], df["xG"])
    prec = precision_score(df["isGoal"], round(df["xG"]))
    acc = accuracy_score(df["isGoal"], round(df["xG"]))
    auc = roc_auc_score(df["isGoal"], df["xG"])

    return pd.DataFrame(
        [
            {
                "model": name,
                "brier": brier / 2,
                "log_loss": ll,
                "precision": prec,
                "accuracy": acc,
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
final_df["roc_auc"] = final_df["roc_auc"].apply(lambda x: f"{x:.3f}")
final_df["precision"] = final_df["precision"].apply(lambda x: round((x * 100), 2))
final_df["accuracy"] = final_df["accuracy"].apply(lambda x: round((x * 100), 2))


final_df = final_df.rename(
    columns={
        "model": "     ",
        "brier": "Brier",
        "log_loss": "Log Loss",
        "roc_auc": "ROC AUC",
        "precision": "Precision",
        "accuracy": "Accuracy",
    }
)

fig, ax = plt.subplots(figsize=(10, 4))
ax.axis("off")

cell_text = final_df.values.tolist()
for col_idx in range(1, final_df.shape[1]):
    col = final_df.iloc[:, col_idx]
    if col_idx > 2:
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
