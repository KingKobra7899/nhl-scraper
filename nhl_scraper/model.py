import pandas as pd
from sklearn.linear_model import Ridge


def calculate_rapm(boxscore, df, metric, sit="even") -> pd.DataFrame:
    """ """
    assert sit in ["even", "pp", "pk", "all"]
    df = df.loc[:, (df != 0).any(axis=0)] #remove players who 
    df = df.copy()
    ev = ["5v5", "ev"]


    if sit == "even":
        df = df[df["manpower"].isin(ev)]
    elif sit == "pp":
        df = df[df["manpower"] == 'pp']
    else:
        df = df[df["manpower"] == 'pk']
    y = df[metric] / (df["duration"] / 60)
    X = df.filter(regex=r"^(for|against)")
    
    X["zone"] = df["zone"].copy()
    X["team"] = df["teamId"].copy()
    X["isHome"] = df["isHome"].copy()
    X["duration"] = df["duration"].copy()
    
    X = X.fillna(0)
    X = pd.get_dummies(X, columns=["zone"])
    w = df["duration"]

    model = Ridge(alpha=1500, 
                  solver='sparse_cg', 
                  max_iter=1000)
    model.fit(X, y, sample_weight=w)

    out = pd.DataFrame({"term": model.feature_names_in_, "estimate": model.coef_})

    out = out[out["term"].str.startswith("for_")].copy()
    out["term"] = out["term"].str.removeprefix("for_").copy()

    out["playerId"] = out["term"].apply(lambda x: float(x))
    out = out.merge(
        boxscore[["playerId", "name", "position", "teamId"]],
        on="playerId",
        how="right",
    ).drop_duplicates()
    out = out.rename(columns={"estimate": f"{sit}_{metric}"})
    return out
