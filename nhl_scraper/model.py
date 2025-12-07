import pandas as pd
from sklearn.linear_model import (
    Ridge,
    RidgeCV,
    LinearRegression,
    LogisticRegression,
    GammaRegressor,
)
import numpy as np
import scipy.sparse as sp
from sklearn.model_selection import GridSearchCV


def calculate_rapm(
    boxscore,
    df,
    metric,
    sit="even",
    n_alpha=10,
    ridge=True,
    cv=3,
    scorer="neg_mean_squared_error",
    per_60=True,
    fit_occ=True,
) -> pd.DataFrame:
    """ """
    assert sit in ["even", "pp", "pk", "all"]
    print(f"initial shifts: {df.shape[0]}")

    print(f"shifts after dropna: {df.shape[0]}")
    df = df.loc[:, (df != 0).any(axis=0)]  # remove players who
    df = df.copy()
    ev = ["5v5", "ev"]

    if sit == "even":
        df = df[df["manpower"].isin(ev)]
    elif sit == "pp":
        df = df[df["manpower"] == "pp"]
    else:
        df = df[df["manpower"] == "pk"]

    y = df[metric] / (df["duration"] / 60) if per_60 else df[metric]
    X = df.filter(regex=r"^(for|against)")

    X["zone"] = df["zone"].copy()
    X["isHome"] = df["isHome"].copy()
    X["duration"] = df["duration"].copy()

    X = pd.get_dummies(X, columns=["zone"])

    w = df["duration"]

    max_alpha = 5000
    print(f"max alpha: {max_alpha}")

    alphas_to_test = np.logspace(-4, np.log10(max_alpha), n_alpha)

    model = Ridge(solver="auto")
    if not ridge:
        model = LinearRegression()

    param_grid = {"alpha": alphas_to_test}
    if not ridge:
        param_grid = {}

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring=scorer,
        verbose=1,
        n_jobs=1 if ridge else -1,
    )
    print("fitting grid search")
    X = X.fillna(0)
    grid_search.fit(X, y, sample_weight=w)

    print(
        f"R^2: {grid_search.best_estimator_.score(X, y, sample_weight=w):.4f}",
        f" | Best alpha: {grid_search.best_params_.get('alpha', 'N/A')}",
    )
    out = pd.DataFrame(
        {
            "term": grid_search.feature_names_in_,
            "estimate": grid_search.best_estimator_.coef_,
        }
    )
    out_non_player = out[
        ~out["term"].str.startswith("for_") & ~out["term"].str.startswith("against_")
    ]
    out = out[out["term"].str.startswith("for_")].copy()
    out["term"] = out["term"].str.removeprefix("for_").copy()

    out["playerId"] = out["term"].apply(lambda x: float(x))
    out = out.merge(
        boxscore[["playerId", "name", "position", "teamId"]],
        on="playerId",
        how="right",
    ).drop_duplicates()
    out = out.rename(columns={"estimate": f"{sit}_{metric}"})
    out = out.drop_duplicates(subset=["playerId", "teamId"])
    out = out.drop(columns=["term"])
    print("Non-player effects (intercept, zone, isHome):")
    out_non_player = pd.concat(
        [
            out_non_player,
            pd.DataFrame(
                [
                    {"term": "intercept"},
                    {"estimate": grid_search.best_estimator_.intercept_},
                ]
            ),
        ],
        axis=0,
    )
    print(out_non_player)
    return out


import numpy as np


def hurdle_model(
    boxscore,
    df,
    metric,
    occ_metric="ff",
    sit="even",
    n_alphas=10,
    cv=3,
    per_60=True,
    fitOcc=True,
) -> pd.DataFrame:
    """
    Accounts for 0 inflation of most shift data by modeling occurrence and magnitude separately.
    1. Logistic regression to model occurrence of event
    2. Ridge regression on log of <metric> to model magnitude of event when it occurs

     - main difference is that we get multiplicative effects rather than additive
    3. final RAPM is P(occurrence) * E(magnitude | occurrence)

    Gives us for a player:
    - occurrence_rapm: how much a player increases the probability of an event occurring per 60 minutes
    - magnitude_rapm: how much a player increases the expected magnitude of an event per 60 minutes, given that it occurs

    Returns a dataframe with player RAPM for occurrence, magnitude, and combined
    """
    assert sit in ["even", "pp", "pk", "all"]

    df = df.loc[:, (df != 0).any(axis=0)]
    df["manpower"] = df["manpower"].astype("category")

    if sit == "even":
        df = df[df["manpower"].isin(["5v5", "ev"])]
    elif sit == "pp":
        df = df[df["manpower"] == "pp"]
    elif sit == "pk":
        df = df[df["manpower"] == "pk"]

    y = df[metric] / (df["duration"] / 60) if per_60 else df[metric]
    cols = [c for c in df.columns if c.startswith(("for", "against"))] + [
        "zone",
        "isHome",
        "duration",
    ]
    X = pd.get_dummies(df[cols], columns=["zone"], dtype=np.uint8)
    w = df["duration"].values

    lamdas = np.logspace(0, np.log10(500), n_alphas)

    if fitOcc:
        C = 1 / lamdas
        logit = LogisticRegression(penalty="l2", max_iter=1000)
        grid_search_logit = GridSearchCV(
            estimator=logit,
            param_grid={"C": C},
            cv=cv,
            scoring="neg_log_loss",
            n_jobs=1,
            verbose=2,
        )
        print("fitting logistic regression grid search")
        X = X.fillna(0)
        grid_search_logit.fit(X, df[occ_metric] > 0, sample_weight=w)
        print(
            f"Log Loss: {grid_search_logit.best_estimator_.score(X, y>0, sample_weight=w):.4f}",
            f" | Best C: {grid_search_logit.best_params_.get('C', 'N/A')}",
        )
        logit_out = pd.DataFrame(
            {
                "term": grid_search_logit.feature_names_in_,
                "occurrence": grid_search_logit.best_estimator_.coef_[0],
            }
        )

        logit_out = logit_out[logit_out["term"].str.startswith("for_")].copy()
        logit_out["term"] = logit_out["term"].str.removeprefix("for_").copy()
        logit_out["playerId"] = logit_out["term"].apply(lambda x: float(x))
        logit_out = logit_out.merge(
            boxscore[["playerId", "name", "position", "teamId"]],
            on="playerId",
            how="right",
        ).drop_duplicates()
        logit_out = logit_out.rename(
            columns={"occurrence": f"{sit}_{metric}_occurrence"}
        )
        logit_out = logit_out.drop_duplicates(subset=["playerId", "teamId"])
        logit_out = logit_out.drop(columns=["term"])
    gamma = Ridge(solver="auto")
    grid_search_gamma = GridSearchCV(
        estimator=gamma,
        param_grid={"alpha": lamdas},
        cv=cv,
        scoring="neg_mean_squared_error",
        verbose=2,
        n_jobs=1,
    )
    print("fitting gamma regression grid search")
    grid_search_gamma.fit(
        X[df[occ_metric] > 0],
        np.log(y[df[occ_metric] > 0]),
        sample_weight=w[df[occ_metric] > 0],
    )
    print(
        f"R^2: {grid_search_gamma.best_estimator_.score(X[df[occ_metric] > 0], np.log(y[df[occ_metric] > 0])):.4f}",
        f" | Best alpha: {grid_search_gamma.best_params_.get('alpha', 'N/A')}",
    )
    gamma_out = pd.DataFrame(
        {
            "term": grid_search_gamma.feature_names_in_,
            "magnitude_rapm": grid_search_gamma.best_estimator_.coef_,
        }
    )
    gamma_non_player = gamma_out[
        ~gamma_out["term"].str.startswith("for_")
        & ~gamma_out["term"].str.startswith("against_")
    ].copy()
    gamma_out = gamma_out[gamma_out["term"].str.startswith("for_")].copy()
    gamma_out["term"] = gamma_out["term"].str.removeprefix("for_").copy()
    gamma_out["playerId"] = gamma_out["term"].apply(lambda x: float(x))
    gamma_out = gamma_out.merge(
        boxscore[["playerId", "name", "position", "teamId"]],
        on="playerId",
        how="right",
    ).drop_duplicates()
    gamma_out = gamma_out.rename(
        columns={"magnitude_rapm": f"{sit}_{metric}_magnitude"}
    )
    gamma_out = gamma_out.drop_duplicates(subset=["playerId", "teamId"])
    gamma_out = gamma_out.drop(columns=["term"])
    if fitOcc:
        out = logit_out.merge(
            gamma_out, on=["playerId", "name", "position", "teamId"], how="outer"
        )
    else:
        out = gamma_out.copy()
    return out
