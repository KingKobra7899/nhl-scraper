import marimo

__generated_with = "0.18.1"
app = marimo.App()


@app.cell
def _():
    import pandas as pd

    rapm_2023 = pd.read_csv("rapm_2023.csv")
    rapm_2024 = pd.read_csv("rapm_2024.csv")
    rapm_2025 = pd.read_csv("rapm_2025.csv")

    rapm_2023['season'] = 2023
    rapm_2024['season'] = 2024
    rapm_2025['season'] = 2025

    combined_rapm = pd.concat([rapm_2023, rapm_2024, rapm_2025], ignore_index=True)
    combined_rapm["even_xgf_percentile"] = combined_rapm.groupby(["season", "position"])["even_xgf"].rank(pct=True)
    combined_rapm["even_xga_percentile"] = 1 - combined_rapm.groupby(["season", "position"])["even_xga"].rank(pct=True)
    combined_rapm["pp_xgf_percentile"] = combined_rapm.groupby(["season", "position"])["pp_xgf"].rank(pct=True)
    combined_rapm["pk_xga_percentile"] = 1 - combined_rapm.groupby(["season", "position"])["pk_xga"].rank(pct=True)
    return combined_rapm, pd


@app.cell
def _(combined_rapm):
    # standard deployments
    even = 1000
    pp = 100
    pk = 100

    even_rapm = (combined_rapm["even_xgf"] - combined_rapm["even_xga"]) * (even / 60)
    pp_rapm = (combined_rapm["pp_xgf"] * (pp / 60))
    pk_rapm = -(combined_rapm["pk_xga"] * (pk / 60))

    rapm = combined_rapm[['playerId', 'name', 'position', 'season']].copy()
    rapm['rapm'] = even_rapm + pp_rapm + pk_rapm
    rapm['evo'] = combined_rapm['even_xgf_percentile'] 
    rapm['evd'] = combined_rapm['even_xga_percentile']
    rapm['ppo'] = combined_rapm['pp_xgf_percentile']
    rapm['pkd'] = combined_rapm['pk_xga_percentile']
    rapm["rapm_rank"] = rapm.groupby(["season", "position"])["rapm"].rank(pct=True)
    return even, pk, pp


@app.cell
def _(combined_rapm, pd):
    import joblib
    boxscore = pd.concat([joblib.load("boxscore.pkl")['boxscore'], joblib.load("boxscore_2025.pkl")['boxscore']])

    boxscore["season"] = boxscore["game"].apply(lambda x: str(x)[0:4])
    boxscore["gp"] = 1
    player_stats = boxscore.groupby(['playerId', 'name', "position", 'season', 'situation'])[['goals', 'a1', 'a2', 'fenwick', 'sog', 'total_toi', 'penalties_drawn', 'penalties_taken', 'xG', 'xgf', "gp"]].sum().reset_index()
    player_stats["xG_team"] = player_stats['xgf'] - player_stats['xG']
    player_stats["assists"] = player_stats['a1'] + player_stats['a2']

    stats = ['goals', 'a1', 'a2', 'fenwick', 'sog', 'penalties_drawn', 'penalties_taken', 'xG', 'xgf', "assists", "xG_team"]

    player_stats[stats] = 60 * player_stats[stats].div(player_stats['total_toi'], axis=0)

    special_toi = 20  
    ev_toi    = 100   

    player_stats = player_stats[
        (
            player_stats["situation"].isin(["pp", "pk"])
            & (player_stats["total_toi"] >= special_toi)
        )
        |
        (
           (player_stats["situation"] == "ev")
            & (player_stats["total_toi"] >= ev_toi)
        )
    ]
    combined_rapm['playerId'] = combined_rapm['playerId'].apply(lambda x: int(x))
    player_stats['playerId'] = player_stats['playerId'].apply(lambda x: int(x))

    combined_rapm['season'] = combined_rapm['season'].apply(lambda x: int(x))
    player_stats['season'] = player_stats['season'].apply(lambda x: int(x))

    player_stats = player_stats.merge(combined_rapm, on=['name', 'playerId', 'position', 'season'])
    player_stats = player_stats.drop_duplicates(subset=['playerId', 'situation', 'season'])
    return (player_stats,)


@app.cell
def _(pd, player_stats):
    import seaborn as sns
    from sklearn.linear_model import LinearRegression
    x = pd.get_dummies(player_stats[["even_xgf", "xgf", "position"]],  columns=["position"])
    y = player_stats["assists"]
    model = LinearRegression()
    model.fit(x, y)
    player_stats["playmaking"] = player_stats["assists"] + model.predict(x)

    player_stats["playmaking_norm"] = (
        player_stats["playmaking"] -
        player_stats.groupby("situation")["playmaking"].transform("mean")
    ) / player_stats.groupby("situation")["playmaking"].transform("std", ddof=0)

    player_stats["finishing"] = player_stats["goals"] / player_stats["xG"]
    player_stats["pen_diff"] = player_stats["penalties_drawn"] - player_stats["penalties_taken"]
    already_split = [
        'even_xgf', 'even_xga',
        'pp_xgf', 'pk_xga',
        "gp"
    ]


    id_cols = ['playerId', 'name', 'position', 'season', 'situation']

    cols = ["finishing", "assists", "penalties_taken", "penalties_drawn", "pen_diff", 'total_toi']

    pivot_other = (
        player_stats.pivot_table(
            index=['playerId', 'name', 'position', 'season'],
            columns='situation',
            values=cols,
            aggfunc='first'
        )
    )


    pivot_other.columns = [f"{col}_{sit}" for col, sit in pivot_other.columns]

    pivot_other = pivot_other.reset_index()


    collapsed_split = (
        player_stats.groupby(['playerId', 'name', 'position', 'season'], as_index=False)[already_split]
          .agg(lambda x: x.dropna().iloc[0] if x.dropna().size else None)
    )


    combined = collapsed_split.merge(pivot_other, on=['playerId', 'name', 'position', 'season'])

    id_cols.remove("situation")
    combined = combined[id_cols +  already_split + ["finishing_ev", "pen_diff_ev", "finishing_pp", "assists_ev", "assists_pp", 'total_toi_ev', 'total_toi_pk',
           'total_toi_pp']]
    print(combined.columns)
    return LinearRegression, combined, id_cols


@app.cell
def _(LinearRegression, combined, even, pd, pk, pp):
    from sklearn.preprocessing import PolynomialFeatures
    import numpy as np
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)

    data = combined[combined.total_toi_ev >= 300]
    ev_x = pd.get_dummies(data[["even_xgf", "total_toi_ev", "position"]], columns=["position"])
    ev_x = poly.fit_transform(ev_x)
    ev_model = LinearRegression()
    ev_model.fit(ev_x, data["assists_ev"].values)

    combined["pmaking_ev"] = (5 * (ev_model.predict(poly.fit_transform(pd.get_dummies(combined[["even_xgf", "total_toi_ev", "position"]], columns=["position"]))) ) + 1 * (combined["assists_ev"])) / 6



    data = combined[combined.total_toi_pp >= 50]
    data = data.dropna(subset=["pp_xgf", "total_toi_pp", "position","assists_pp"])
    ev_x = pd.get_dummies(data[["pp_xgf", "total_toi_pp", "position"]], columns=["position"])



    ev_x = poly.fit_transform(ev_x)
    ev_model = LinearRegression()
    ev_model.fit(ev_x, data["assists_pp"].values)

    X = pd.get_dummies(
        combined[["even_xgf", "total_toi_pp", "position"]],
        columns=["position"]
    )

    nan_mask = X.isna().any(axis=1) | combined["assists_pp"].isna()

    pmaking_pp = pd.Series(np.nan, index=combined.index)

    valid_idx = ~nan_mask
    if valid_idx.any():
        X_valid = X.loc[valid_idx]
        pred_valid = ev_model.predict(poly.fit_transform(X_valid))
        pmaking_pp.loc[valid_idx] = (5 * pred_valid + combined.loc[valid_idx, "assists_pp"]) / 6

    combined["pmaking_pp"] = pmaking_pp

    combined["even_ovr"] = combined["even_xgf"] - combined["even_xga"]

    combined["off"] = even*combined["even_xgf"] + pp*combined["pp_xgf"]
    combined["def"] = even*combined["even_xga"] + pk*combined["pk_xga"]

    combined["ovr"] = combined["off"] - combined["def"]
    return


@app.cell
def _(combined, id_cols):
    rank_df = combined[id_cols + ["gp", "total_toi_ev", "total_toi_pp", "total_toi_pk"]].copy()
    rank_df["even_xgf"] = combined.groupby(["season", "position"])["even_xgf"].rank(pct=True)
    rank_df["even_xga"] = 1-combined.groupby(["season", "position"])["even_xga"].rank(pct=True)
    rank_df["even_ovr"] = combined.groupby(["season", "position"])["even_ovr"].rank(pct=True)

    rank_df["off"] = combined.groupby(["season", "position"])["off"].rank(pct=True)
    rank_df["def"] = 1-combined.groupby(["season", "position"])["def"].rank(pct=True)
    rank_df["ovr"] = combined.groupby(["season", "position"])["ovr"].rank(pct=True)

    rank_df
    return


if __name__ == "__main__":
    app.run()
