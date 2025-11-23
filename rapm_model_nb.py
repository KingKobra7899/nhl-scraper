import marimo

__generated_with = "0.18.0"
app = marimo.App()


@app.cell
def _():
    import joblib
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import RidgeCV, Ridge
    data = joblib.load("boxscore.pkl")

    import pandas as pd
    from warnings import simplefilter
    simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
    return Ridge, data, pd


@app.cell
def _(Ridge, data, pd):
    def calculate_rapm(df, metric, sit="even") -> pd.DataFrame:
        """
        """
        assert sit in ["even","pp","pk", "all"]
        df = df.copy()
        ev = ["5v5", "4v4", "3v3"]
        pp = ["5v4", "4v3", "5v3"]
        pk = ["4v5", "3v4", "3v5"]

        if sit == "even":
            df = df[df["manpower"].isin(ev)]
        elif sit == "pp":
            df = df[df["manpower"].isin(pp)]
        else:
            df = df[df["manpower"].isin(pk)]
        y = df[metric] / (df["duration"] / 60)
        X = df.filter(regex=r'^(for|against)')
        X["zone"] = df["zone"].copy()
        X["manpower"] = df["manpower"].copy()
        X = X.fillna(0)
        X = pd.get_dummies(X, columns=["zone", "manpower"])
        w = df["duration"]

        model = Ridge(alpha=1500)
        model.fit(X, y, sample_weight=w)

        out = pd.DataFrame({
            "term": model.feature_names_in_,
            "estimate": model.coef_
        })


        out = out[out["term"].str.startswith("for_")].copy()
        out["term"] = out["term"].str.removeprefix("for_").copy()

        out["playerId"] = out["term"].apply(lambda x:float(x))
        out = out.merge(data["boxscore"][["playerId", "name"]],   on="playerId",how="right").drop_duplicates()
        out = out.rename(columns={'estimate':f"{sit}_{metric}"})
        return out
    return (calculate_rapm,)


@app.cell
def _(calculate_rapm, data):
    xgf_rapm = calculate_rapm(data["stints"], "xgf", "even")
    xga_rapm = calculate_rapm(data["stints"], "xga", "even")
    ppo = calculate_rapm(data["stints"], "xgf", "pp")
    pkd = calculate_rapm(data["stints"], "xga", "pk")
    return pkd, ppo, xga_rapm, xgf_rapm


@app.cell
def _(pkd, ppo, xga_rapm, xgf_rapm):
    import seaborn as sns
    import matplotlib.pyplot as plt
    from adjustText import adjust_text

    rapm = xgf_rapm
    rapm = rapm.merge(xga_rapm[["playerId", "even_xga"]], on="playerId")
    rapm = rapm.merge(ppo[["playerId", "pp_xgf"]], on="playerId")
    rapm = rapm.merge(pkd[["playerId", "pk_xga"]], on="playerId")
    rapm = rapm.drop_duplicates(subset="playerId")
    def plot_rapm(df, players, x, y, xlabel, ylabel, title): 
        sns.set_style("ticks")
        df = df.copy()
        df["highlight"] = df["playerId"].isin(players)

        g = sns.jointplot(
            data=df, 
            x=x, 
            y=y, 
            height=6,
       
            s=30,
            alpha=0.25,
            color="#505A5B",edgecolor=None
        )

        ax = g.ax_joint

        ax.axhline(0, color="#9e9e9e", linewidth=0.8, linestyle="--", alpha=0.7, zorder=1)
        ax.axvline(0, color="#9e9e9e", linewidth=0.8, linestyle="--", alpha=0.7, zorder=1)

        highlight = df[df["playerId"].isin(players)]
        highlight = highlight.drop_duplicates(subset="playerId")
        ax.scatter(
            highlight[x],
            highlight[y],
            s=40,
            color="#6CCFF6",
            zorder=100
        )

        texts = []
        for _, row in highlight.iterrows():
            t = ax.text(
                s=row["name"],
                x=row[x], 
                y=row[y],
                fontsize=8,
                zorder=101
            )
            texts.append(t)

        adjust_text(
            texts, 
            expand_points=(1, 1),
            arrowprops=dict(
                arrowstyle="-", 
                color="black",
                lw=1
            ),
            ax=ax
        )

        ax.invert_yaxis()
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        g.fig.suptitle(title)
        g.fig.tight_layout()
        plt.show()
    return plot_rapm, rapm


@app.cell
def _(rapm):
    rapm
    return


@app.cell
def _(plot_rapm, rapm):
    import nhl_scraper.player as p
    plot_rapm(df=rapm, players=p.getTeamSeasonRoster(team="NYR", season=20252026, positions=['defensemen']),x="even_xgf",y="even_xga", xlabel="xGF RAPM", ylabel="xGA RAPM (Inverted)", title="Even Strength RAPM 2024-2025")
    return


if __name__ == "__main__":
    app.run()
