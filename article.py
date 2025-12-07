import marimo

__generated_with = "0.18.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    coef = pd.read_csv("decomp_rapm.csv")
    coef.columns
    return (coef,)


@app.cell
def _(coef):
    import seaborn as sns

    cols = ['even_xga_occurrence',  'even_xgf_occurrence', 'even_xga_magnitude',
           'even_xgf_magnitude']
    corr = coef[cols].corr()
    sns.heatmap(corr, square=True, vmin=-1, vmax=1, annot=True, cmap="seismic")
    return (cols,)


@app.cell
def _(coef, cols):
    orders = ["asc",  "desc",  "asc", "desc"]  

    group_cols = ["position", "season"]        

    df_out = coef.copy()

    for col, order in zip(cols, orders):
    
        r = (
            df_out.groupby(group_cols)[col]
            .rank(method="max", pct=True)
        )

        if order.lower() == "asc":
            # smallest value -> 1.0
            df_out[col] = 1.0 - r
        else:
            # largest value -> 1.0 (default behavior)
            df_out[col] = r
    df_out

    return (df_out,)


@app.cell
def _(coef, df_out):
    import matplotlib.pyplot as plt
    import nhl_scraper.player as pl
    def plot_player(playerId,df=coef, metrics = ["even_xga_occurrence", "even_xgf_magnitude"]):
    
        data = df.copy()
        data = data[data["playerId"] == playerId]
        data = data.sort_values("season")
        fig, ax = plt.subplots()
        for metric in metrics:
            ax.plot(data["season"], data[metric], label=metric, marker="o")
        ax.legend()
        ax.set_title(f"{pl.get_player_name(playerId)}")
        ax.set_xticks([2021, 2022, 2023, 2024, 2025])
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_ylim(0,1)
        ax.grid(axis='y')
    plot_player(df=df_out,playerId=8482109)
    plt.show()
    return


if __name__ == "__main__":
    app.run()
