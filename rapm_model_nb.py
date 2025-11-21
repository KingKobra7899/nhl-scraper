import marimo

__generated_with = "0.18.0"
app = marimo.App()


@app.cell
def _():
    import joblib
    import pandas as pd
    from sklearn.linear_model import RidgeCV, Ridge
    data = joblib.load("boxscore.pkl")
    return RidgeCV, data, pd


@app.cell
def _(data):
    data['stints'].columns
    return


@app.cell
def _(RidgeCV, pd):
    def calculate_rapm(df, metric):
        y = df[metric] / (df["duration"] / 60)
        X = df.filter(regex=r'^(for|against)')
        X = X.fillna(0)
        w = df["duration"]

        model = RidgeCV(cv=10, alphas = [2**n for n in range(-1, 3)])
        model.fit(X, y, sample_weight=w)

        out = pd.DataFrame({
            "term": model.feature_names_in_,
            "estimate": model.coef_
        })

    
        out = out[out["term"].str.startswith("for_")].copy()
        out["term"] = out["term"].str.removeprefix("for_")
        return out

    
    return (calculate_rapm,)


@app.cell
def _(calculate_rapm, data):
    data["stints"]["gsax"] = data["stints"]["gf"] - data["stints"]["xgf"]
    off_model = calculate_rapm(data['stints'], 'gsax')
    return (off_model,)


@app.cell
def _(data, off_model):
    off_model["playerId"] = off_model["term"].apply(lambda x: float(x))
    off_model.merge(data["boxscore"][["playerId", "name"]], on="playerId", how="right").drop_duplicates()
    return


if __name__ == "__main__":
    app.run()
