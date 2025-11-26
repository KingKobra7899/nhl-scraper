import marimo

__generated_with = "0.18.0"
app = marimo.App()


@app.cell
def _():
    import joblib
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import RidgeCV, Ridge
    from warnings import simplefilter

    simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

    data = joblib.load("boxscore.pkl")
    data_2025 = joblib.load("boxscore_2025.pkl")
    bs = data["boxscore"]
    stints = data["stints"]
    return bs, data_2025, stints


@app.cell
def _(bs, data_2025, stints):
    bs["season"] = bs["game"].apply(lambda x: str(x)[0:4])
    stints["season"] = stints["game"].apply(lambda x: str(x)[0:4])

    bs_2023 = bs[bs["season"] == "2023"]
    stints_2023 = stints[stints["season"] == "2023"]

    bs_2024 = bs[bs["season"] == "2024"]
    stints_2024 = stints[stints["season"] == "2024"]

    bs_2025 = data_2025["boxscore"]
    stints_2025 = data_2025["stints"]
    return bs_2023, bs_2024, bs_2025, stints_2023, stints_2024, stints_2025


@app.cell
def _(
    bs_2023,
    bs_2024,
    bs_2025,
    calculate_rapm,
    stints_2023,
    stints_2024,
    stints_2025,
):
    import nhl_scraper.model as m

    xgf_rapm_2023 = m.calculate_rapm(bs_2023, stints_2023, "xgf", "even")
    xga_rapm_2023 = calculate_rapm(bs_2023, stints_2023, "xga", "even")
    ppo_2023 = m.calculate_rapm(bs_2023, stints_2023, "xgf", "pp")
    pkd_2023 = m.calculate_rapm(bs_2023, stints_2023, "xga", "pk")
    print("2023-2024 complete")
    xgf_rapm_2024 = m.calculate_rapm(bs_2024, stints_2024, "xgf", "even")
    xga_rapm_2024 = calculate_rapm(bs_2024, stints_2024, "xga", "even")
    ppo_2024 = m.calculate_rapm(bs_2024, stints_2024, "xgf", "pp")
    pkd_2024 = m.calculate_rapm(bs_2024, stints_2024, "xga", "pk")
    print("2024-2025 complete")
    xgf_rapm_2025 = m.calculate_rapm(bs_2025, stints_2025, "xgf", "even")
    xga_rapm_2025 = calculate_rapm(bs_2025, stints_2025, "xga", "even")
    ppo_2025 = m.calculate_rapm(bs_2025, stints_2025, "xgf", "pp")
    pkd_2025 = m.calculate_rapm(bs_2025, stints_2025, "xga", "pk")
    print("2025-2026 complete")
    return (xgf_rapm_2025,)


@app.cell
def _(xgf_rapm_2025):
    xgf_rapm_2025
    return


@app.cell
def _(rapm):
    import nhl_scraper.player as p
    import nhl_scraper.graphs as gr

    gr.plot_rapm(
        df=rapm,
        players=p.getTeamSeasonRoster(
            team="NYR", season=20252026, positions=["defensemen"]
        ),
        x="even_xgf",
        y="even_xga",
        xlabel="xGF RAPM",
        ylabel="xGA RAPM (Inverted)",
        title="Even Strength RAPM 2024-2025",
    )
    return


if __name__ == "__main__":
    app.run()
