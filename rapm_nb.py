import marimo

__generated_with = "0.18.0"
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
    return (combined_rapm,)


app._unparsable_cell(
    r"""
    # standard deployments
    even = 1000
    pp = 100
    pk = 100

    even_rapm = (combined_rapm[\"even_xgf\"] - combined_rapm[\"even_xga\"]) * (even / 60)
    pp_rapm = (combined_rapm[\"pp_xgf\"] * (pp / 60))
    pk_rapm = -(combined_rapm[\"pk_xga\"] * (pk / 60))

    rapm = combined_rapm[['playerId', 'name', 'position', 'season']].copy()
    rapm['rapm'] = even_rapm + pp_rapm pk_rapm
    rapm['evo'] = combined_rapm['even_xgf_percentile'] 
    rapm['evd'] = combined_rapm['even_xga_percentile']
    rapm['ppo'] = combined_rapm['pp_xgf_percentile']
    rapm['pkd'] = combined_rapm['pk_xga_percentile']
    rapm[\"rapm_rank\"] = rapm.groupby([\"season\", \"position\"])[\"rapm\"].rank(pct=True)
    """,
    name="_"
)


@app.cell
def _(rapm):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import numpy as np
    import nhl_scraper.player as p
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import numpy as np

    def plot_player_rapm(player_name, cmap=None):
        player_data = rapm[rapm['name'] == player_name]
        if player_data.empty:
            player_data = rapm[rapm['playerId'] == player_name]
            player_name = p.get_player_name(player_name)
        if player_data.empty:
            print("Player not found.")
            return

        plt.figure(figsize=(10, 6))

        metrics = {
            'rapm_rank': 'Overall RAPM',
            'evo': 'Even Strength Offense',
            'evd': 'Even Strength Defense',
            'ppo': 'Power Play Offense',
            'pkd': 'Penalty Kill Defense'
        }

        # Default simple color cycle
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        color_iter = iter(color_cycle)

        # === DRAW LINES ===
        for col, label in metrics.items():
            df = player_data.sort_values('season')
            alpha = 1 if col == "rapm_rank" else 0.5
            zorder = 2 if col == "rapm_rank" else 1

            color = next(color_iter)

            solid_df = df[df['season'] < 2025]
            if len(solid_df) >= 2:
                plt.plot(
                    solid_df['season'], solid_df[col],
                    marker='o', linewidth=3, alpha=alpha,
                    zorder=zorder, label=label, color=color
                )
            elif len(solid_df) == 1:
                plt.plot(
                    solid_df['season'], solid_df[col],
                    marker='o', color=color
                )

            # dotted projected segment
            if 2025 in df['season'].values:
                seg = df[df['season'].isin([df['season'].max() - 1, 2025])]
                if len(seg) == 2:
                    plt.plot(
                        seg['season'], seg[col],
                        marker='o', linestyle=':',
                        linewidth=3, alpha=alpha,
                        zorder=zorder, color=color
                    )

        seasons = sorted(player_data['season'].unique())
        n_metrics = len(metrics)

        # === TILE PANEL BELOW AXIS ===
        ax = plt.gca()
        ax.set_ylim(-0.05, 1.05)

        # Reserve extra vertical space
        pad_height = 0.12 + n_metrics * 0.06
        ax.set_ylim(-pad_height, 1.05)

        tile_height = 0.05
        base_y = -0.16 

        # Pick colormap or fallback to uniform tiles
        if cmap is not None:
            cmap_obj = plt.get_cmap(cmap)
        else:
            cmap_obj = None

        for r, (col, _) in enumerate(metrics.items()):
            for s in seasons:
                val = player_data[player_data['season'] == s][col].iloc[0]

                if cmap_obj:
                    color = cmap_obj(val)
                    # choose readable text color
                    brightness = mcolors.rgb_to_hsv(color[:3])[2]
                    text_color = "black" if brightness > 0.55 else "white"
                else:
                    # no color-coding
                    color = "lightgray"
                    text_color = "black"

                rect = plt.Rectangle(
                    (s - 0.2, base_y - r*tile_height),
                    0.4, tile_height*0.9,
                    color=color, clip_on=False, zorder=500
                )
                ax.add_patch(rect)

                ax.text(
                    s, base_y - r*tile_height + tile_height*0.45,
                    f"{int(val*100)}%",
                    ha="center", va="center",
                    fontsize=10, color=text_color,
                    zorder=600
                )

        ax.set_title(f'RAPM Over Seasons for {player_name}')
        ax.set_xlabel('Season')
        ax.set_ylabel('Percentile Rank')
        ax.set_xticks(seasons)
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
        ax.legend(loc='upper left')
        plt.ylim(-0.05, 1.05)
        plt.tight_layout()
        plt.show()

    return plot_player_rapm, sns


@app.cell
def _(plot_player_rapm):
    plot_player_rapm("Lane Hutson", cmap="coolwarm_r")
    return


@app.cell
def _(combined_rapm, sns):
    sns.histplot(combined_rapm, x="pk_xga")
    return


if __name__ == "__main__":
    app.run()
