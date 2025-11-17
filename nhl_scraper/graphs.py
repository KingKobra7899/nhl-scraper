import nhl_scraper.games as ga
import numpy as np
from scipy.ndimage import gaussian_filter
import pandas as pd
import matplotlib.pyplot as plt


def plot_game_shot_density(game_id, mode="both", sigma=10):
    df: pd.DataFrame = ga.getPbpData(game_id)["shots"]
    teams = df["teamId"].unique()
    team1_df = df[df["teamId"] == teams[0]]
    team2_df = df[df["teamId"] == teams[1]]
    # Get team names
    team1_name, team1_tricode = ga.getTeamName(teams[0])
    team2_name, team2_tricode = ga.getTeamName(teams[1])
    xmin, xmax = 0, 100
    ymin, ymax = -44.5, 44.5
    H1, xedges, yedges = np.histogram2d(
        team1_df["xCoord"],
        team1_df["yCoord"],
        bins=[100 * 3, 89 * 3],
        range=[[xmin, xmax], [ymin, ymax]],
    )
    H2, _, _ = np.histogram2d(
        team2_df["xCoord"],
        team2_df["yCoord"],
        bins=[100 * 3, 89 * 3],
        range=[[xmin, xmax], [ymin, ymax]],
    )
    density1 = gaussian_filter(H1.T, sigma=sigma * 3)
    density2 = gaussian_filter(H2.T, sigma=sigma * 3)
    diff = density1 - density2
    vmax = max(density1.max(), density2.max())
    vmin = 0
    if mode == "both":
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.set_dpi(200)
        ax1.imshow(
            density1,
            extent=[xmin, xmax, ymin, ymax],
            origin="lower",
            cmap="Blues",
            vmin=0,
            vmax=vmax,
        )
        ax1.contourf(
            density1,
            levels=np.linspace(0, vmax, 30),
            extent=[xmin, xmax, ymin, ymax],
            alpha=0.6,
            cmap="Blues",
            vmin=0,
            vmax=vmax,
        )
        ax1.set_title(f"{team1_tricode} Shot Density", fontsize=14)
        ax1.set_xlabel("X Coord")
        ax1.set_ylabel("Y Coord")
        ga.draw_rink_features(
            ax1, xmin, xmax, ymin, ymax, color="black", alpha=0.5, linewidth=1.5
        )
        ax2.imshow(
            density2,
            extent=[xmin, xmax, ymin, ymax],
            origin="lower",
            cmap="Reds",
            vmin=0,
            vmax=vmax,
        )
        ax2.contourf(
            density2,
            levels=np.linspace(0, vmax, 30),
            extent=[xmin, xmax, ymin, ymax],
            alpha=0.6,
            cmap="Reds",
            vmin=0,
            vmax=vmax,
        )
        ax2.set_title(f"{team2_tricode} Shot Density", fontsize=14)
        ax2.set_xlabel("X Coord")
        ax2.set_ylabel("Y Coord")
        ga.draw_rink_features(
            ax2, xmin, xmax, ymin, ymax, color="black", alpha=0.5, linewidth=1.5
        )

    elif mode == "diff":
        fig, ax = plt.subplots(1, 1, figsize=(10, 8.5))
        fig.set_dpi(200)
        ax.set_xlim(0, 100)
        ax.set_ylim(-42.5, 42.5)
        im = ax.imshow(
            diff,
            extent=[xmin, xmax, ymin, ymax],
            origin="lower",
            cmap="bwr",
            vmin=-diff.max(),
            vmax=diff.max(),
        )

        total = diff.sum()

        # Normalize teamId to same colormap
        norm = plt.Normalize(df["teamId"].min(), df["teamId"].max())

        ax.scatter(
            x=df["xCoord"],
            y=df["yCoord"],
            c=df["teamId"],
            cmap="bwr",
            norm=norm,
            s=28,  # slightly larger for clarity
            edgecolors="black",  # sharpens appearance
            linewidths=0.4,
            alpha=0.9,
        )

        ga.draw_rink_features(
            ax, xmin, xmax, ymin, ymax, color="black", alpha=0.5, linewidth=1.5
        )

        ax.set_title(
            f"{ga.getGameString(game_id)}, Fenwick Differential: {total:-.0f}",
            fontsize=14,
        )
        ax.set_xlabel("")
        ax.set_ylabel("")

        cbar = plt.colorbar(im, ax=ax, orientation="vertical", shrink=0.75)
        cbar.ax.tick_params(labelsize=0)
        plt.text(x=115, y=38, s=team1_tricode)
        plt.text(x=115, y=-39.5, s=team2_tricode)

    plt.tight_layout()
    plt.show()


def plot_team_shot_density(df, id, gp, mode="both", sigma=10):
    team1_df = df[df["teamId"] == id]
    team2_df = df[df["teamId"] != id]
    # Get team names
    team1_name, team1_tricode = ga.getTeamName(id)

    xmin, xmax = 0, 100
    ymin, ymax = -44.5, 44.5
    H1, xedges, yedges = np.histogram2d(
        team1_df["xCoord"],
        team1_df["yCoord"],
        bins=[100 * 3, 89 * 3],
        range=[[xmin, xmax], [ymin, ymax]],
    )
    H2, _, _ = np.histogram2d(
        team2_df["xCoord"],
        team2_df["yCoord"],
        bins=[100 * 3, 89 * 3],
        range=[[xmin, xmax], [ymin, ymax]],
    )
    density1 = gaussian_filter(H1.T, sigma=sigma * 3)
    density2 = gaussian_filter(H2.T, sigma=sigma * 3)
    diff = density1 - density2
    vmax = max(density1.max(), density2.max())
    vmin = 0
    if mode == "both":
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.set_dpi(200)
        ax1.imshow(
            density1,
            extent=[xmin, xmax, ymin, ymax],
            origin="lower",
            cmap="Blues",
            vmin=0,
            vmax=vmax,
        )
        ax1.contourf(
            density1,
            levels=np.linspace(0, vmax, 30),
            extent=[xmin, xmax, ymin, ymax],
            alpha=0.6,
            cmap="Blues",
            vmin=0,
            vmax=vmax,
        )
        ax1.set_title("Shot Density For", fontsize=14)
        ax1.set_xlabel("")
        ax1.set_ylabel("")
        ga.draw_rink_features(
            ax1, xmin, xmax, ymin, ymax, color="black", alpha=0.5, linewidth=1.5
        )
        ax2.imshow(
            density2,
            extent=[xmin, xmax, ymin, ymax],
            origin="lower",
            cmap="Reds",
            vmin=0,
            vmax=vmax,
        )
        ax2.contourf(
            density2,
            levels=np.linspace(0, vmax, 30),
            extent=[xmin, xmax, ymin, ymax],
            alpha=0.6,
            cmap="Reds",
            vmin=0,
            vmax=vmax,
        )
        ax2.set_title(f"Shot Density Against", fontsize=14)
        ax2.set_xlabel("")
        ax2.set_ylabel("")
        ga.draw_rink_features(
            ax2, xmin, xmax, ymin, ymax, color="black", alpha=0.5, linewidth=1.5
        )

        return fig

    elif mode == "diff":
        fig, ax = plt.subplots(1, 1, figsize=(10, 8.5))
        fig.set_dpi(200)

        diff /= gp
        im = ax.imshow(
            diff,
            extent=[xmin, xmax, ymin, ymax],
            origin="lower",
            cmap="bwr_r",
            vmin=-diff.max(),
            vmax=diff.max(),
        )
        # ax.contourf(
        #     diff,
        #     levels=np.linspace(diff.min(), diff.max(), 20),
        #     extent=[xmin, xmax, ymin, ymax],
        #     alpha=1.0,
        #     cmap="bwr_r",
        #     vmin=-diff.max(),
        #     vmax=diff.max(),
        # )

        ga.draw_rink_features(
            ax, xmin, xmax, ymin, ymax, color="black", alpha=0.5, linewidth=1.5
        )

        ax.set_title(
            f"{team1_tricode} Fenwick Differential  (Per-Game Differential: {diff.sum():+.2f})",
            fontsize=14,
        )
        ax.set_xlabel("")
        ax.set_ylabel("")
        cbar = plt.colorbar(im, ax=ax, orientation="vertical", shrink=0.75)
        cbar.set_label(f"Good ← → Bad", rotation=270, labelpad=20)

        return fig

    plt.tight_layout()
    plt.show()
