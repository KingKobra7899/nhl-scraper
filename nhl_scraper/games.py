import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter
import requests
from typing import List


def str_to_float(time_str: str) -> float:
    parts = time_str.split(":")
    if len(parts) == 2:
        minutes = int(parts[0])
        seconds = float(parts[1])
        return minutes + (seconds / 60)
    else:
        print("ERROR")
        return -1


def getPbpData(gameid: str):
    """
    - Returns all play-by-play events for a game (that are tracked by NHL) \\
    - Shot-attempts are in a seperate df\\
    - Also returns the venue and home/away teams
    \\
    Parameters:\\
        gameid: the nhl's id for a game

    """

    url = f"https://api-web.nhle.com/v1/gamecenter/{gameid}/play-by-play"
    response = requests.get(url)
    data = response.json()

    venue = data["venue"]["default"]
    home = data["homeTeam"]["id"]
    away = data["awayTeam"]["id"]

    plays = pd.json_normalize(data["plays"])
    plays["timeInPeriod"] = (
        plays["timeInPeriod"].apply(str_to_float)
        + (plays["periodDescriptor.number"] - 1) * 20
    )
    plays["timeRemaining"] = plays["timeRemaining"].apply(str_to_float)
    shot_mask = plays["typeDescKey"].isin(["goal", "shot-on-goal", "missed-shot"])
    shots = plays[shot_mask]
    plays = plays[~shot_mask]

    plays = plays[
        [
            "timeInPeriod",
            "typeDescKey",
            "timeRemaining",
            "situationCode",
            "homeTeamDefendingSide",
            "periodDescriptor.number",
            "details.eventOwnerTeamId",
            "details.playerId",
            "details.losingPlayerId",
            "details.winningPlayerId",
            "details.xCoord",
            "details.yCoord",
            "details.zoneCode",
            "details.playerId",
            "details.hittingPlayerId",
            "details.hitteePlayerId",
            "details.duration",
            "details.committedByPlayerId",
            "details.drawnByPlayerId",
            "details.blockingPlayerId",
        ]
    ]

    plays = plays.rename(
        columns={
            "periodDescriptor.number": "period",
            "details.eventOwnerTeamId": "teamId",
            "details.playerId": "playerId",
            "details.losingPlayerId": "losingPlayerId",
            "details.winningPlayerId": "winningPlayerId",
            "details.xCoord": "xCoord",
            "details.yCoord": "yCoord",
            "details.zoneCode": "zoneCode",
            "details.hittingPlayerId": "hittingPlayerId",
            "details.hitteePlayerId": "hitteePlayerId",
            "details.duration": "duration",
            "details.committedByPlayerId": "committedByPlayerId",
            "details.drawnByPlayerId": "drawnByPlayerId",
            "details.blockingPlayerId": "blockingPlayerId",
        }
    )
    shots = shots[
        [
            "timeInPeriod",
            "timeRemaining",
            "situationCode",
            "typeDescKey",
            "homeTeamDefendingSide",
            "periodDescriptor.number",
            "periodDescriptor.periodType",
            "details.shootingPlayerId",
            "details.awaySOG",
            "details.homeSOG",
            "details.awayScore",
            "details.homeScore",
            "details.xCoord",
            "details.yCoord",
            "details.zoneCode",
            "details.eventOwnerTeamId",
            "details.goalieInNetId",
            "details.shotType",
            "details.scoringPlayerId",
            "details.assist1PlayerId",
            "details.assist2PlayerId",
        ]
    ]
    shots = shots.rename(
        columns={
            "details.shootingPlayerId": "shootingPlayerId",
            "details.scoringPlayerId": "scoringPlayerId",
            "details.awaySOG": "awaySOG",
            "details.homeSOG": "homeSOG",
            "details.awayScore": "awayScore",
            "details.homeScore": "homeScore",
            "details.xCoord": "xCoord",
            "details.yCoord": "yCoord",
            "details.zoneCode": "zoneCode",
            "details.eventOwnerTeamId": "teamId",
            "details.goalieInNetId": "goalieId",
            "details.shotType": "shotType",
            "details.assist1PlayerId": "a1PlayerId",
            "details.assist2PlayerId": "a2PlayerId",
            "periodDescriptor.number": "period",
            "periodDescriptor.periodType": "periodType",
        }
    )

    shots["isHome"] = shots["teamId"] == home
    shooting_right = np.where(
        shots["isHome"],
        shots["homeTeamDefendingSide"]
        == "left",  # home team shoots right if defending left
        shots["homeTeamDefendingSide"]
        == "right",  # away team shoots right if home defends right
    )
    shots["shootingRight"] = shooting_right
    shots["xCoord"] = np.where(shooting_right, shots["xCoord"], -shots["xCoord"])
    shots["yCoord"] = np.where(shooting_right, shots["yCoord"], -shots["yCoord"])

    shots["dist"] = np.sqrt((shots["xCoord"] - 89) ** 2 + shots["yCoord"] ** 2)
    shots["angle"] = np.arctan2(shots["xCoord"] - 89, shots["yCoord"])

    return {
        "venue": venue,
        "homeTeamId": home,
        "awayTeamId": away,
        "plays": plays,
        "shots": shots[shots["periodType"] != "SO"],
    }


def getGameString(gameid: str):
    url = f"https://api-web.nhle.com/v1/gamecenter/{gameid}/landing"
    response = requests.get(url)
    data = response.json()

    home_team = f"{data['homeTeam']['abbrev']}"
    away_team = f"{data['awayTeam']['abbrev']}"
    return f"{away_team} @ {home_team}"


def getTeamName(teamid: int):
    """
    recieves full nhl team name from that teams nhl api id
    """
    url = f"https://api.nhle.com/stats/rest/en/team/id/{int(teamid)}"
    response = requests.get(url)
    data = response.json()
    try:
        return (
            data["data"][0]["fullName"],
            data["data"][0]["triCode"],
        )
    except:
        print(url)
        return None


def draw_rink_features(
    ax,
    xmin=-100,
    xmax=100,
    ymin=-42.5,
    ymax=42.5,
    zone="offensive",
    color="white",
    alpha=0.3,
    linewidth=1.5,
):
    """
    Draw NHL rink features for offensive or defensive zone.
    Coordinate system: center faceoff dot at (0, 0), x from -100 to 100, y from -42.5 to 42.5.

    Parameters:
    - ax: matplotlib axis object
    - xmin, xmax, ymin, ymax: rink dimensions
    - zone: 'offensive' (right/positive x) or 'defensive' (left/negative x)
    - color: color for all rink features
    - alpha: transparency of the lines
    - linewidth: width of the lines
    """

    if zone == "offensive":
        # Blue line: 25 ft from center (positive x)
        blue_line_x = 25
        ax.axvline(
            blue_line_x,
            color=color,
            alpha=alpha,
            linewidth=linewidth * 1.5,
            linestyle="-",
        )

        # Goal line: 89 ft from center (11 ft from end boards at x=100)
        goal_line_x = 89
        ax.axvline(
            goal_line_x, color=color, alpha=alpha, linewidth=linewidth, linestyle="-"
        )

        # Goal crease (semicircle, radius 6 ft, centered at goal line)
        theta = np.linspace(-np.pi / 2, np.pi / 2, 100)
        crease_radius = 6
        crease_x = goal_line_x + crease_radius * np.cos(theta)
        crease_y = crease_radius * np.sin(theta)
        ax.plot(crease_x, crease_y, color=color, alpha=alpha, linewidth=linewidth)
        ax.plot(
            [goal_line_x, goal_line_x],
            [-crease_radius, crease_radius],
            color=color,
            alpha=alpha,
            linewidth=linewidth,
        )

        # Net (4 ft deep from goal line, 6 ft wide)
        net_depth = 4
        net_width = 6
        net = plt.Rectangle(
            (goal_line_x, -net_width / 2),
            net_depth,
            net_width,
            color=color,
            fill=False,
            alpha=alpha,
            linewidth=linewidth * 2,
        )
        ax.add_patch(net)

        # Rounded corners
        corner_radius = 28
        corner_x = xmax - corner_radius
        theta_top = np.linspace(0, np.pi / 2, 50)
        theta_bottom = np.linspace(-np.pi / 2, 0, 50)

        # Top right corner arc
        arc_x_top = corner_x + corner_radius * np.cos(theta_top)
        arc_y_top = (ymax - corner_radius) + corner_radius * np.sin(theta_top)
        ax.plot(arc_x_top, arc_y_top, color=color, alpha=alpha, linewidth=linewidth)

        # Bottom right corner arc
        arc_x_bottom = corner_x + corner_radius * np.cos(theta_bottom)
        arc_y_bottom = (ymin + corner_radius) + corner_radius * np.sin(theta_bottom)
        ax.plot(
            arc_x_bottom, arc_y_bottom, color=color, alpha=alpha, linewidth=linewidth
        )

        # Faceoff circles: 69 ft from center (20 ft from goal line), ±22 ft from center line
        faceoff_x = 69

    else:  # defensive zone
        # Blue line: -25 ft from center (negative x)
        blue_line_x = -25
        ax.axvline(
            blue_line_x,
            color=color,
            alpha=alpha,
            linewidth=linewidth * 1.5,
            linestyle="-",
        )

        # Goal line: -89 ft from center (11 ft from end boards at x=-100)
        goal_line_x = -89
        ax.axvline(
            goal_line_x, color=color, alpha=alpha, linewidth=linewidth, linestyle="-"
        )

        # Goal crease (semicircle, radius 6 ft, centered at goal line, facing left)
        theta = np.linspace(np.pi / 2, 3 * np.pi / 2, 100)
        crease_radius = 6
        crease_x = goal_line_x + crease_radius * np.cos(theta)
        crease_y = crease_radius * np.sin(theta)
        ax.plot(crease_x, crease_y, color=color, alpha=alpha, linewidth=linewidth)
        ax.plot(
            [goal_line_x, goal_line_x],
            [-crease_radius, crease_radius],
            color=color,
            alpha=alpha,
            linewidth=linewidth,
        )

        # Net (4 ft deep from goal line toward boards, 6 ft wide)
        net_depth = 4
        net_width = 6
        net = plt.Rectangle(
            (goal_line_x - net_depth, -net_width / 2),
            net_depth,
            net_width,
            color=color,
            fill=False,
            alpha=alpha,
            linewidth=linewidth * 2,
        )
        ax.add_patch(net)

        # Rounded corners
        corner_radius = 28
        corner_x = xmin + corner_radius
        theta_top = np.linspace(np.pi / 2, np.pi, 50)
        theta_bottom = np.linspace(np.pi, 3 * np.pi / 2, 50)

        # Top left corner arc
        arc_x_top = corner_x + corner_radius * np.cos(theta_top)
        arc_y_top = (ymax - corner_radius) + corner_radius * np.sin(theta_top)
        ax.plot(arc_x_top, arc_y_top, color=color, alpha=alpha, linewidth=linewidth)

        # Bottom left corner arc
        arc_x_bottom = corner_x + corner_radius * np.cos(theta_bottom)
        arc_y_bottom = (ymin + corner_radius) + corner_radius * np.sin(theta_bottom)
        ax.plot(
            arc_x_bottom, arc_y_bottom, color=color, alpha=alpha, linewidth=linewidth
        )

        # Faceoff circles: -69 ft from center (20 ft from goal line), ±22 ft from center line
        faceoff_x = -69

    # Draw the two faceoff circles with dots (radius 15 ft, at y = ±22 ft)
    faceoff_radius = 15
    faceoff_positions = [
        (faceoff_x, 22),  # Top faceoff circle
        (faceoff_x, -22),  # Bottom faceoff circle
    ]

    for x, y in faceoff_positions:
        circle = plt.Circle(
            (x, y),
            faceoff_radius,
            color=color,
            fill=False,
            alpha=alpha,
            linewidth=linewidth,
        )
        ax.add_patch(circle)
        # Faceoff dot (1 ft radius)
        dot = plt.Circle(
            (x, y), 1, color=color, fill=True, alpha=alpha * 1.5, linewidth=0
        )
        ax.add_patch(dot)


def plot_game_shot_density(game_id, mode="both", sigma=10):
    df: DataFrame = getPbpData(game_id)["shots"]
    teams = df["teamId"].unique()
    team1_df = df[df["teamId"] == teams[0]]
    team2_df = df[df["teamId"] == teams[1]]
    # Get team names
    team1_name, team1_tricode = getTeamName(teams[0])
    team2_name, team2_tricode = getTeamName(teams[1])
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
        draw_rink_features(
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
        draw_rink_features(
            ax2, xmin, xmax, ymin, ymax, color="black", alpha=0.5, linewidth=1.5
        )

    elif mode == "diff":
        fig, ax = plt.subplots(1, 1, figsize=(10, 8.5))
        fig.set_dpi(200)
        im = ax.imshow(
            diff,
            extent=[xmin, xmax, ymin, ymax],
            origin="lower",
            cmap="RdBu",
            vmin=-diff.max(),
            vmax=diff.max(),
        )
        ax.contourf(
            diff,
            levels=np.linspace(diff.min(), diff.max(), 30),
            linewidths=0.5,
            extent=[xmin, xmax, ymin, ymax],
            alpha=0.6,
            cmap="RdBu",
            vmin=-diff.max(),
            vmax=diff.max(),
        )

        draw_rink_features(
            ax, xmin, xmax, ymin, ymax, color="black", alpha=0.5, linewidth=1.5
        )

        ax.set_title(getGameString(game_id), fontsize=14)
        ax.set_xlabel("")
        ax.set_ylabel("")
        cbar = plt.colorbar(im, ax=ax, orientation="vertical", shrink=0.75)
        cbar.set_label(
            f"{team1_tricode} ← → {team2_tricode}", rotation=270, labelpad=20
        )

    plt.tight_layout()
    plt.show()


def scrapeGamesShots(games: List[str]) -> pd.DataFrame | pd.Series:
    shots = pd.DataFrame()
    for game in games:
        shots = pd.concat([shots, getPbpData(game)["shots"]])
    return shots


def getSeasonPlayedGames(season1: int, season2: int) -> List[str]:
    url = "https://api.nhle.com/stats/rest/en/game"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch game metadata: {response.status_code}")

    games_data = response.json().get("data", [])
    games_df = pd.DataFrame(games_data)

    print("filtering seasons")
    games = games_df[
        (games_df["season"] >= season1)
        & (games_df["season"] <= season2)
        & (games_df["gameType"] == 2)
        & (games_df["gameStateId"] == 7)
    ]["id"].tolist()

    return games


def getTeamSeasonGames(team: int, season: int) -> List[str]:
    url = f"https://api-web.nhle.com/v1/club-schedule-season/{getTeamName(team)[1]}/{season}"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch game metadata: {response.status_code}")

    games_data = response.json()["games"]
    games_df = pd.DataFrame(games_data)
    games_df = games_df[games_df["gameType"] == 2]
    games_df = games_df[games_df["gameState"] == "OFF"]
    return games_df["id"].to_list()
