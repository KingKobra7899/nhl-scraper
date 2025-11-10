from functools import lru_cache
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import requests
from typing import List
from tqdm import tqdm


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

    # Define the columns we want for plays
    plays_columns = [
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
        "details.hittingPlayerId",
        "details.hitteePlayerId",
        "details.duration",
        "details.committedByPlayerId",
        "details.drawnByPlayerId",
        "details.blockingPlayerId",
        "details.shootingPlayerId",
    ]

    # Add missing columns with NaN
    for col in plays_columns:
        if col not in plays.columns:
            plays[col] = np.nan

    # Select and reorder columns
    plays = plays[plays_columns]

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
            "details.shootingPlayerId": "shootingPlayerId",
            "details.blockingPlayerId": "blockingPlayerId",
        }
    )

    # Define the columns we want for shots
    shots_columns = [
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

    # Add missing columns with NaN
    for col in shots_columns:
        if col not in shots.columns:
            shots[col] = np.nan

    # Select and reorder columns
    shots = shots[shots_columns]

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


@lru_cache(32)
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
        return ("", "")


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


def scrapeGamesShots(games: List[str]) -> pd.DataFrame | pd.Series:
    shots = pd.DataFrame()
    for game in tqdm(games):
        try:
            shots = pd.concat([shots, getPbpData(game)["shots"]])
        except Exception as e:
            print(f"{game}, {e}")

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


# x in y
def getTeamSeasonGames(team: int, season: int) -> List[str]:
    url = f"https://api-web.nhle.com/v1/club-schedule-season/{getTeamName(team)[1]}/{season}"
    response = requests.get(url)
    if response.status_code != 200:
        print(url)
        raise Exception(f"Failed to fetch game metadata: {response.status_code}")

    games_data = response.json()["games"]
    games_df = pd.DataFrame(games_data)
    games_df = games_df[games_df["gameType"] == 2]
    games_df = games_df[games_df["gameState"] == "OFF"]
    return games_df["id"].to_list()


def getPlayerGames(playerId: int) -> list[str]:
    url = f"https://api-web.nhle.com/v1/player/{playerId}/game-log/20252026/2"
    response = requests.get(url)
    if response.status_code != 200:
        print(url)
        raise Exception(f"Failed to fetch game metadata: {response.status_code}")

    szn = response.json()["playerStatSeasons"]
    games = []
    for season in szn:
        url = f"https://api-web.nhle.com/v1/player/{playerId}/game-log/{season['season']}/2"
        response = requests.get(url)
        if response.status_code != 200:
            print(url)
            raise Exception(f"Failed to fetch game metadata: {response.status_code}")
        log = response.json()["gameLog"]
        for game in log:
            games.append(
                {
                    "season": season["season"],
                    "gameId": game["gameId"],
                    "date": game["gameDate"],
                }
            )
    return games


def getPlayerPuckPossession(
    game_data: pd.DataFrame, shots: pd.DataFrame
) -> pd.DataFrame:
    # mapping: eventType -> columns holding playerId(s)
    event_player = {
        "hit": ["hiteePlayerId"],
        "giveaway": ["playerId"],
        "takeaway": ["playerId"],
        "blocked-shot": ["shootingPlayerId"],
        "penalty": ["drawnByPlayerId"],
        "shot-on-goal": ["shootingPlayerId"],
        "missed-shot": ["shootingPlayerId"],
        "goal": ["scoringPlayerId", "a1PlayerId", "a2PlayerId"],
    }

    df = pd.concat([game_data, shots], ignore_index=True)

    df = df[df["typeDescKey"].isin(event_player.keys())].copy()

    records = []

    for _, row in df.iterrows():
        etype = row["typeDescKey"]
        zone = row.get("zoneCode", None)

        for col in event_player[etype]:
            if col in row and pd.notna(row[col]):
                pid = row[col]
                records.append({"playerId": pid, "eventType": etype, "zoneCode": zone})

    return pd.DataFrame(records, columns=["playerId", "eventType", "zoneCode"])
