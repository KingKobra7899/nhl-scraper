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


def getPbpData(gameid: str | int):
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
        shots["homeTeamDefendingSide"] == "left",
        shots["homeTeamDefendingSide"] == "right",
    )
    shots["shootingRight"] = shooting_right
    shots["xCoord"] = np.where(shooting_right, shots["xCoord"], -shots["xCoord"])
    shots["yCoord"] = np.where(shooting_right, shots["yCoord"], -shots["yCoord"])

    shots["dist"] = np.sqrt((shots["xCoord"] - 89) ** 2 + shots["yCoord"] ** 2)
    shots["angle"] = np.arctan2(shots["xCoord"] - 89, shots["yCoord"])

    shots["venue"] = venue
    plays["venue"] = venue
    return {
        "venue": venue,
        "homeTeamId": home,
        "awayTeamId": away,
        "plays": plays,
        "shots": shots[shots["periodType"] != "SO"],
    }


def getGameString(gameid: str | int):
    url = f"https://api-web.nhle.com/v1/gamecenter/{gameid}/landing"
    response = requests.get(url)
    data = response.json()

    home_team = f"{data['homeTeam']['abbrev']}"
    away_team = f"{data['awayTeam']['abbrev']}"
    return f"{away_team} @ {home_team}"


@lru_cache(32)
def getTeamName(teamid: str | int):
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


def scrapeGamesPbp(games: List[str]) -> dict[str, pd.DataFrame | pd.Series]:
    shots = pd.DataFrame()
    plays = pd.DataFrame()
    for game in tqdm(games):
        try:
            shots = pd.concat([shots, getPbpData(game)["shots"]])
            plays = pd.concat([plays, getPbpData(game)["plays"]])
        except Exception as e:
            print(f"{game}, {e}")

    return {"shots": shots, "plays": plays}


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
                records.append(
                    {
                        "playerId": pid,
                        "eventType": etype,
                        "zoneCode": zone,
                    }
                )

    return pd.DataFrame(records, columns=["playerId", "eventType", "zoneCode"])


def generateGameShifts(gameId):
    url = f"https://api.nhle.com/stats/rest/en/shiftcharts?cayenneExp=gameId={gameId}"
    r = requests.get(url)
    df = pd.DataFrame(r.json()["data"])
    df = df[df["detailCode"] == 0]
    # Convert times to float minutes
    df["startTime"] = df.startTime.apply(str_to_float) + (df.period - 1) * 20
    df["endTime"] = df.endTime.apply(str_to_float) + (df.period - 1) * 20

    # Collect all unique time points
    time_points = pd.unique(pd.concat([df.startTime, df.endTime]))
    time_points.sort()

    # Create stints: each interval between consecutive time points
    stints = pd.DataFrame({"start": time_points[:-1], "end": time_points[1:]})
    stints["duration"] = stints["end"] - stints["start"]
    stints["stintId"] = stints.index

    # For each player, record which stints they are on ice
    df["stintIds"] = df.apply(
        lambda row: stints[
            (stints["start"] < row["endTime"]) & (stints["end"] > row["startTime"])
        ].stintId.tolist(),
        axis=1,
    )

    return df, stints


def getSituation(situationCode: int | str, is_home: bool) -> str:
    h = str(situationCode)[1]
    a = str(situationCode)[2]
    return f"{a}v{h}" if is_home else f"{h}v{a}"


def get_sit(code, is_home):
    """Get situation short code (ev/pp/pk)."""
    s = getSituation(code, is_home)
    left, right = s.split("v")
    if left == right:
        return "ev"
    elif int(left) > int(right):
        return "pp"
    else:
        return "pk"


def getBoxScore(gameId: str | int) -> dict[str, pd.DataFrame]:
    pbp = getPbpData(gameId)
    shots = pbp["shots"].copy()
    plays = pbp["plays"].copy()
    player_shifts, stints = generateGameShifts(gameId)

    home_team_id = int(pbp["homeTeamId"])
    away_team_id = int(pbp["awayTeamId"])

    # Assign each shot to a stint
    shots["stint_idx"] = shots["timeInPeriod"].apply(
        lambda t: stints.index[(stints["start"] < t) & (t <= stints["end"])][0]
        if len(stints.index[(stints["start"] < t) & (t <= stints["end"])]) > 0
        else -1
    )

    plays["stint_idx"] = plays["timeInPeriod"].apply(
        lambda t: stints.index[(stints["start"] < t) & (t <= stints["end"])][0]
        if len(stints.index[(stints["start"] < t) & (t <= stints["end"])]) > 0
        else -1
    )

    # Prepare boxscore skeleton
    all_players = player_shifts["playerId"].unique()
    situations = ["ev", "pp", "pk"]
    boxscore = pd.DataFrame(
        [(pid, sit) for pid in all_players for sit in situations],
        columns=["playerId", "situation"],
    )
    boxscore["name"] = boxscore["playerId"].map(
        lambda pid: (
            f"{player_shifts.loc[player_shifts['playerId'] == pid, 'firstName'].iloc[0]} "
            f"{player_shifts.loc[player_shifts['playerId'] == pid, 'lastName'].iloc[0]}"
        )
        if not player_shifts.loc[player_shifts["playerId"] == pid].empty
        else str(pid)
    )

    stat_cols = [
        "goals",
        "a1",
        "a2",
        "corsi",
        "fenwick",
        "sog",
        "giveaways",
        "takeaways",
        "hits_for",
        "hits_taken",
        "blocks",
        "faceoffs_won",
        "faceoffs_lost",
        "penalties_taken",
        "penalties_drawn",
        "total_toi",
        "ff",
        "fa",
        "sogf",
        "soga",
        "gf",
        "ga",
    ]
    for col in stat_cols:
        boxscore[col] = 0.0

    # Explode player shifts by stintIds
    player_stints = player_shifts[["playerId", "teamId", "stintIds"]].explode(
        "stintIds"
    )
    player_stints = player_stints.merge(
        stints, left_on="stintIds", right_index=True, how="left"
    )
    player_stints["toi"] = player_stints["end"] - player_stints["start"]

    # Map situation from shots to stints
    stint_sit = shots.groupby("stint_idx")["situationCode"].first()
    player_stints["situationCode"] = player_stints["stintIds"].map(stint_sit)
    player_stints["situation"] = player_stints.apply(
        lambda r: get_sit(r["situationCode"], r["teamId"] == home_team_id)
        if pd.notna(r["situationCode"])
        else "ev",
        axis=1,
    )

    # Compute stint-level metrics
    stint_metrics = {}
    for idx, stint in stints.iterrows():
        stint_shots = shots[shots["stint_idx"] == idx]
        if stint_shots.empty:
            stint_metrics[idx] = {
                "ff_home": 0,
                "fa_home": 0,
                "sogf_home": 0,
                "soga_home": 0,
                "gf_home": 0,
                "ga_home": 0,
            }
            continue

        cf_home = stint_shots[stint_shots["isHome"]].shape[0]
        ca_home = stint_shots[~stint_shots["isHome"]].shape[0]
        sogf_home = stint_shots[
            (stint_shots["isHome"])
            & (stint_shots["typeDescKey"].isin(["shot-on-goal", "goal"]))
        ].shape[0]
        soga_home = stint_shots[
            (~stint_shots["isHome"])
            & (stint_shots["typeDescKey"].isin(["shot-on-goal", "goal"]))
        ].shape[0]
        gf_home = stint_shots[
            (stint_shots["isHome"]) & (stint_shots["typeDescKey"] == "goal")
        ].shape[0]
        ga_home = stint_shots[
            (~stint_shots["isHome"]) & (stint_shots["typeDescKey"] == "goal")
        ].shape[0]

        stint_metrics[idx] = {
            "ff_home": cf_home,
            "fa_home": ca_home,
            "sogf_home": sogf_home,
            "soga_home": soga_home,
            "gf_home": gf_home,
            "ga_home": ga_home,
        }

    stint_metrics = pd.DataFrame.from_dict(stint_metrics, orient="index")
    player_stints = player_stints.merge(
        stint_metrics, left_on="stintIds", right_index=True, how="left"
    ).fillna(0)

    # Assign correct metrics based on player team
    def assign_metrics(row):
        if row["teamId"] == home_team_id:
            row["ff"], row["fa"] = row["ff_home"], row["fa_home"]
            row["sogf"], row["soga"] = row["sogf_home"], row["soga_home"]
            row["gf"], row["ga"] = row["gf_home"], row["ga_home"]
        else:
            row["ff"], row["fa"] = row["fa_home"], row["ff_home"]
            row["sogf"], row["soga"] = row["soga_home"], row["sogf_home"]
            row["gf"], row["ga"] = row["ga_home"], row["gf_home"]
        return row

    player_stints = player_stints.apply(assign_metrics, axis=1)
    player_stints["total_toi"] = player_stints["toi"]

    # Aggregate per player + situation
    player_metrics = (
        player_stints.groupby(["playerId", "situation"])[
            ["ff", "fa", "sogf", "soga", "gf", "ga", "total_toi"]
        ]
        .sum()
        .reset_index()
    )

    # Merge to boxscore
    boxscore = boxscore.merge(
        player_metrics, on=["playerId", "situation"], how="left", suffixes=("", "_agg")
    )
    for col in ["ff", "fa", "sogf", "soga", "gf", "ga", "total_toi"]:
        boxscore[col] = boxscore[f"{col}_agg"].fillna(0)
        boxscore.drop(columns=[f"{col}_agg"], inplace=True)

    shots["situation"] = shots.apply(
        lambda r: get_sit(r["situationCode"], r["teamId"] == home_team_id)
        if pd.notna(r["situationCode"]) and pd.notna(r["teamId"])
        else "ev",
        axis=1,
    )

    plays["situation"] = plays.apply(
        lambda r: get_sit(r["situationCode"], r["teamId"] == home_team_id)
        if pd.notna(r["situationCode"]) and pd.notna(r["teamId"])
        else "ev",
        axis=1,
    )

    # ===== SHOTS-BASED STATS =====

    # Goals (use scoringPlayerId)
    goals = (
        shots[shots["typeDescKey"] == "goal"]
        .groupby(["scoringPlayerId", "situation"])
        .size()
        .reset_index(name="goals")
    )
    goals.rename(columns={"scoringPlayerId": "playerId"}, inplace=True)

    # Assist1
    assist1 = (
        shots[shots["typeDescKey"] == "goal"]
        .dropna(subset=["a1PlayerId"])
        .groupby(["a1PlayerId", "situation"])
        .size()
        .reset_index(name="a1")
    )
    assist1.rename(columns={"a1PlayerId": "playerId"}, inplace=True)

    # Assist2
    assist2 = (
        shots[shots["typeDescKey"] == "goal"]
        .dropna(subset=["a2PlayerId"])
        .groupby(["a2PlayerId", "situation"])
        .size()
        .reset_index(name="a2")
    )
    assist2.rename(columns={"a2PlayerId": "playerId"}, inplace=True)

    # Fenwick (SOG + missed + goals, excluding blocked)
    fenwick_non_goal = (
        shots[shots["typeDescKey"].isin(["shot-on-goal", "missed-shot"])]
        .groupby(["shootingPlayerId", "situation"])
        .size()
        .reset_index(name="fenwick")
    )
    fenwick_non_goal.rename(columns={"shootingPlayerId": "playerId"}, inplace=True)
    fenwick_goals = (
        shots[shots["typeDescKey"] == "goal"]
        .groupby(["scoringPlayerId", "situation"])
        .size()
        .reset_index(name="fenwick")
    )
    fenwick_goals.rename(columns={"scoringPlayerId": "playerId"}, inplace=True)
    fenwick = (
        pd.concat([fenwick_non_goal, fenwick_goals])
        .groupby(["playerId", "situation"])["fenwick"]
        .sum()
        .reset_index()
    )

    # Shots on goal (SOG + goals, excluding missed and blocked)
    sog_shots = (
        shots[shots["typeDescKey"] == "shot-on-goal"]
        .groupby(["shootingPlayerId", "situation"])
        .size()
        .reset_index(name="sog")
    )
    sog_shots.rename(columns={"shootingPlayerId": "playerId"}, inplace=True)
    sog_goals = (
        shots[shots["typeDescKey"] == "goal"]
        .groupby(["scoringPlayerId", "situation"])
        .size()
        .reset_index(name="sog")
    )
    sog_goals.rename(columns={"scoringPlayerId": "playerId"}, inplace=True)
    sog = (
        pd.concat([sog_shots, sog_goals])
        .groupby(["playerId", "situation"])["sog"]
        .sum()
        .reset_index()
    )

    # ===== PLAYS-BASED STATS =====

    # Blocked shots (as shooter - shots that got blocked)
    shots_blocked_against = (
        plays[plays["typeDescKey"] == "blocked-shot"]
        .dropna(subset=["shootingPlayerId"])
        .groupby(["shootingPlayerId", "situation"])
        .size()
        .reset_index(name="shots_blocked_against")
    )
    shots_blocked_against.rename(columns={"shootingPlayerId": "playerId"}, inplace=True)

    # Corsi (Fenwick + blocked shots)
    corsi = fenwick.copy()
    corsi.rename(columns={"fenwick": "corsi"}, inplace=True)
    corsi = corsi.merge(shots_blocked_against, on=["playerId", "situation"], how="left")
    corsi["corsi"] = corsi["corsi"] + corsi["shots_blocked_against"].fillna(0)
    corsi.drop(columns=["shots_blocked_against"], inplace=True)

    # Giveaways
    giveaways = (
        plays[plays["typeDescKey"] == "giveaway"]
        .groupby(["playerId", "situation"])
        .size()
        .reset_index(name="giveaways")
    )

    # Takeaways
    takeaways = (
        plays[plays["typeDescKey"] == "takeaway"]
        .groupby(["playerId", "situation"])
        .size()
        .reset_index(name="takeaways")
    )

    # Hits (as hitter)
    hits_for = (
        plays[plays["typeDescKey"] == "hit"]
        .dropna(subset=["hittingPlayerId"])
        .groupby(["hittingPlayerId", "situation"])
        .size()
        .reset_index(name="hits_for")
    )
    hits_for.rename(columns={"hittingPlayerId": "playerId"}, inplace=True)

    # Hits taken (as hittee)
    hits_taken = (
        plays[plays["typeDescKey"] == "hit"]
        .dropna(subset=["hitteePlayerId"])
        .groupby(["hitteePlayerId", "situation"])
        .size()
        .reset_index(name="hits_taken")
    )
    hits_taken.rename(columns={"hitteePlayerId": "playerId"}, inplace=True)

    # Shots blocked (as blocker)
    shots_blocked_for = (
        plays[plays["typeDescKey"] == "blocked-shot"]
        .dropna(subset=["blockingPlayerId"])
        .groupby(["blockingPlayerId", "situation"])
        .size()
        .reset_index(name="blocks")
    )
    shots_blocked_for.rename(columns={"blockingPlayerId": "playerId"}, inplace=True)

    # Faceoffs won
    faceoffs_won = (
        plays[plays["typeDescKey"] == "faceoff"]
        .dropna(subset=["winningPlayerId"])
        .groupby(["winningPlayerId", "situation"])
        .size()
        .reset_index(name="faceoffs_won")
    )
    faceoffs_won.rename(columns={"winningPlayerId": "playerId"}, inplace=True)

    # Faceoffs lost
    faceoffs_lost = (
        plays[plays["typeDescKey"] == "faceoff"]
        .dropna(subset=["losingPlayerId"])
        .groupby(["losingPlayerId", "situation"])
        .size()
        .reset_index(name="faceoffs_lost")
    )
    faceoffs_lost.rename(columns={"losingPlayerId": "playerId"}, inplace=True)

    # Penalties taken
    penalties_taken = (
        plays[plays["typeDescKey"] == "penalty"]
        .dropna(subset=["committedByPlayerId"])
        .groupby(["committedByPlayerId", "situation"])
        .size()
        .reset_index(name="penalties_taken")
    )
    penalties_taken.rename(columns={"committedByPlayerId": "playerId"}, inplace=True)

    # Penalties drawn
    penalties_drawn = (
        plays[plays["typeDescKey"] == "penalty"]
        .dropna(subset=["drawnByPlayerId"])
        .groupby(["drawnByPlayerId", "situation"])
        .size()
        .reset_index(name="penalties_drawn")
    )
    penalties_drawn.rename(columns={"drawnByPlayerId": "playerId"}, inplace=True)

    # ===== MERGE ALL STATS INTO BOXSCORE =====

    all_stats = [
        goals,
        assist1,
        assist2,
        fenwick,
        sog,
        corsi,
        giveaways,
        takeaways,
        hits_for,
        hits_taken,
        shots_blocked_for,
        faceoffs_won,
        faceoffs_lost,
        penalties_taken,
        penalties_drawn,
    ]

    for df_stat in all_stats:
        stat_col = [c for c in df_stat.columns if c not in ["playerId", "situation"]][0]
        boxscore = boxscore.merge(
            df_stat, on=["playerId", "situation"], how="left", suffixes=("", "_new")
        )
        if f"{stat_col}_new" in boxscore.columns:
            boxscore[stat_col] = boxscore[f"{stat_col}_new"].fillna(0)
            boxscore.drop(columns=[f"{stat_col}_new"], inplace=True)
        else:
            boxscore[stat_col] = boxscore[stat_col].fillna(0)

    return {"home_stints": stints, "away_stints": stints, "boxscore": boxscore}
