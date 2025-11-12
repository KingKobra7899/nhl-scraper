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

    # Convert to absolute game time
    df["startTime"] = df.startTime.apply(str_to_float) + (df.period - 1) * 20
    df["endTime"] = df.endTime.apply(str_to_float) + (df.period - 1) * 20

    # Create all unique stint boundaries
    bounds = pd.unique(pd.concat([df.startTime, df.endTime]))
    bounds.sort()

    # Create stints DataFrame
    stints = pd.DataFrame(
        {"stintId": range(len(bounds) - 1), "start": bounds[:-1], "end": bounds[1:]}
    )
    stints["duration"] = stints["end"] - stints["start"]

    # Assign list of stintIds to each player shift
    def overlapping_stints(row):
        overlapping = stints[
            (stints["start"] < row["endTime"]) & (stints["end"] > row["startTime"])
        ]
        return overlapping["stintId"].tolist()

    df["stintIds"] = df.apply(overlapping_stints, axis=1)

    return df, stints


def getSituation(situationCode: int | str, is_home: bool) -> str:
    h = str(situationCode)[1]
    a = str(situationCode)[2]
    return f"{h}v{a}" if is_home else f"{a}v{h}"


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

    # Add player names
    def resolve_name(pid):
        row = player_shifts[player_shifts["playerId"] == pid]
        if row.empty:
            return str(pid)
        r = row.iloc[0]
        return f"{r['firstName']} {r['lastName']}"

    # Build home/away stints (on-ice metrics)
    def make_stints(df):
        df = df.copy()
        df["cf"] = 0
        df["ca"] = 0
        df["sogf"] = 0
        df["soga"] = 0
        df["gf"] = 0
        df["ga"] = 0
        df["toi"] = df["end"] - df["start"]
        return df

    home_stints = make_stints(stints)
    away_stints = make_stints(stints)

    # Build player boxscore (individual stats per situation)
    all_players = sorted(player_shifts["playerId"].unique())
    situations = ["ev", "pp", "pk"]
    box_rows = [(pid, sit) for pid in all_players for sit in situations]
    boxscore = pd.DataFrame(box_rows, columns=["playerId", "situation"])
    boxscore["name"] = [resolve_name(pid) for pid, _ in box_rows]

    for col in [
        "goals",
        "assist1",
        "assist2",
        "shot_attempts",
        "sog",
        "giveaways",
        "takeaways",
        "hits",
        "shots_blocked",
        "faceoffs_won",
        "faceoffs_lost",
        "penalties_committed",
        "penalties_drawn",
        "total_toi",
        "onice_cf",
        "onice_ca",
        "onice_sogf",
        "onice_soga",
        "onice_gf",
        "onice_ga",
    ]:
        boxscore[col] = 0.0

    # Add situationCode to shots and plays based on timeInPeriod
    # Match each event to the stint it occurred in
    shots["stint_idx"] = shots["timeInPeriod"].apply(
        lambda t: stints[(stints["start"] <= t) & (t <= stints["end"])].index[0]
        if len(stints[(stints["start"] <= t) & (t <= stints["end"])]) > 0
        else -1
    )

    plays["stint_idx"] = plays["timeInPeriod"].apply(
        lambda t: stints[(stints["start"] <= t) & (t <= stints["end"])].index[0]
        if len(stints[(stints["start"] <= t) & (t <= stints["end"])]) > 0
        else -1
    )

    # Process each stint (on-ice metrics)
    for idx, stint in stints.iterrows():
        start, end = stint["start"], stint["end"]
        stint_shots = shots[shots["stint_idx"] == idx]
        stint_toi = end - start

        cf_home = stint_shots["isHome"].sum()
        ca_home = (~stint_shots["isHome"]).sum()
        sogf_home = stint_shots[
            (stint_shots["typeDescKey"].isin(["shot-on-goal", "goal"]))
            & stint_shots["isHome"]
        ].shape[0]
        soga_home = stint_shots[
            (stint_shots["typeDescKey"].isin(["shot-on-goal", "goal"]))
            & (~stint_shots["isHome"])
        ].shape[0]
        gf_home = stint_shots[
            (stint_shots["typeDescKey"] == "goal") & stint_shots["isHome"]
        ].shape[0]
        ga_home = stint_shots[
            (stint_shots["typeDescKey"] == "goal") & (~stint_shots["isHome"])
        ].shape[0]

        # Fill stint stats
        home_stints.loc[idx, ["cf", "ca", "sogf", "soga", "gf", "ga", "toi"]] = [
            cf_home,
            ca_home,
            sogf_home,
            soga_home,
            gf_home,
            ga_home,
            stint_toi,
        ]
        away_stints.loc[idx, ["cf", "ca", "sogf", "soga", "gf", "ga", "toi"]] = [
            ca_home,
            cf_home,
            soga_home,
            sogf_home,
            ga_home,
            gf_home,
            stint_toi,
        ]

        # Get situationCode from a shot in this stint if available
        stint_situation = None
        if not stint_shots.empty and "situationCode" in stint_shots.columns:
            stint_situation = stint_shots.iloc[0]["situationCode"]

        # Update on-ice metrics for players
        home_players = player_shifts[
            (player_shifts["teamId"] == home_team_id)
            & (player_shifts["startTime"] <= end)
            & (player_shifts["endTime"] >= start)
        ]["playerId"].unique()

        away_players = player_shifts[
            (player_shifts["teamId"] == away_team_id)
            & (player_shifts["startTime"] <= end)
            & (player_shifts["endTime"] >= start)
        ]["playerId"].unique()

        if stint_situation:
            sit_home = get_sit(stint_situation, True)
            sit_away = get_sit(stint_situation, False)

            for pid in home_players:
                mask = (boxscore["playerId"] == pid) & (
                    boxscore["situation"] == sit_home
                )
                boxscore.loc[mask, "onice_cf"] += cf_home
                boxscore.loc[mask, "onice_ca"] += ca_home
                boxscore.loc[mask, "onice_sogf"] += sogf_home
                boxscore.loc[mask, "onice_soga"] += soga_home
                boxscore.loc[mask, "onice_gf"] += gf_home
                boxscore.loc[mask, "onice_ga"] += ga_home
                boxscore.loc[mask, "total_toi"] += stint_toi

            for pid in away_players:
                mask = (boxscore["playerId"] == pid) & (
                    boxscore["situation"] == sit_away
                )
                boxscore.loc[mask, "onice_cf"] += ca_home
                boxscore.loc[mask, "onice_ca"] += cf_home
                boxscore.loc[mask, "onice_sogf"] += soga_home
                boxscore.loc[mask, "onice_soga"] += sogf_home
                boxscore.loc[mask, "onice_gf"] += ga_home
                boxscore.loc[mask, "onice_ga"] += gf_home
                boxscore.loc[mask, "total_toi"] += stint_toi

    # Process shots for individual player stats
    for _, s in shots.iterrows():
        if "situationCode" not in s or pd.isna(s["situationCode"]):
            continue

        sit = get_sit(s["situationCode"], s["isHome"])

        # Goals
        if s["typeDescKey"] == "goal" and pd.notna(s.get("scoringPlayerId")):
            scorer = int(s["scoringPlayerId"])
            mask = (boxscore["playerId"] == scorer) & (boxscore["situation"] == sit)
            boxscore.loc[mask, "goals"] += 1
            boxscore.loc[mask, "shot_attempts"] += 1
            boxscore.loc[mask, "sog"] += 1

            for a, col in zip(
                [s.get("a1PlayerId"), s.get("a2PlayerId")], ["assist1", "assist2"]
            ):
                if pd.notna(a):
                    a = int(a)
                    mask = (boxscore["playerId"] == a) & (boxscore["situation"] == sit)
                    boxscore.loc[mask, col] += 1

        # Non-goal shots
        elif pd.notna(s.get("shootingPlayerId")):
            shooter_id = int(s["shootingPlayerId"])
            mask = (boxscore["playerId"] == shooter_id) & (boxscore["situation"] == sit)
            boxscore.loc[mask, "shot_attempts"] += 1
            if s["typeDescKey"] == "shot-on-goal":
                boxscore.loc[mask, "sog"] += 1

    # Process plays for remaining stats
    for _, ev in plays.iterrows():
        typ = ev["typeDescKey"]

        # Determine situation from the stint this play occurred in
        if ev["stint_idx"] == -1:
            continue

        stint_plays = shots[shots["stint_idx"] == ev["stint_idx"]]
        if stint_plays.empty or "situationCode" not in stint_plays.columns:
            continue

        situation_code = stint_plays.iloc[0]["situationCode"]

        # Determine which team the player belongs to
        pid = (
            ev.get("playerId") or ev.get("hitterPlayerId") or ev.get("winningPlayerId")
        )
        if pd.notna(pid):
            pid = int(pid)
            # Check if player is on home or away team
            is_home = (
                pid
                in player_shifts[player_shifts["teamId"] == home_team_id][
                    "playerId"
                ].values
            )
            sit = get_sit(situation_code, is_home)

            if typ == "giveaway":
                mask = (boxscore["playerId"] == pid) & (boxscore["situation"] == sit)
                boxscore.loc[mask, "giveaways"] += 1

            elif typ == "takeaway":
                mask = (boxscore["playerId"] == pid) & (boxscore["situation"] == sit)
                boxscore.loc[mask, "takeaways"] += 1

            elif typ == "hit":
                hitter = ev.get("hitterPlayerId")
                if pd.notna(hitter):
                    hitter = int(hitter)
                    is_home_h = (
                        hitter
                        in player_shifts[player_shifts["teamId"] == home_team_id][
                            "playerId"
                        ].values
                    )
                    sit_h = get_sit(situation_code, is_home_h)
                    mask_h = (boxscore["playerId"] == hitter) & (
                        boxscore["situation"] == sit_h
                    )
                    boxscore.loc[mask_h, "hits"] += 1

            elif typ == "faceoff":
                winner = ev.get("winningPlayerId")
                loser = ev.get("losingPlayerId")
                if pd.notna(winner):
                    winner = int(winner)
                    is_home_w = (
                        winner
                        in player_shifts[player_shifts["teamId"] == home_team_id][
                            "playerId"
                        ].values
                    )
                    sit_w = get_sit(situation_code, is_home_w)
                    boxscore.loc[
                        (boxscore["playerId"] == winner)
                        & (boxscore["situation"] == sit_w),
                        "faceoffs_won",
                    ] += 1
                if pd.notna(loser):
                    loser = int(loser)
                    is_home_l = (
                        loser
                        in player_shifts[player_shifts["teamId"] == home_team_id][
                            "playerId"
                        ].values
                    )
                    sit_l = get_sit(situation_code, is_home_l)
                    boxscore.loc[
                        (boxscore["playerId"] == loser)
                        & (boxscore["situation"] == sit_l),
                        "faceoffs_lost",
                    ] += 1

            elif typ == "penalty":
                comm = ev.get("commitedByPlayerId")
                drawn = ev.get("drawnByPlayerId")
                if pd.notna(comm):
                    comm = int(comm)
                    is_home_c = (
                        comm
                        in player_shifts[player_shifts["teamId"] == home_team_id][
                            "playerId"
                        ].values
                    )
                    sit_c = get_sit(situation_code, is_home_c)
                    boxscore.loc[
                        (boxscore["playerId"] == comm)
                        & (boxscore["situation"] == sit_c),
                        "penalties_committed",
                    ] += 1
                if pd.notna(drawn):
                    drawn = int(drawn)
                    is_home_d = (
                        drawn
                        in player_shifts[player_shifts["teamId"] == home_team_id][
                            "playerId"
                        ].values
                    )
                    sit_d = get_sit(situation_code, is_home_d)
                    boxscore.loc[
                        (boxscore["playerId"] == drawn)
                        & (boxscore["situation"] == sit_d),
                        "penalties_drawn",
                    ] += 1

    return {
        "home_stints": home_stints,
        "away_stints": away_stints,
        "boxscore": boxscore,
    }
