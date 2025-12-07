from functools import lru_cache
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import requests
from typing import List
from tqdm import tqdm
import joblib
import warnings
import sys
import traceback
import time
warnings.simplefilter(action="ignore", category=FutureWarning)


def str_to_float(time_str: str) -> float:
    parts = time_str.split(":")
    if len(parts) == 2:
        minutes = int(parts[0])
        seconds = float(parts[1])
        return minutes + (seconds / 60)
    else:
        print("ERROR")
        return -1


def getPbpData(gameid):
    """
    - Returns all play-by-play events for a game (that are tracked by NHL) \\
    - Shot-attempts are in a seperate df\\
    - Also returns the venue and home/away teams
    \\
    Parameters:\\
        gameid: the nhl's id for a game

    """

    url = f"https://api-web.nhle.com/v1/gamecenter/{gameid}/play-by-play"
    r = requests.get(url)
    try:
        data = r.json()
    except:
        retry_after = r.headers.get('Retry-After')
        if retry_after:
            wait_time = int(retry_after)
            print(f"Rate limited. Waiting {wait_time} seconds...")
            time.sleep(wait_time)
            # Retry the request
            r = requests.get(url)  # Replace with your actual request
            data = r.json()
        else:
            
            print("Rate limited but no Retry-After header. Waiting 60 seconds...")
            time.sleep(60)
            r = requests.get(url)
            data = r.json()

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

    shots["situation"] = shots.apply(
        lambda row: getSituation(row["situationCode"], row["isHome"]), axis=1
    )

    shots["venue"] = venue
    plays["venue"] = venue
    plays["game"] = gameid
    shots["game"] = gameid

    shots = shots[
        ~shots["situation"].isin(
            [
                "0v1",
                "1v0",
                "1v4",
                "1v5",
                "3v1",
                "4v1",
                "3v6",
                "6v3",
                "5v1",
            ]
        )
    ]
    X = pd.get_dummies(
        shots[
            [
                "xCoord",
                "yCoord",
                "shotType",
                "isHome",
                "timeRemaining",
                "periodType",
                "situation",
            ]
        ],
        columns=["periodType", "shotType", "situation"],
    )
    cols = [
        "xCoord",
        "yCoord",
        "isHome",
        "timeRemaining",
        "periodType_OT",
        "periodType_REG",
        "shotType_backhand",
        "shotType_bat",
        "shotType_between-legs",
        "shotType_cradle",
        "shotType_deflected",
        "shotType_poke",
        "shotType_slap",
        "shotType_snap",
        "shotType_tip-in",
        "shotType_wrap-around",
        "shotType_wrist",
        "situation_3v3",
        "situation_3v4",
        "situation_3v5",
        "situation_4v3",
        "situation_4v4",
        "situation_4v5",
        "situation_4v6",
        "situation_5v3",
        "situation_5v4",
        "situation_5v5",
        "situation_5v6",
        "situation_6v4",
        "situation_6v5",
    ]
    x = X.reindex(columns=cols, fill_value=0)
    xg_model = joblib.load("nhl_scraper/xg_model.joblib")

    shots["xG"] = xg_model.predict_proba(x)[:, 1]
    shots["isEN"] = shots["situation"] == "5v6"
    shots = shots[~shots["isEN"]]
    shots = shots.drop(columns="isEN")
    return {
        "venue": venue,
        "homeTeamId": home,
        "awayTeamId": away,
        "plays": plays,
        "shots": shots[shots["periodType"] != "SO"],
    }


def getGameString(gameid):
    url = f"https://api-web.nhle.com/v1/gamecenter/{gameid}/landing"
    response = requests.get(url)
    data = response.json()

    home_team = f"{data['homeTeam']['abbrev']}"
    away_team = f"{data['awayTeam']['abbrev']}"
    return f"{away_team} @ {home_team}"


@lru_cache(32)
def getTeamName(teamid):
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


def scrapeGamesPbp(games: List[str]):
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
def getTeamSeasonGames(team: str, season: int) -> List[str]:
    url = f"https://api-web.nhle.com/v1/club-schedule-season/{team}/{season}"
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
    try:
        df = pd.DataFrame(r.json()["data"])
    except:
        retry_after = r.headers.get('Retry-After')
        if retry_after:
            wait_time = int(retry_after)
            print(f"Rate limited. Waiting {wait_time} seconds...")
            time.sleep(wait_time)
            # Retry the request
            r = requests.get(url)  # Replace with your actual request
            df = pd.DataFrame(r.json()["data"])
        else:
            
            print("Rate limited but no Retry-After header. Waiting 60 seconds...")
            time.sleep(60)
            r = requests.get(url)
            df = pd.DataFrame(r.json()["data"])
    df = df[df["detailCode"] == 0]

    df["startTime"] = df.startTime.apply(str_to_float) + (df.period - 1) * 20
    df["endTime"] = df.endTime.apply(str_to_float) + (df.period - 1) * 20

    time_points = pd.unique(pd.concat([df.startTime, df.endTime]))
    time_points.sort()

    stints = pd.DataFrame({"start": time_points[:-1], "end": time_points[1:]})
    stints["duration"] = stints["end"] - stints["start"]
    stints["stintId"] = stints.index

    df["stintIds"] = df.apply(
        lambda row: stints[
            (stints["start"] < row["endTime"]) & (stints["end"] > row["startTime"])
        ].stintId.tolist(),
        axis=1,
    )

    return df, stints


def getSituation(situationCode, is_home: bool) -> str:
    if pd.isna(situationCode):
        return "0v0"
    s_code = str(situationCode)
    if len(s_code) == 4:
        h = s_code[1]
        a = s_code[2]
        return f"{a}v{h}" if is_home else f"{h}v{a}"
    elif len(s_code) == 3:
        s_code = "0" + s_code
        return getSituation(s_code, is_home)


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


def getBoxScore(gameId) -> dict[str, pd.DataFrame]:
    url = f"https://api-web.nhle.com/v1/gamecenter/{str(gameId)}/boxscore"
    r = requests.get(url)

    try:
        data = r.json()
    except:
        retry_after = r.headers.get('Retry-After')
        if retry_after:
            wait_time = int(retry_after)
            print(f"Rate limited. Waiting {wait_time} seconds...")
            time.sleep(wait_time)
            # Retry the request
            r = requests.get(url)  # Replace with your actual request
            data = r.json()
        else:
            
            print("Rate limited but no Retry-After header. Waiting 60 seconds...")
            time.sleep(60)
            r = requests.get(url)
            data = r.json()

    players_stats = data["playerByGameStats"]
    home_team_id = int(getPbpData(gameId).get("homeTeamId", 0))
    away_team_id = int(getPbpData(gameId).get("awayTeamId", 0))

    teams_data = [
        (players_stats["homeTeam"], home_team_id),
        (players_stats["awayTeam"], away_team_id),
    ]
    metadata_frames = []
    goalies_list = []

    for team_data, team_id in teams_data:
        for pos_group in ["forwards", "defense", "goalies"]:
            df = pd.DataFrame(team_data[pos_group])
            if not df.empty:
                df = df[["playerId", "sweaterNumber", "position"]].assign(
                    teamId=team_id
                )
                metadata_frames.append(df)
                if pos_group == "goalies":
                    goalies_list.extend(df["playerId"].tolist())

    metadata = pd.concat(metadata_frames, ignore_index=True)

    pbp = getPbpData(gameId)
    shots = pbp["shots"].copy()
    plays = pbp["plays"].copy()
    player_shifts, stints = generateGameShifts(gameId)

    # Filter out goalies
    player_shifts = player_shifts[~player_shifts["playerId"].isin(goalies_list)].copy()

    # Use pd.IntervalIndex for accurate and vectorized lookups
    stints_interval = pd.IntervalIndex.from_arrays(
        stints["start"], stints["end"], closed="right"
    )

    def assign_stint_idx(df, intervals):
        time_series = df["timeInPeriod"]

        cut_result = pd.cut(
            time_series, bins=intervals, labels=False, include_lowest=False, right=True
        )

        stint_codes = cut_result.cat.codes
        return stint_codes.astype(int)

    shots["stint_idx"] = assign_stint_idx(shots, stints_interval)
    plays["stint_idx"] = assign_stint_idx(plays, stints_interval)

    all_players = player_shifts["playerId"].unique()
    situations = ["ev", "pp", "pk"]

    # Use pd.MultiIndex.from_product and reset_index for creation
    midx = pd.MultiIndex.from_product(
        [all_players, situations], names=["playerId", "situation"]
    )
    boxscore = pd.DataFrame(index=midx).reset_index()

    names = player_shifts[["playerId", "firstName", "lastName"]].drop_duplicates(
        "playerId"
    )
    names["name"] = names["firstName"] + " " + names["lastName"]

    boxscore = boxscore.merge(names[["playerId", "name"]], on="playerId", how="left")
    boxscore["name"] = boxscore["name"].fillna(boxscore["playerId"].astype(str))

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
        "xgf",
        "xga",
        "ff",
        "fa",
        "sogf",
        "soga",
        "gf",
        "ga",
    ]
    boxscore = boxscore.reindex(
        columns=boxscore.columns.tolist() + stat_cols, fill_value=0.0
    )

    # Explode shifts and merge with stints
    player_stints = player_shifts[["playerId", "teamId", "stintIds"]].explode(
        "stintIds"
    )
    player_stints = player_stints.merge(
        stints, left_on="stintIds", right_index=True, how="left"
    )
    player_stints["toi"] = player_stints["end"] - player_stints["start"]
    player_stints["total_toi"] = player_stints["toi"]  # Rename for final aggregation

    # Calculate stint-level metrics (FF/FA/xGF/xGA/etc.)

    # Vectorized assignment of CF/CA and goals/sogs
    shots["home_attack"] = shots["isHome"]
    shots["away_attack"] = ~shots["isHome"]

    shots["is_sog"] = shots["typeDescKey"].isin(["shot-on-goal", "goal"])
    shots["is_goal"] = shots["typeDescKey"].eq("goal")
    shots["is_fenwick"] = shots["typeDescKey"].isin(
        ["shot-on-goal", "goal", "missed-shot"]
    )

    # Home Team Metrics (F=For, A=Against)
    shots["ff_home"] = shots["home_attack"].astype(int)
    shots["fa_home"] = shots["away_attack"].astype(int)
    shots["sogf_home"] = (shots["home_attack"] & shots["is_sog"]).astype(int)
    shots["soga_home"] = (shots["away_attack"] & shots["is_sog"]).astype(int)
    shots["gf_home"] = (shots["home_attack"] & shots["is_goal"]).astype(int)
    shots["ga_home"] = (shots["away_attack"] & shots["is_goal"]).astype(int)
    shots["xgf_home"] = shots["xG"] * shots["home_attack"]
    shots["xga_home"] = shots["xG"] * shots["away_attack"]

    # Aggregate only the required columns
    agg_cols = [
        "ff_home",
        "fa_home",
        "sogf_home",
        "soga_home",
        "gf_home",
        "ga_home",
        "xgf_home",
        "xga_home",
    ]
    agg = shots.groupby("stint_idx")[agg_cols].sum()
    stint_metrics = agg.reindex(stints.index, fill_value=0)

    # Merge stint metrics onto player stints
    player_stints = player_stints.merge(
        stint_metrics, left_on="stintIds", right_index=True, how="left"
    ).fillna(0)  # Fill NaN from left merge with 0

    # Vectorized assignment of team-relative metrics (FF, FA, xGF, xGA, etc.)
    is_home_mask = player_stints["teamId"] == home_team_id

    metrics_to_process = ["ff", "fa", "sogf", "soga", "gf", "ga", "xgf", "xga"]

    for metric in metrics_to_process:
        # 1. Column for Home Player (straight assignment)
        home_col = f"{metric}_home"
        player_stints.loc[is_home_mask, metric] = player_stints.loc[
            is_home_mask, home_col
        ]

        # 2. Column for Away Player (swapped assignment)

        # Determine the opposite metric name (ff <-> fa, sogf <-> soga, etc.)
        if metric.endswith("f"):
            # If 'ff', opposite is 'fa'
            base_opposite_metric = metric[:-1] + "a"
        elif metric.endswith("a"):
            # If 'fa', opposite is 'ff'
            base_opposite_metric = metric[:-1] + "f"
        else:
            # Should not happen with this list, but a safeguard
            continue

        # The column to pull for the away player is the home team's opposite metric.
        opposite_home_col = f"{base_opposite_metric}_home"

        player_stints.loc[~is_home_mask, metric] = player_stints.loc[
            ~is_home_mask, opposite_home_col
        ]

    # Assign situation
    stint_sit = shots.groupby("stint_idx")["situationCode"].first()
    player_stints["situationCode"] = player_stints["stintIds"].map(stint_sit)
    vec_get_sit = np.vectorize(get_sit)
    player_stints["situation"] = vec_get_sit(
        player_stints["situationCode"], player_stints["teamId"] == home_team_id
    )
    player_stints["situation"] = player_stints["situation"].fillna(
        "ev"
    )  # Default to 'ev'

    # Final aggregation and merge to boxscore
    player_metrics = (
        player_stints.groupby(["playerId", "situation"])[
            ["ff", "fa", "sogf", "soga", "xgf", "xga", "gf", "ga", "total_toi"]
        ]
        .sum()
        .reset_index()
    )

    # Simplified merge and update for on-ice metrics
    boxscore = boxscore.merge(
        player_metrics, on=["playerId", "situation"], how="left", suffixes=("", "_agg")
    ).fillna(0)
    for col in ["ff", "fa", "sogf", "soga", "xgf", "xga", "gf", "ga", "total_toi"]:
        boxscore[col] = boxscore[f"{col}_agg"]
        boxscore.drop(columns=[f"{col}_agg"], inplace=True)

    # 6. Play-by-Play Events Stats (Goals, Assists, Hits, etc.)

    # Vectorized situation assignment for shots and plays
    is_home_mask = shots["teamId"] == home_team_id
    shots["situation"] = vec_get_sit(shots["situationCode"], is_home_mask)
    shots["situation"] = shots["situation"].fillna("ev")

    is_home_mask = plays["teamId"] == home_team_id
    plays["situation"] = vec_get_sit(plays["situationCode"], is_home_mask)
    plays["situation"] = plays["situation"].fillna("ev")

    # Define a reusable function for event counting
    def count_events(df, event_type, player_col, stat_name):
        df_event = df[df["typeDescKey"] == event_type].dropna(subset=[player_col])
        return (
            df_event.groupby([player_col, "situation"])
            .size()
            .reset_index(name=stat_name)
            .rename(columns={player_col: "playerId"})
        )

    # Goals, Assists, xG
    shots_goal = shots[shots["typeDescKey"] == "goal"]
    goals = count_events(shots_goal, "goal", "scoringPlayerId", "goals")
    assist1 = count_events(shots_goal, "goal", "a1PlayerId", "a1")
    assist2 = count_events(shots_goal, "goal", "a2PlayerId", "a2")

    # Calculate xG (used scoringPlayerId for goals, shootingPlayerId for others)
    shots["playerId"] = np.where(
        shots["typeDescKey"] == "goal",
        shots["scoringPlayerId"],
        shots["shootingPlayerId"],
    )
    xg = (
        shots.dropna(subset=["playerId"])
        .groupby(["playerId", "situation"])
        .agg({"xG": "sum"})
        .reset_index()
    )

    # Fenwick & SOG (Non-goals + Goals)
    fenwick_stats = []
    sog_stats = []

    # Fenwick (all attempts excluding blocked)
    for player_col in ["shootingPlayerId", "scoringPlayerId"]:
        is_shooting = (
            shots["typeDescKey"].isin(["shot-on-goal", "missed-shot"])
            & shots[player_col].notna()
        )
        is_scoring = (shots["typeDescKey"] == "goal") & (shots[player_col].notna())

        df_fen = shots[is_shooting | is_scoring].dropna(subset=[player_col])
        if not df_fen.empty:
            fenwick_stats.append(
                df_fen.groupby([player_col, "situation"])
                .size()
                .reset_index(name="fenwick")
                .rename(columns={player_col: "playerId"})
            )

    fenwick = (
        pd.concat(fenwick_stats, ignore_index=True)
        .groupby(["playerId", "situation"], as_index=False)["fenwick"]
        .sum()
    )

    # SOG (Shots on goal + Goals)
    for player_col in ["shootingPlayerId", "scoringPlayerId"]:
        is_shooting = (shots["typeDescKey"] == "shot") & (shots[player_col].notna())
        is_scoring = (shots["typeDescKey"] == "goal") & (shots[player_col].notna())

        df_sog = shots[is_shooting | is_scoring].dropna(subset=[player_col])
        if not df_sog.empty:
            sog_stats.append(
                df_sog.groupby([player_col, "situation"])
                .size()
                .reset_index(name="sog")
                .rename(columns={player_col: "playerId"})
            )

    sog = (
        pd.concat(sog_stats, ignore_index=True)
        .groupby(["playerId", "situation"], as_index=False)["sog"]
        .sum()
    )

    # Corsi (Fenwick + Blocked Against)
    shots_blocked_against = count_events(
        plays, "blocked-shot", "shootingPlayerId", "shots_blocked_against"
    )
    corsi = (
        fenwick.merge(shots_blocked_against, on=["playerId", "situation"], how="left")
        .assign(corsi=lambda df: df["fenwick"] + df["shots_blocked_against"].fillna(0))
        .drop(columns=["shots_blocked_against"])
    )

    # Other PBP stats
    giveaways = count_events(plays, "giveaway", "playerId", "giveaways")
    takeaways = count_events(plays, "takeaway", "playerId", "takeaways")
    hits_for = count_events(plays, "hit", "hittingPlayerId", "hits_for")
    hits_taken = count_events(plays, "hit", "hitteePlayerId", "hits_taken")
    shots_blocked_for = count_events(
        plays, "blocked-shot", "blockingPlayerId", "blocks"
    )
    faceoffs_won = count_events(plays, "faceoff", "winningPlayerId", "faceoffs_won")
    faceoffs_lost = count_events(plays, "faceoff", "losingPlayerId", "faceoffs_lost")
    penalties_taken = count_events(
        plays, "penalty", "committedByPlayerId", "penalties_taken"
    )
    penalties_drawn = count_events(
        plays, "penalty", "drawnByPlayerId", "penalties_drawn"
    )

    # Combine and merge all stats
    all_stats_dfs = [
        goals,
        xg,
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

    # Merge all stats in a single loop
    for df_stat in all_stats_dfs:
        stat_col = df_stat.columns[-1]
        boxscore = boxscore.merge(
            df_stat, on=["playerId", "situation"], how="left", suffixes=("", "_new")
        )
        # Use np.where for vectorized update if the column exists
        new_col = f"{stat_col}_new"
        if new_col in boxscore.columns:
            boxscore[stat_col] = np.where(
                boxscore[new_col].notna(), boxscore[new_col], boxscore[stat_col]
            )
            boxscore.drop(columns=[new_col], inplace=True)
        # Handle the edge case where the column may already exist and was not updated
        boxscore[stat_col] = boxscore[stat_col].fillna(0)

    # 7. On-Ice Player Matrix (Simplified)
    player_stints["on"] = 1
    onice = player_stints.pivot_table(
        index="stintIds",
        columns="playerId",
        values="on",
        aggfunc="max",
        fill_value=0,
    )

    home_players = player_shifts[player_shifts["teamId"] == home_team_id][
        "playerId"
    ].unique()
    away_players = player_shifts[player_shifts["teamId"] == away_team_id][
        "playerId"
    ].unique()

    # Join metrics to stints *once*
    stints_metrics = stints.join(stint_metrics, how="left").fillna(0)

    # Create home and away stints frames by joining on-ice matrices
    home_stints = stints_metrics.copy()
    away_stints = stints_metrics.copy()

    # Optimized column creation/renaming for home stints
    home_for = onice[home_players].add_prefix("for_")
    home_against = onice[away_players].add_prefix("against_")
    home_stints = home_stints.join(home_for).join(home_against).fillna(0)

    home_stints = home_stints.rename(
        columns={
            "ff_home": "ff",
            "fa_home": "fa",
            "sogf_home": "sogf",
            "soga_home": "soga",
            "xgf_home": "xgf",
            "xga_home": "xga",
            "gf_home": "gf",
            "ga_home": "ga",
        }
    )

    # Optimized column creation/renaming for away stints (swapped F/A)
    away_for = onice[away_players].add_prefix("for_")
    away_against = onice[home_players].add_prefix("against_")
    away_stints = away_stints.join(away_for).join(away_against).fillna(0)

    away_stints = away_stints.rename(
        columns={
            "ff_home": "fa",
            "fa_home": "ff",
            "sogf_home": "soga",
            "soga_home": "sogf",
            "xgf_home": "xga",
            "xga_home": "xgf",
            "gf_home": "ga",
            "ga_home": "gf",
        }
    )

    # 8. Final Calculation and Cleanup

    # Percentage calculations (handle division by zero with fillna)
    boxscore["f%"] = (boxscore["ff"] / (boxscore["ff"] + boxscore["fa"])).fillna(0)
    boxscore["s%"] = (boxscore["sogf"] / (boxscore["sogf"] + boxscore["soga"])).fillna(
        0
    )
    boxscore["g%"] = (boxscore["gf"] / (boxscore["gf"] + boxscore["ga"])).fillna(0)
    boxscore["xg%"] = (boxscore["xgf"] / (boxscore["xgf"] + boxscore["xga"])).fillna(0)

    # Add gameId and isHome/away flag
    home_stints["game"] = gameId
    away_stints["game"] = gameId
    boxscore["game"] = gameId
    home_stints["isHome"] = 1
    away_stints["isHome"] = 0

    # Merge player teams for metadata
    player_teams = metadata[["playerId", "teamId"]].drop_duplicates()
    boxscore = pd.merge(boxscore, player_teams, on="playerId", how="left")

    shots = shots.drop(columns=["scoringPlayerId", "shootingPlayerId"])
    shots = pd.merge(
        shots,
        boxscore[["playerId", "name"]].drop_duplicates(),
        on="playerId",
        how="left",
    )

    # 9. Zone Assignment

    # Prepare faceoff lookup table
    faceoffs = plays.loc[
        plays["typeDescKey"] == "faceoff", ["timeInPeriod", "zoneCode", "teamId"]
    ].rename(
        columns={
            "timeInPeriod": "start",
            "zoneCode": "raw_zone",
            "teamId": "faceoff_owner",
        }
    )

    # Merge once
    stints_merged = stints_metrics.merge(faceoffs, on="start", how="left")
    stints_merged["raw_zone"] = stints_merged["raw_zone"].fillna("FLY")
    stints_merged["faceoff_owner"] = stints_merged["faceoff_owner"].astype("float")

    # Attach zone/owner
    zone_data = stints_merged[["raw_zone", "faceoff_owner"]]
    home_stints = home_stints.join(zone_data)
    away_stints = away_stints.join(zone_data)

    def compute_zone_vectorized(df, is_home_view):
        z = df["raw_zone"].copy()
        owner = df["faceoff_owner"]

        # Mask for when the zone needs to be flipped (O <-> D)
        # Flip if: (zone is O or D) AND ((owner is away team) IF home view) OR ((owner is home team) IF away view)
        is_home_owner = owner == home_team_id
        flip_mask = (
            (z.isin(["O", "D"])) & (owner.notna()) & (is_home_owner != is_home_view)
        )

        z.loc[flip_mask & (z == "O")] = "D"
        z.loc[flip_mask & (z == "D")] = "O"
        return z

    home_stints["zone"] = compute_zone_vectorized(home_stints, True)
    away_stints["zone"] = compute_zone_vectorized(away_stints, False)

    # Drop intermediate columns
    home_stints = home_stints.drop(columns=["raw_zone", "faceoff_owner"])
    away_stints = away_stints.drop(columns=["raw_zone", "faceoff_owner"])

    # 10. Manpower Assignment (Simplified)

    # Create a Series for the first event in each stint_idx
    all_events = pd.concat(
        [shots.assign(is_shot=1), plays.assign(is_shot=0)]
    ).sort_values("timeInPeriod")

    # Filter only events that fall within a defined stint
    events_in_stints = all_events[all_events["stint_idx"] != -1]

    # Get the *first* event within each stint index
    first_events = (
        events_in_stints.sort_values("timeInPeriod").groupby("stint_idx").first()
    )

    # Map situation to stints (and then to home/away stints)
    stint_manpower = vec_get_sit(
        first_events["situationCode"], first_events["teamId"] == home_team_id
    )
    manpower_series = pd.Series(stint_manpower, index=first_events.index)

    # 2. Convert the Series to a dictionary for use in .map()
    manpower_map = manpower_series.to_dict()
    home_stints["manpower"] = stints.index.map(manpower_map).fillna("5v5")

    # Recalculate away manpower using the correct view
    away_stint_manpower = vec_get_sit(
        first_events["situationCode"], first_events["teamId"] == away_team_id
    )

    away_manpower_map = pd.Series(
        away_stint_manpower, index=first_events.index
    ).to_dict()

    away_stints["manpower"] = stints.index.map(away_manpower_map).fillna("5v5")

    penalty_events = plays[plays["typeDescKey"] == "penalty"]

    def update_manpower_penalty_optimized(
        stints_df, penalty_events, all_events, home_team_id
    ):
        if penalty_events.empty:
            return stints_df

        stints_df = stints_df.copy()

        # Prepare lookup for the manpower *after* the penalty
        other_events = all_events[all_events["typeDescKey"] != "penalty"].sort_values(
            "timeInPeriod"
        )

        for _, pen in penalty_events.iterrows():
            # Get the first non-penalty event that starts on or after the penalty time
            next_ev = other_events[
                other_events["timeInPeriod"] >= pen["timeInPeriod"]
            ].head(1)

            if next_ev.empty or next_ev["situationCode"].iloc[0] is None:
                continue

            
            is_home_view = stints_df["isHome"].iloc[0] == 1

            manpower_to_propagate = get_sit(
                next_ev["situationCode"].iloc[0], is_home_view
            )

            penalty_start = pen["timeInPeriod"]
            penalty_end = penalty_start + pen["duration"]

            # Check for goals that end the penalty early (within the initial duration)
            goals_in_penalty = all_events[
                (all_events["typeDescKey"] == "goal")
                & (all_events["timeInPeriod"] >= penalty_start)
                & (all_events["timeInPeriod"] < penalty_end)
            ]
            if not goals_in_penalty.empty:
                penalty_end = goals_in_penalty["timeInPeriod"].min()

            # Apply the new manpower to relevant 5v5 stints
            mask = (
                (stints_df["start"] < penalty_end)
                & (stints_df["end"] > penalty_start)
                & (stints_df["manpower"] == "5v5")
            )
            stints_df.loc[mask, "manpower"] = manpower_to_propagate

        return stints_df

    
    home_stints = update_manpower_penalty_optimized(
        home_stints, penalty_events, all_events, home_team_id
    )
    away_stints = update_manpower_penalty_optimized(
        away_stints, penalty_events, all_events, home_team_id
    )
    home_stints["teamId"] = home_team_id
    away_stints["teamId"] = away_team_id
    return {
        "home_stints": home_stints,
        "away_stints": away_stints,
        "shots": shots,
        "boxscore": boxscore.merge(
            metadata.drop(columns=["teamId"]), on="playerId", how="left"
        ).drop_duplicates(subset=["playerId", "situation"]),
    }


def getGamesBoxscore(games: list[str]) -> dict[str, pd.DataFrame]:
    stints_list = []
    shots_list = []
    boxscore_list = []

    for game in tqdm(games):
        try:
            result = getBoxScore(game)
            stints_list.append(result["home_stints"])
            stints_list.append(result["away_stints"])
            shots_list.append(result["shots"])
            boxscore_list.append(result["boxscore"])
        except Exception as e:
            print(f"Error processing game {game}: {e}")
            
    stints = pd.concat(stints_list, ignore_index=True)
    shots = pd.concat(shots_list, ignore_index=True)
    boxscore = pd.concat(boxscore_list, ignore_index=True)

            

    return {"stints": stints, "shots": shots, "boxscore": boxscore}
