import requests
import pandas as pd
import numpy as np


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
    shots["angle"] = np.sin(np.arctan2(shots["xCoord"] - 89, shots["yCoord"]))
    return {
        "venue": venue,
        "homeTeamId": home,
        "awayTeamId": away,
        "plays": plays,
        "shots": shots[shots["periodType"] != "SO"],
    }
