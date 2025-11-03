from typing import Tuple
import requests
import pandas as pd


def getPbpData(gameid: str):
    """
    - Returns all play-by-play events for a game (that are tracked by NHL) \\
    - Shot-attempts are in a seperate df\\
    \\
    Parameters:\\
        gameid: the nhl's id for a game
    """

    url = f"https://api-web.nhle.com/v1/gamecenter/{gameid}/play-by-play"
    response = requests.get(url)
    data = response.json()

    return print(data)
