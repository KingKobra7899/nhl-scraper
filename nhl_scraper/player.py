from functools import lru_cache
import requests
import pandas as pd


def get_player_pic_url(player_id):
    url = f"https://api-web.nhle.com/v1/player/{player_id}/landing"
    response = requests.get(url)
    data = response.json()
    return data["headshot"]


def get_player_metadata(player_id):
    url = f"https://api-web.nhle.com/v1/player/{player_id}/landing"
    response = requests.get(url)
    data = response.json()

    first_name = data["firstName"]["default"]
    last_name = data["lastName"]["default"]
    name = f"{first_name} {last_name}"
    height = data["heightInInches"]
    feet = height // 12
    inches = height % 12

    birth_date = data["birthDate"]
    birth_date = datetime.strptime(birth_date, "%Y-%m-%d")
    avg_toi_str = next(
        (
            s.get("avgToi")
            for s in data.get("seasonTotals", [])
            if s.get("season") == 20252026
        ),
        None,  # default if not found
    )

    return {
        "name": name,
        "team": data["fullTeamName"]["default"],
        "team_tricode": data["currentTeamAbbrev"],
        "team_logo": data["teamLogo"],
        "headshot": data["headshot"],
        "weight": data["weightInPounds"],
        "feet": feet,
        "inches": inches,
        "birth_date": birth_date,
        "sweater_number": data["sweaterNumber"],
        "position": data["position"],  # C, L, R, or D
        "handidness": data["shootsCatches"],  # L or R
        "toi_string": avg_toi_str,
        "age": math.floor(((datetime.today() - birth_date).days) / 365),
    }


@lru_cache(None)
def get_player_name(player_id):
    url = f"https://api-web.nhle.com/v1/player/{int(player_id)}/landing"
    if int(player_id) == 8480012:
        return "Elias Pettersson (F)"
    elif int(player_id) == 8483678:
        return "Elias Pettersson (D)"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()

        first_name = data["firstName"]["default"]
        last_name = data["lastName"]["default"]
        return f"{first_name} {last_name}"
    except Exception as e:
        print(f"Failed to fetch player name for ID {player_id}: {e}")
        return "N/A"


def getTeamSeasonRoster(team, season, positions=["forwards", "defensemen", "goalies"]):
    url = f"https://api-web.nhle.com/v1/roster/{team}/{season}"
    data = requests.get(url).json()
    players = []
    for pos in positions:
        df = pd.DataFrame(data[pos])
        players.extend(df["id"].to_list())
    return players


if __name__ == "__main__":
    print(getTeamSeasonRoster("NYR", 20242025, ["forwards"]))
