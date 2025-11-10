import nhl_scraper.games as g
import nhl_scraper.player as p

game = g.scrapeGamesPbp(g.getTeamSeasonGames(3, 20252026))

puck_possess = g.getPlayerPuckPossession(game["plays"], game["shots"])

puck_possess["time"] = 5 / 60

puck_possess = (
    puck_possess.groupby(["playerId", "zoneCode"])["time"].sum().reset_index()
)
puck_possess["name"] = puck_possess["playerId"].apply(p.get_player_name)
puck_possess = puck_possess.sort_values(by="time", ascending=False)
# puck_possess = puck_possess[puck_possess["teamId"] == 3]
puck_possess.to_csv("puck_possession.csv")
