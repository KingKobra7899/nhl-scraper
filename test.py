import nhl_scraper.games as g

game = g.getPbpData(2025020251)

puck_possess = g.getPlayerPuckPossession(game["plays"], game["shots"])
