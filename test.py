import nhl_scraper.games as g

print(g.getPbpData("2025020056")["plays"].to_csv("test.csv"))
