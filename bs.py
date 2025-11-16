import nhl_scraper.games as g

bs = g.getBoxScore(2025020290)

bs["boxscore"].to_csv("bs.csv")
