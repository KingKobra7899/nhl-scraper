import nhl_scraper.games as g

bs = g.getBoxScore(2025020299)

bs["boxscore"].to_csv("bs.csv")
