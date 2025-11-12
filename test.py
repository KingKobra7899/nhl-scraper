import nhl_scraper.games as g
import nhl_scraper.player as p

stints = g.getBoxScore(2025020253)

stints["boxscore"].to_csv("test.csv")
