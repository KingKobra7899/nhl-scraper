import nhl_scraper.games as ga
import nhl_scraper.graphs as gr

ga.getBoxScore(2025020313)["boxscore"].to_csv("test.csv")
