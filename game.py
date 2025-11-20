import nhl_scraper.games as ga
import nhl_scraper.graphs as gr

ga.getBoxScore(2025020313)["away_stints"].to_csv("test.csv")
