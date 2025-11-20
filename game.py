import nhl_scraper.games as ga
import nhl_scraper.graphs as gr

ga.getPbpData(2025020313)["shots"].to_csv("test.csv")
