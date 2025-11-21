import nhl_scraper.games as ga
import nhl_scraper.graphs as gr
import matplotlib

fig = ga.getBoxScore(2025020326)["shots"].to_csv("test.csv")
