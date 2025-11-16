import nhl_scraper.games as g
import nhl_scraper.player as p
import pandas as pd
import seaborn as sns

shots = g.scrapeGamesPbp(g.getSeasonPlayedGames(20202021, 20242025))["shots"]

shots.to_csv("current_season.csv")
