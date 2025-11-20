import nhl_scraper.games as ga

games = ga.getSeasonPlayedGames(20202021, 20242025)
shots = ga.scrapeGamesPbp(games)["shots"]
shots.to_csv("tests/shots.csv")
