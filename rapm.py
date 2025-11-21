import nhl_scraper.games as ga
import joblib

games = ga.getSeasonPlayedGames(20242025, 20242025)

boxscore = ga.getGamesBoxscore(games)
joblib.dump(boxscore, "boxscore.pkl")
