import nhl_scraper.games as ga
import nhl_scraper.graphs as gr
import matplotlib.pyplot as plt

games = ga.getTeamSeasonGames(3, 20252026)
shots = ga.scrapeGamesPbp(games)["shots"]

gr.plot_team_shot_density(shots, 3, len(games), mode="diff", sigma=3, xG=True)
plt.show()
