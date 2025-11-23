import nhl_scraper.games as ga
import nhl_scraper.graphs as gr
import matplotlib.pyplot as plt

games = ga.getTeamSeasonGames("NYR", 20252026)
shots = ga.scrapeGamesPbp(games)["shots"]

fig = gr.plot_team_shot_density(
    shots, id="NYR", gp=len(games), mode="diff", xG=True, heightmap=False, sigma=2.5
)

plt.show()
