import nhl_scraper.games as ga
import nhl_scraper.graphs as gr
import matplotlib.pyplot as plt

games = ga.getTeamSeasonGames("CHI", 20252026)
shots = ga.scrapeGamesPbp(games)["shots"]

fig = gr.plot_team_shot_density(
    shots, id="CHI", gp=len(games), mode="diff", xG=True, heightmap=False, sigma=3
)

plt.show()
