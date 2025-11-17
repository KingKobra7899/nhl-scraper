import nhl_scraper.graphs as gr
import nhl_scraper.games as ga
import matplotlib.pyplot as plt

team_id = 21

games = ga.getTeamSeasonGames(team_id, 20252026)
shots = ga.scrapeGamesPbp(games)["shots"]
fig = gr.plot_team_shot_density(
    df=shots, id=team_id, gp=len(games), mode="diff", sigma=3
)
plt.show()
