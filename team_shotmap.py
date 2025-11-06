import nhl_scraper.games as g

games = g.getTeamSeasonGames(3, 20242025)

for i in range(1, len(games) + 1):
    game = games[:i]
    shots = g.scrapeGamesShots(game)
    fig = g.plot_team_shot_density(df=shots, id=3, gp=i, mode="diff", sigma=5)
    fig.savefig(f"img/NYR_{i}.png")

games = g.getSeasonPlayedGames(20102011, 20242025)
g.scrapeGamesShots(games).to_csv("shots.csv")
