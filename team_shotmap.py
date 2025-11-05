import nhl_scraper.games as g

shots = g.scrapeGamesShots(g.getTeamSeasonGames(23, 20252026))


g.plot_team_shot_density(shots, 23, mode="diff", sigma=5)
