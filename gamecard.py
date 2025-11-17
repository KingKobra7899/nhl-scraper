import nhl_scraper.graphs as gr

gameid = 2025020299

gr.plot_game_shot_density(gameid, sigma=3, mode="diff")
