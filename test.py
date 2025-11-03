import nhl_scraper.games as g

if __name__ == "__main__":
    g.plot_game_shot_density("2025020197", mode='diff', sigma=7)
