import nhl_scraper.games as ga
import nhl_scraper.graphs as gr
import matplotlib.pyplot as plt

gr.plot_game_shot_density(2025020326, mode="diff", sigma=3, xG=True)
plt.show()
