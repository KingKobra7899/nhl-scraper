import nhl_scraper.games as g
import nhl_scraper.player as p
import pandas as pd
import seaborn as sns

shots: pd.DataFrame = g.getPbpData(2025020253)["shots"]


import nhl_scraper.games as g
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load shots data
shots: pd.DataFrame = g.getPbpData(2019020203)["shots"]

# Set default coloring variable
color_var = "typeDescKey"

# Get unique teams (assumes only two)
teams = shots["teamId"].unique()

# Create a subplot for each team
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True, sharey=True)

for ax, team in zip(axes, teams):
    team_shots = shots[shots["teamId"] == team]
    g.draw_rink_features(ax, color="black")
    sns.scatterplot(
        data=team_shots, x="xCoord", y="yCoord", hue=color_var, palette="tab10", ax=ax
    )
    ax.set_title(g.getTeamName(team)[0])
    ax.set_xlim(-100, 100)  # optional, adjust to rink dimensions
    ax.set_ylim(-43, 43)
    ax.invert_yaxis()  # optional: matches typical rink orientation
    ax.set_aspect("equal", adjustable="box")

plt.tight_layout()
plt.show()
