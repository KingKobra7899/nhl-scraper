from pandas import DataFrame
import nhl_scraper.games as g
import seaborn as sns
import matplotlib.pyplot as plt

df: DataFrame = g.getPbpData("2025020109")["shots"]
df.to_csv("test.csv")
ax = sns.FacetGrid(df, col="teamId")
ax.map_dataframe(sns.scatterplot, x="xCoord", y="yCoord", hue="angle")
plt.show()
