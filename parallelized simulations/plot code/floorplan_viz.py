import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv("floorplan_results.csv")


fig, axes = plt.subplots(1, 2, figsize=(12, 5))

x = np.arange(1,1+len(df["mean_temp"]))
y = df["mean_temp"]
e = df["std_temp"]

axes[0].errorbar(x, y,yerr=e,  marker='o',capsize=5, color='orange', ecolor='orange')
axes[0].set_xticks(x)  # Ensure all ticks are shown
axes[0].set_title("Mean temperature with standard deviation") 
axes[0].set_ylabel("Mean temperature")
axes[0].set_xlabel("Floorplan index")


axes[1].plot(x, df["pct_above_18"], marker='o', color='red')
axes[1].set_xticks(x)  # Ensure all ticks are shown
axes[1].plot(x, df["pct_below_15"], marker='o', color='blue')
axes[1].set_title("Percentage above 18 and below 15")
axes[1].set_ylabel("Percentage")
axes[1].set_xlabel("Floorplan index")
axes[1].legend(["Above 18","Below 15"])

plt.tight_layout()
fig.savefig("floorplan_viz.png")
plt.show()

