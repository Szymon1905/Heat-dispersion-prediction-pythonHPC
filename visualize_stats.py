import subprocess
import pandas as pd
from pyarrow import csv
import sys
import os

# Load the stats
def pyarrow_load(file):
    table = csv.read_csv(file)
    table1 = table.to_pandas()
    return table1


# Run the simulate.py script and get the stats.csv file
if not os.path.exists("stats.csv"):
    subprocess.run([sys.executable, "simulate.py"])

# subprocess.run([sys.executable, "simulate.py"])
stats = pyarrow_load("stats.csv")


import matplotlib.pyplot as plt
# visualize the stats in a table
print(stats)

fig = plt.figure(figsize=(8, 5))

# Plot the mean_temp for each index and add the std_temp
ax1 = fig.add_subplot(1, 2, 1)
ax1.errorbar(stats.index, stats['mean_temp'], yerr=stats['std_temp'], fmt='-o', ecolor='red', capsize=5)
ax1.set_xlabel('Index')
ax1.set_ylabel('Temperature')
ax1.set_title('Mean Temperature with Standard Deviation')

# Plot the pct_above_18
ax2 = fig.add_subplot(2, 2, 2)
ax2.plot(stats.index, stats['pct_above_18'], '-o')
ax2.set_xlabel('Index')
ax2.set_ylabel('Percentage')
ax2.set_title('Percentage of Building Above 18')

# Plot the pct_below_15
ax3 = fig.add_subplot(2, 2, 4)
ax3.plot(stats.index, stats['pct_below_15'], '-o')
ax3.set_xlabel('Index')
ax3.set_ylabel('Percentage')
ax3.set_title('Percentage of Building Below 15')

plt.tight_layout()
plt.show()

# Save the figure
fig.savefig("stats.png")