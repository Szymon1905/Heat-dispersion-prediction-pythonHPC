import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys

task = sys.argv[1]

f = open(task, "r")

lines = f.readlines()
f.close()
x = []
y = []
for i in lines:
    n,t = i.split()
    x.append(int(n))
    y.append(float(t))

x = np.array(x)
y = np.array(y)
y = y[0]/y
plt.plot(x,y, marker='o', linestyle='--')
plt.grid(which='both', linestyle='--', linewidth=0.7, alpha=0.7)
plt.yticks(np.arange(1, max(y)+1))
plt.xlabel("Number of processors")
plt.ylabel("Speedup")
plt.title(f"{task[:-4]}: Speedup vs Number of processors")
plt.savefig(f"{task[:-4]}.png")

