import matplotlib.pyplot as plt

# Load and compute data
workers = []
times = []

with open('timings.txt') as f:
    for line in f:
        n, t = line.strip().split()
        workers.append(int(n))
        times.append(float(t))

baseline = times[0]
speedups = [baseline / t for t in times]

# Plot
plt.figure()
plt.plot(workers, speedups, marker='o')
plt.xlabel('Number of Workers')
plt.ylabel('Speedup')
plt.title('Parallel Speedup')
plt.grid(True)
plt.savefig('speedup_plot.png')

