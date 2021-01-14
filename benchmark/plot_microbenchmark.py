import matplotlib.pyplot as plt
import numpy as np

mem = np.array([73.7, 75.56, 80.22, 80.61, 79.59, 82.33, 83.78])
comp = np.array([48.88, 50.36, 53.82, 53.60, 53.07, 54.65, 55.86])

mem2 = np.array([81.51, 81.85, 81.02, 80.08, 82.47, 82.33, 82.97])
comp2 = np.array([53.49, 54.08, 53.88, 52.78, 53.64, 54.54, 55.24])

mem3 = np.array([81.35, 82.67, 82.50, 82.90, 83.39, 83.63, 85.54])
comp3 = np.array([55.00, 54.56, 54.79, 53.67, 54.61, 55.67, 56.27])

plt.plot(mem, comp, 'o', markersize=14, label="blocksize = (32,1,1)")
plt.plot(mem2, comp2, 'o', markersize=14, label="blocksize = (32,8,1)")
plt.plot(mem3, comp3, 'o', markersize=14, label="blocksize = (64,1,1)")

for i in range(len(mem)):
    plt.annotate(str(i+1),  xy=(mem[i], comp[i]), color='white',
                fontsize="large", weight='heavy',
                horizontalalignment='center',
                verticalalignment='center')
    plt.annotate(str(i+1),  xy=(mem2[i], comp2[i]), color='white',
                fontsize="large", weight='heavy',
                horizontalalignment='center',
                verticalalignment='center')
    plt.annotate(str(i+1),  xy=(mem3[i], comp3[i]), color='white',
                fontsize="large", weight='heavy',
                horizontalalignment='center',
                verticalalignment='center')

plt.ylabel("Computation throughput (%)")
plt.xlabel("Memory throughput (%)")
plt.legend()
plt.savefig("Microbenchmark.pdf")
plt.show()
