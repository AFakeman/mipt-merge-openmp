import sys
from functools import reduce
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

def average(lst):
    return reduce(lambda x, y: x + y, lst) / float(len(lst))

header = input().split(' ')

numbers = np.array([[float(num) for num in line.split(' ')] for line in sys.stdin])

qsort_by_size = defaultdict(list)

for i in range(numbers.shape[0]):
    qsort_by_size[numbers[i][0]].append(numbers[i][3])

qsort = np.array([[average(time), size] for size, time in qsort_by_size.items()])

merge_by_thread = defaultdict(list)

for i in range(numbers.shape[0]):
    merge_by_thread[numbers[i][1]].append([numbers[i][2], numbers[i][0]])
 
plt.plot(qsort[:, 0], qsort[:, 1], label="qsort")

for threads in merge_by_thread.keys():
    runtimes = np.array(merge_by_thread[threads])
    plt.plot(runtimes[:, 0], runtimes[:, 1], label=str(threads))

plt.yscale('log')
plt.xscale('log')
plt.legend()
plt.savefig("stats.png")
