import numpy as np
import matplotlib.pyplot as plt

time_small_map = {
    3: [0.000267097, 0.00028116, 0.000269773, 0.000285456, 0.00026707900000000004],
    4: [0.0006196029999999999, 0.000555563, 0.000593323, 0.000596124, 0.000650057],
    5: [0.000806675, 0.000815554, 0.000883308, 0.0008018519999999999, 0.0008147619999999999],
    6: [0.001347506, 0.001402417, 0.001477262, 0.001542084, 0.0012305089999999999],
    7: [0.001369637, 0.001323809, 0.0017221019999999999],
    8: [0.002000764, 0.001761007, 0.0017765419999999999],
    9: [0.002241492, 0.0021741959999999998, 0.002346946],
    10: [0.002991846]
}

time_large_map = {
    3: [0.00040948799999999995, 0.000282081, 0.000319145, 0.000386333, 0.000279777],
    4: [0.000639022, 0.000581287, 0.000590823, 0.00065749, 0.000632444],
    5: [0.000896848, 0.000976476, 0.0009090979999999999, 0.0015694910000000001, 0.001085664],
    6: [0.0019011759999999999, 0.001283466, 0.001222118, 0.001271272, 0.0012450400000000002],
    7: [0.0013343690000000002, 0.001347985, 0.00138202, 0.0016485669999999999],
    8: [0.001905686, 0.001735099, 0.001754326, 0.001920264],
    9: [0.002419488, 0.002356285, 0.002418689, 0.004169394],
    10: [0.002752713, 0.002751876, 0.0029286959999999997]
}


node_count = [3, 4, 5, 6, 7, 8, 9, 10]
small_count = [x - 0.2 for x in node_count]
small_times = [np.mean(time_small_map[x]) for x in node_count]
small_errs = [np.std(time_small_map[x]) for x in node_count]
large_count = [x + 0.2 for x in node_count]
large_times = [np.mean(time_large_map[x]) for x in node_count]
large_errs = [np.std(time_large_map[x]) for x in node_count]
large_count = [x + 0.2 for x in node_count]
small_bars = plt.bar(small_count, small_times, 0.4, yerr=small_errs, label='20x20 map')
print(len(small_bars))
large_bars = plt.bar(large_count, large_times, 0.4, yerr=large_errs, label='50x50 map')
plt.title('Solution times for varying POI count')
plt.xlabel('Number of POIs')
plt.ylabel('Solve time (s)')
plt.legend(handles=[small_bars, large_bars])
plt.show()
