import numpy as np
import matplotlib.pyplot as plt


rewards_op = {
    600: [6, 5, 5, 5, 5],
    500: [5, 5, 5, 5, 5],
    400: [5, 5, 5, 5, 5],
    300: [5, 5, 5, 5, 5],
    200: [6, 5, 5, 5, 5]
}

rewards_tw = {
    600: [9, 9, 9],
    500: [7, 8, 7],
    400: [7, 7, 6],
    300: [6, 7, 6],
    200: [7, 7, 6]
}

rewards_twst = {
    600: [10, 10, 10],
    500: [9, 9, 9],
    400: [9, 6, 8],
    300: [8, 8, 8],
    200: [7, 7, 6]
}


bar_width = 20

budget = [600, 500, 400, 300, 200]
op_count = [x - bar_width for x in budget]
op = [np.mean(rewards_op[x]) for x in rewards_op]
op_errs = [np.std(rewards_op[x]) for x in rewards_op]

tw_count = [x for x in budget]
tw = [np.mean(rewards_tw[x]) for x in rewards_tw]
tw_errs = [np.std(rewards_tw[x]) for x in rewards_tw]

twst_count = [x + bar_width for x in budget]
twst = [np.mean(rewards_twst[x]) for x in rewards_twst]
twst_errs = [np.std(rewards_twst[x]) for x in rewards_twst]

op_bars = plt.bar(op_count, op, bar_width, yerr=op_errs, label='Vanilla OP')
tw_bars = plt.bar(tw_count, tw, bar_width, yerr=tw_errs, label='OP w/ Time Windows')
twst_bars = plt.bar(twst_count, twst, bar_width, yerr=twst_errs, label='OPTW w/ Service Times')
plt.title('True reward for N=10 environment')
plt.xlabel('Allocated Budget')
plt.ylabel('Accumulated Reward')
plt.legend(handles=[op_bars, tw_bars, twst_bars])
plt.ylim(4, 11)
plt.show()
