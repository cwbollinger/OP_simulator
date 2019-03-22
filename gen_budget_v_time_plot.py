import numpy as np
import matplotlib.pyplot as plt


times_op = {
    600: [0.38562599000033515, 0.4195561779997661, 0.3660243560007075, 0.40224052899975504, 0.5112269530000049],
    500: [0.41710121799951594, 0.36924671700035105, 0.3715244370014261, 0.3815944439993473, 0.5184771259991976],
    400: [0.373526180999761, 0.37486540300051274, 0.42246421100026055, 0.3827827490003983, 0.5128667120006867],
    300: [0.37570247599978757, 0.4183859269996901, 0.36896512400016945, 0.3806347450008616, 0.5653778859996237],
    200: [0.371711243999016, 0.37008766399958404, 0.36908665299961285, 0.43720295400089526, 0.5340144500005408]
}

times_tw = {
    600: [0.4550141800009442, 0.4497910600002797, 0.4481457279998722],
    500: [0.7874602470001264, 6.587966625000263, 1.7509006170002976],
    400: [0.4540632159987581, 0.6823536369993235, 2.117766388000746],
    300: [0.49240956399989955, 0.4481401770008233, 0.46574077799959923],
    200: [0.5015908450004645, 0.5000979970009212, 0.45190504799938935]
}

times_twst = {
    600: [0.5233315700006642, 0.4474803789998987, 0.45274709899968],
    500: [1.1704666759997053, 4.739196114000151, 2.225776863999272],
    400: [0.4517718029983371, 0.750440196999989, 2.0693786930005444],
    300: [0.49731554000027245, 0.478945727001701, 0.6084395360012422],
    200: [0.5056049089998851, 0.5053363620008895, 0.45568421600000875]
}

bar_width = 20

budget = [600, 500, 400, 300, 200]
op_count = [x - bar_width for x in budget]
op = [np.mean(times_op[x]) for x in times_op]
op_errs = [np.std(times_op[x]) for x in times_op]

tw_count = [x for x in budget]
tw = [np.mean(times_tw[x]) for x in times_tw]
tw_errs = [np.std(times_tw[x]) for x in times_tw]

twst_count = [x + bar_width for x in budget]
twst = [np.mean(times_twst[x]) for x in times_twst]
twst_errs = [np.std(times_twst[x]) for x in times_twst]

op_bars = plt.bar(op_count, op, bar_width, yerr=op_errs, label='Vanilla OP')
tw_bars = plt.bar(tw_count, tw, bar_width, yerr=tw_errs, label='OP w/ Time Windows')
twst_bars = plt.bar(twst_count, twst, bar_width, yerr=twst_errs, label='OPTW w/ Service Times')
plt.title('Time to Solve Budgeted Plans')
plt.xlabel('Allocated Budget')
plt.ylabel('Solve time (s)')
plt.legend(handles=[op_bars, tw_bars, twst_bars])
plt.ylim(0, 4)
plt.show()
