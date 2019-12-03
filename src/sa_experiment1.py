import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline
from scipy.ndimage.filters import gaussian_filter1d

if __name__ == "__main__":
    script_dir = os.path.dirname(__file__)
    dfs = []
    for i in range(0,10):
        rel_path = "../Data/Trace Files/LS1/Atlanta_LS1_600_{}.trace".format(i)
        abs_file_path = os.path.join(script_dir, rel_path)
        df = pd.read_csv(abs_file_path, index_col = 0, header=None)
        df = df.loc[~df.index.duplicated(keep='last')]
        df.columns = ["Run{}".format(i)]
        dfs.append(df)

    df = pd.concat(dfs, axis = 1, join='outer')
    df = df.fillna(method = "ffill", axis = 0)
    optimal = 2003763
    df = df/optimal - 1
    df = df.round(3)
    thresholds = np.asarray([0.0, .02, 0.04, 0.06, 0.08, 0.1])
    df2 = pd.DataFrame(index = df.index, columns = thresholds)
    df3 = pd.DataFrame(index = np.arange(df2.index[-1], df2.index[-1]+4, 0.01), columns = thresholds)
    df3 = df3.fillna(value = 1)
    for threshold in thresholds:
        for index in df2.index:
            df2.at[index,threshold] = len(np.where(np.asarray(df.loc[index]) <= threshold)[0])
    df2 = df2/10
    df2 = df2.append(df3, sort = False)
    for threshold in thresholds:
        df2[threshold] = gaussian_filter1d(df2[threshold].values.tolist(), sigma = 1)



    plt.figure()
    plt.gcf().subplots_adjust(bottom=0.2)
    plt.axis([0.3,0.5,-0.1,1.1])
    plt.plot(df2[0.0], color = 'b', linewidth = 1.0)
    plt.plot(df2[0.02], color = 'g', linewidth = 1.0)
    plt.plot(df2[0.04], color = 'r', linewidth = 1.0)
    plt.plot(df2[0.06], color = 'b', linewidth = 1.0, linestyle = '--')
    plt.plot(df2[0.08], color = 'g', linewidth = 1.0, linestyle = '--')
    plt.plot(df2[0.1], color = 'r', linewidth = 1.0, linestyle = '--')
    plt.legend([0.0, .02, 0.04, 0.06, 0.08, 0.1])
    plt.title("Qualified Runtime for Atlanta - Zoomed In", fontsize = 10)
    plt.ylabel("Probability", fontsize = 8)
    plt.xlabel("Time (s)", fontsize = 8)
    plt.savefig("qrtd_ls1_atlanta_in.png")

    plt.figure()
    plt.gcf().subplots_adjust(bottom=0.2)
    plt.axis([0.0,10.0,-0.1,1.1])
    plt.plot(df2[0.0], color = 'b', linewidth = 1.0)
    plt.plot(df2[0.02], color = 'g', linewidth = 1.0)
    plt.plot(df2[0.04], color = 'r', linewidth = 1.0)
    plt.plot(df2[0.06], color = 'b', linewidth = 1.0, linestyle = '--')
    plt.plot(df2[0.08], color = 'g', linewidth = 1.0, linestyle = '--')
    plt.plot(df2[0.1], color = 'r', linewidth = 1.0, linestyle = '--')
    plt.legend([0.0, .02, 0.04, 0.06, 0.08, 0.1])
    plt.title("Qualified Runtime for Atlanta - Zoomed Out", fontsize = 10)
    plt.ylabel("Probability", fontsize = 8)
    plt.xlabel("Time (s)", fontsize = 8)
    plt.savefig("qrtd_ls1_atlanta_out.png")
