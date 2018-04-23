import re
import os

from matplotlib import pyplot

figs_path = "../figures"


def get_next_fig_filename():
    regex = re.compile(r'fig([0-9]+).png')
    figs = [f for f in os.listdir(figs_path) if os.path.isfile(os.path.join(figs_path, f)) and regex.search(f)]
    extension = 0
    for fig in figs:
        capture_groups = re.findall(r'fig([0-9]+).png', fig, flags=0)
        if int(capture_groups[0]) > extension:
            extension = int(capture_groups[0])
    return figs_path + "/fig" + str(extension) + ".png"


def graph_df(df, values, save_fig=False):
    num_cols = len(df.columns.values)
    # specify columns to plot
    groups = [i for i in range(0, num_cols)]
    i = 1
    # plot each column W x H
    pyplot.figure(figsize=(45, 28))
    for group in groups:
        pyplot.subplot(len(groups), 1, i)
        pyplot.plot(values[:, group])
        pyplot.title(df.columns[group], y=0.5, loc='right')
        i += 1
    if save_fig:
        pyplot.savefig(get_next_fig_filename())
