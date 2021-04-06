import numpy as np
import preprocess.preprocessing as pre
import matplotlib.pyplot as plt

# This file is meant to analyse and visualize various aspects of the dataset
if __name__ == '__main__':
    doc2meta = pre.prepro_file_load('doc2category', folder_name='full')
    id2meta = pre.prepro_file_load('id2category', folder_name='full')
    distribution = {}
    taxonomy = False
    if taxonomy:
        for v in doc2meta.values():
            for tax in v:
                distribution[id2meta[tax]] = distribution.get(id2meta[tax], 0) + 1
    else:
        for k, v in doc2meta.items():
            distribution[v] = distribution.get(v, 0) + 1
        distribution = {id2meta[k]: v for k, v in distribution.items()}

    other = [(k,v) for k, v in distribution.items() if v <= 139]
    distribution_cut = {k: v for k, v in distribution.items() if v > 139}
    distribution_cut['other'] = int(np.sum([v for k, v in other]))
    sizes = list(distribution.values())
    sizes2 = list(distribution_cut.values())
    print("Mean: {}, Median: {}, Min: {}, Max: {}".format(np.mean(sizes), np.median(sizes), np.min(sizes), np.max(sizes)))
    print("Mean: {}, Median: {}, Min: {}, Max: {}".format(np.mean(sizes2), np.median(sizes2), np.min(sizes2), np.max(sizes2)))
    fig1, ax1 = plt.subplots()
    # Cumulative Histogram
    #n, bins, patches = ax1.hist(sizes2, 100, density=False, histtype='step',
    #                           cumulative=False, label='Empirical')

    # Boxplot
    box = ax1.boxplot(sizes, sym='')
    for line in box['medians']:
        x, y = line.get_xydata()[0]
        ax1.text(x - 0.06, y, '%.0f' % y, horizontalalignment='center', verticalalignment='center')
    for line in box['boxes']:
        x, y = line.get_xydata()[0]
        ax1.text(x - 0.06, y, '%.0f' % y, horizontalalignment='center', verticalalignment='center')
        x, y = line.get_xydata()[3]
        ax1.text(x - 0.06, y, '%.0f' % y, horizontalalignment='center', verticalalignment='center')
    for line in box['caps']:
        x, y = line.get_xydata()[0]
        ax1.text(x - 0.06, y, '%.0f' % y, horizontalalignment='center', verticalalignment='center')
    for line in box['fliers']:
        x, y = line.get_xydata()[0]
        ax1.text(x - 0.06, y, '%.0f' % y, horizontalalignment='center', verticalalignment='center')

    # Pie Chart
    #names = [str(x) for x in sizes]
    #ax1.pie(sizes, autopct='%1.1f', labels=names, startangle=90)
    #ax1.axis('equal')
    plt.show()
    print()

