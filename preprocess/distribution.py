import numpy as np
import preprocess.preprocessing as pre
import matplotlib.pyplot as plt

# This file is meant to analyse and visualize various aspects of the dataset
if __name__ == '__main__':
    doc2meta = pre.prepro_file_load('doc2category', folder_name='full')
    id2meta = pre.prepro_file_load('id2category', folder_name='full')
    distribution = {}
    for k, v in doc2meta.items():
        distribution[v] = distribution.get(v, 0) + 1
    distribution = {id2meta[k]: v for k, v in distribution.items()}
    #distribution = {k: v for k, v in distribution.items() if v > 140}
    taxonomy = False
    if taxonomy:
        distribution2 = {}
        for k, v in distribution.items():
            taxo = k.split('/')
            for tax in taxo:
                distribution2[tax] = distribution2.get(tax, 0) + v
        distribution = distribution2
    sizes = list(distribution.values())
    print("Mean: {}, Median: {}, Min: {}, Max: {}".format(np.mean(sizes), np.median(sizes), np.min(sizes), np.max(sizes)))
    names = [str(x) for x in sizes]
    fig1, ax1 = plt.subplots()
    # Cumulative Histogram
    #n, bins, patches = ax1.hist(sizes, 1000, density=True, histtype='step',
    #                           cumulative=True, label='Empirical')
    # Boxplot
    box = ax1.boxplot(sizes)
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
    #ax1.pie(sizes, autopct='%1.1f', labels=names, startangle=90)
    #ax1.axis('equal')
    plt.show()

