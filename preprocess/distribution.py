import math

import numpy as np
import preprocess.preprocessing as pre
import matplotlib.pyplot as plt
from collections import Counter


def get_distribution(meta_data_name: str):
    doc2meta = pre.prepro_file_load(f'doc2{meta_data_name}', folder_name="full")
    id2meta = pre.prepro_file_load(f'id2{meta_data_name}', folder_name="full")
    distribution = {}
    if meta_data == "taxonomy":
        for k, v in doc2meta.items():
            for v2 in v:
                distribution[v2] = distribution.get(v2, 0) + 1
    else:
        for k, v in doc2meta.items():
            distribution[v] = distribution.get(v, 0) + 1
    return {id2meta[k]: v for k, v in distribution.items()}


def print_boxplot():
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
    plt.show()


def print_histogram(meta_data):
    # Cumulative Histogram
    plt.hist(sizes, color='blue', edgecolor='black', bins=50)
    plt.title(f'Histogram of Articles and {meta_data.capitalize() + "s"}')
    plt.xlabel('Number of Articles')
    plt.ylabel(f'Number of {meta_data.capitalize() + "s"}')
    plt.savefig(f"{meta_data}_hist_plot.pdf")
    plt.show()


def print_pie_chart(meta_data):
    # Pie Chart
    plt.pie(sizes, autopct='%1.1f', labels=names, startangle=90)
    plt.axis('equal')
    plt.show()


def print_latex_table(distribution: dict, columns: int = 4):
    sorted_dict = {k: v for k, v in sorted(distribution.items(), key=lambda item: item[1], reverse=True)}
    number_of_items = len(sorted_dict)
    items_per_column = math.ceil(number_of_items / columns)

    latex_table = []
    # generate table based on given distribution values and number of columns
    for index, item in enumerate(sorted_dict.items()):
        latexItem = f"{item[0]} & {item[1]}"
        if index < items_per_column:
            latex_table.append(latexItem)
        else:
            latex_table[index % items_per_column] += " & " + latexItem

    # add required end of row characters and print row
    for index, row in enumerate(latex_table):
        if latex_table[index].count('&') < (columns * 2) - 1:
            latex_table[index] += " & & \\\\"
        else:
            latex_table[index] += " \\\\"
        print(latex_table[index])


# This file is meant to analyse and visualize various aspects of the dataset
if __name__ == '__main__':
    meta_data = "taxonomy"
    distribution = get_distribution(meta_data)
    # distribution = {k: v for k, v in distribution.items() if v < 2000}


    sizes = list(Counter(distribution.values()))
    print(f"Mean: {np.mean(sizes)}, "
          f"Median: {np.median(sizes)}, "
          f"Min: {np.min(sizes)}, "
          f"Max: {np.max(sizes)}")
    names = [str(x) for x in sizes]
    fig1, ax1 = plt.subplots()
    print_histogram(meta_data)

    print_latex_table(distribution, columns=4)
