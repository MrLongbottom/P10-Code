import numpy as np
import preprocess.preprocessing as pre
import matplotlib.pyplot as plt

if __name__ == '__main__':
    doc2meta = pre.prepro_file_load('fulldoc2category')
    id2meta = pre.prepro_file_load('fullid2category')
    distribution = {}
    for k, v in doc2meta.items():
        distribution[v] = distribution.get(v, 0) + 1
    distribution = {id2meta[k]: v for k, v in distribution.items()}
    taxonomy = False
    if taxonomy:
        distribution2 = {}
        for k, v in distribution.items():
            taxo = k.split('/')
            for tax in taxo:
                distribution2[tax] = distribution2.get(tax, 0) + v
        distribution = distribution2
    sizes = list(distribution.values())
    print(np.mean(sizes))
    print(np.average(sizes))
    print(np.median(sizes))
    names = [str(x) for x in sizes]
    fig1, ax1 = plt.subplots()
    #n, bins, patches = ax1.hist(sizes, 1000, density=True, histtype='step',
    #                           cumulative=True, label='Empirical')
    ax1.boxplot(sizes)
    #ax1.pie(sizes, autopct='%1.1f', labels=names, startangle=90)
    #ax1.axis('equal')
    plt.show()

