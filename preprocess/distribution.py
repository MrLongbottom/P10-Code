import preprocess.preprocessing as pre

if __name__ == '__main__':
    doc2meta = pre.prepro_file_load('doc2category')
    id2meta = pre.prepro_file_load('id2category')
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
    print('test')
