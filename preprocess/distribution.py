import preprocess.preprocessing as pre

if __name__ == '__main__':
    doc2meta = pre.prepro_file_load('doc2author')
    id2meta = pre.prepro_file_load('id2author')
    distribution = {}
    for k, v in doc2meta.items():
        distribution[v] = distribution.get(v, 0) + 1
    distribution = {id2meta[k]: v for k, v in distribution.items()}
    print('test')
