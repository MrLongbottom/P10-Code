import loading
import gensim
from tqdm import tqdm


def preprocessing(printouts=False, save=False):
    # load data file
    if printouts:
        print("Loading dataset")
    data = loading.load_document_file("data/2017_data.json")

    # tokenize (document token generators)
    if printouts:
        print("Tokenization")
    documents = [gensim.utils.tokenize(x[0], lowercase=True, deacc=True, encoding='utf-8') for x in tqdm(data.values())]

    # normalize (currently doesn't work)
    # documents = [[y for y in gensim.utils.tokenize(x[0], lowercase=True, deacc=True, encoding='utf-8')] for x in data.values()]
    # norm_documents = gensim.models.normmodel.NormModel(corpus=[corpora.doc2bow(x) for x in documents], norm='l1')
    # norm_corpora = gensim.corpora.Dictionary.from_corpus(norm_documents)

    # Build Corpora object
    if printouts:
        print("Building Corpora")
    corpora = gensim.corpora.Dictionary(documents)

    # Extreme filter
    if printouts:
        print("Filtering out extreme words")
    corpora.filter_extremes(no_below=10, no_above=0.1)

    # clean and save corpora
    corpora.compactify()
    if save:
        if printouts:
            print("Saving Corpora")
        corpora.save("generated_files/corpora")

    if printouts:
        print('Preprocessing Finished.')
    return corpora


if __name__ == '__main__':
    corpora = preprocessing(printouts=True)
