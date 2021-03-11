import os
import logging

import pyro
import pyro.distributions as dist
import torch
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
import math
import torch.nn as nn
import torch.nn.functional as F
from pyro.infer import SVI, TraceMeanField_ELBO
from tqdm import trange
from wordcloud import WordCloud
import matplotlib.pyplot as plt

from preprocess.preprocessing import prepro_file_load

logging.basicConfig(format='%(relativeCreated) 9d %(message)s', level=logging.INFO)


class CategoryEncoder(nn.Module):
    # Base class for the encoder net, used in the guide
    def __init__(self, vocab_size, num_topics, num_categories, hidden, dropout):
        super().__init__()
        self.drop = nn.Dropout(dropout)  # to avoid component collapse
        self.fc1 = nn.Linear(vocab_size, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fcmu = nn.Linear(hidden, num_topics)
        self.fclv = nn.Linear(hidden, num_topics)
        self.bnmu = nn.BatchNorm1d(num_topics)  # to avoid component collapse
        self.bnlv = nn.BatchNorm1d(num_topics)  # to avoid component collapse

    def forward(self, inputs):
        h = F.softplus(self.fc1(inputs))
        h = F.softplus(self.fc2(h))
        h = self.drop(h)
        # μ and Σ are the outputs
        theta_loc = self.bnmu(self.fcmu(h))
        theta_scale = self.bnlv(self.fclv(h))
        theta_scale = (0.5 * theta_scale).exp()  # Enforces positivity
        return theta_loc, theta_scale


class CategoryDecoder(nn.Module):
    # Base class for the decoder net, used in the model
    def __init__(self, vocab_size, num_topics, num_categories, dropout):
        super().__init__()
        self.beta = nn.Linear(num_topics, vocab_size)
        self.bn = nn.BatchNorm1d(vocab_size)
        self.drop = nn.Dropout(dropout)

    def forward(self, inputs):
        inputs = self.drop(inputs)
        # the output is σ(βθ)
        return F.softmax(self.bn(self.beta(inputs)), dim=1)


class CategoryProdLDA(nn.Module):
    def __init__(self, vocab_size, num_topics, num_categories, hidden, dropout):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_topics = num_topics
        self.num_categories = num_categories
        self.encoder = CategoryEncoder(vocab_size, num_topics, num_categories, hidden, dropout)
        self.decoder = CategoryDecoder(vocab_size, num_topics, num_categories, dropout)

    def model(self, docs=None, doc_categories=None):
        pyro.module("decoder", self.decoder)
        with pyro.plate("documents", docs.shape[0]):
            # Dirichlet prior  𝑝(𝜃|𝛼) is replaced by a log-normal distribution
            theta_loc = docs.new_zeros((docs.shape[0], self.num_topics))
            theta_scale = docs.new_ones((docs.shape[0], self.num_topics))
            theta = pyro.sample(
                "theta", dist.LogNormal(theta_loc, theta_scale).to_event(1))
            theta = theta / theta.sum(-1, keepdim=True)

            # conditional distribution of 𝑤𝑛 is defined as
            # 𝑤𝑛|𝛽,𝜃 ~ Categorical(𝜎(𝛽𝜃))
            count_param = self.decoder(theta)
            pyro.sample(
                'obs',
                dist.Multinomial(docs.shape[1], count_param).to_event(1),
                obs=docs
            )

    def guide(self, docs=None, doc_categories=None):
        pyro.module("encoder", self.encoder)
        with pyro.plate("documents", docs.shape[0]):
            # Dirichlet prior  𝑝(𝜃|𝛼) is replaced by a log-normal distribution,
            # where μ and Σ are the encoder network outputs
            theta_loc, theta_scale = self.encoder(docs)
            theta = pyro.sample(
                "theta", dist.LogNormal(theta_loc, theta_scale).to_event(1))

    def beta(self):
        # beta matrix elements are the weights of the FC layer on the decoder
        return self.decoder.beta.weight.cpu().detach().T


def main():
    assert pyro.__version__.startswith('1.6.0')
    # Enable smoke test to test functionality
    smoke_test = True

    logging.info(f"CUDA available: {torch.cuda.is_available()}")

    logging.info("Loading data...")
    # News dataset for testing
    # news = fetch_20newsgroups(subset='all')
    # vectorizer = CountVectorizer(max_df=0.5, min_df=20)
    # docs = torch.from_numpy(vectorizer.fit_transform(news['data']).toarray())

    # Loading data
    docs = prepro_file_load("doc_word_matrix").to_dense()
    doc_categories = torch.t(torch.reshape(torch.Tensor(list(prepro_file_load("doc2category").values())), (1, -1)))
    id2word = prepro_file_load("id2word")
    id2cat = prepro_file_load("id2category")

    vocab = pd.DataFrame(columns=['index', 'word'])
    vocab['index'] = list(id2word.keys())
    vocab['word'] = list(id2word.values())

    logging.info(f"Dictionary size: {len(vocab)}")
    logging.info(f"Corpus size: {docs.shape}")

    # Setting global variables
    seed = 0
    torch.manual_seed(seed)
    pyro.set_rng_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    docs = docs.float()
    doc_categories = doc_categories.float()
    num_categories = len(id2cat)
    num_topics = num_categories * 2 if not smoke_test else 3
    batch_size = 32
    learning_rate = 1e-3
    num_epochs = 50 if not smoke_test else 1

    # Training
    pyro.clear_param_store()

    prodLDA = CategoryProdLDA(
        vocab_size=docs.shape[1],
        num_topics=num_topics,
        num_categories=num_categories,
        hidden=100 if not smoke_test else 10,
        dropout=0.2
    )
    prodLDA.to(device)

    optimizer = pyro.optim.Adam({"lr": learning_rate})
    svi = SVI(prodLDA.model, prodLDA.guide, optimizer, loss=TraceMeanField_ELBO())
    num_batches = int(math.ceil(docs.shape[0] / batch_size)) if not smoke_test else 1

    losses = []

    logging.info("Training...")
    bar = trange(num_epochs)
    for epoch in bar:
        running_loss = 0.0
        for i in range(num_batches):
            batch_docs = docs[i * batch_size:(i + 1) * batch_size, :].to(device)
            batch_cats = doc_categories[i * batch_size:(i + 1) * batch_size, :].to(device)
            loss = svi.step(batch_docs, batch_cats)
            running_loss += loss / batch_docs.size(0)

        losses.append(running_loss)
        bar.set_postfix(epoch_loss='{:.2e}'.format(running_loss))
        if epoch % 5 == 0:
            logging.info('{: >5d}\t{}'.format(epoch, '{:.2e}'.format(running_loss)))
    logging.info(f"Final loss: {'{:.2e}'.format(losses[-1])}/{losses[-1]}")

    if not smoke_test:
        # Plot loss over epochs
        plt.plot(losses)
        plt.title("ELBO")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plot_file_name = "../ProdCategoryLDA-loss-2017_categories-" + str(num_categories) + \
                         "_topics-" + str(num_topics) + \
                         "_batch-" + str(batch_size) + \
                         "_lr-" + str(learning_rate) + \
                         "_epochs-" + str(num_epochs) + \
                         ".png"
        plt.savefig(plot_file_name)
        plt.show()

        # Logging top 10 weighted words in topics
        beta = prodLDA.beta()
        for n in range(beta.shape[0]):
            sorted_, indices = torch.sort(beta[n], descending=True)
            df = pd.DataFrame(indices[:10].numpy(), columns=['index'])
            words = pd.merge(df, vocab[['index', 'word']], how='left', on='index')['word'].values.tolist()
            logging.info(f"Topic {n} sorted words: {words}")


if __name__ == '__main__':
    main()
