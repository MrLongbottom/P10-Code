import math

import pandas as pd
import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn
import torch.nn.functional as F
from pyro.infer import SVI, TraceMeanField_ELBO
from pyro.optim import ClippedAdam
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.optim import Adam
from tqdm import trange

from loading import load_document_file


class Encoder(nn.Module):
    # Base class for the inference net, used in the guide
    def __init__(self, vocab_size, num_topics, hidden, dropout):
        super().__init__()
        self.drop = nn.Dropout(dropout)  # to avoid component collapsing
        self.fc1 = nn.Linear(vocab_size, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fcmu = nn.Linear(hidden, num_topics)
        self.fclv = nn.Linear(hidden, num_topics)
        self.bnmu = nn.BatchNorm1d(num_topics)  # to avoid component collapsing
        self.bnlv = nn.BatchNorm1d(num_topics)  # to avoid component collapsing

    def forward(self, inputs):
        h = F.softplus(self.fc1(inputs))
        h = F.softplus(self.fc2(h))
        h = self.drop(h)
        # μ and Σ are the outputs
        theta_loc = self.bnmu(self.fcmu(h))
        theta_scale = self.bnlv(self.fclv(h))
        theta_scale = (0.5 * theta_scale).exp()  # Enforces positivity
        return theta_loc, theta_scale


class Decoder(nn.Module):
    # Base class for the recognition net, used in the model
    def __init__(self, vocab_size, num_topics, dropout):
        super().__init__()
        self.beta = nn.Linear(num_topics, vocab_size)
        self.bn = nn.BatchNorm1d(vocab_size)
        self.drop = nn.Dropout(dropout)

    def forward(self, inputs):
        inputs = self.drop(inputs)
        # the output is σ(βθ)
        return F.softmax(self.bn(self.beta(inputs)), dim=1)


class ProdLDA(nn.Module):
    def __init__(self, vocab_size, num_topics, hidden, dropout):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_topics = num_topics
        self.inference_net = Encoder(vocab_size, num_topics, hidden, dropout)
        self.recognition_net = Decoder(vocab_size, num_topics, dropout)

    def model(self, docs=None):
        pyro.module("recognition_net", self.recognition_net)
        with pyro.plate("documents", docs.shape[0]):
            # Dirichlet prior  𝑝(𝜃|𝛼) is replaced by a log-normal distribution
            theta_loc = docs.new_zeros((docs.shape[0], self.num_topics))
            theta_scale = docs.new_ones((docs.shape[0], self.num_topics))
            theta = pyro.sample(
                "theta", dist.LogNormal(theta_loc, theta_scale).to_event(1))
            theta = theta / theta.sum(-1, keepdim=True)

            # conditional distribution of 𝑤𝑛 is defined as
            # 𝑤𝑛|𝛽,𝜃 ~ Categorical(𝜎(𝛽𝜃))
            count_param = self.recognition_net(theta)
            pyro.sample(
                'obs',
                dist.Multinomial(docs.shape[1], count_param).to_event(1),
                obs=docs
            )

    def guide(self, docs=None):
        pyro.module("inference_net", self.inference_net)
        with pyro.plate("documents", docs.shape[0]):
            # Dirichlet prior  𝑝(𝜃|𝛼) is replaced by a log-normal distribution,
            # where μ and Σ are the inference net outputs
            theta_loc, theta_scale = self.inference_net(docs)
            theta = pyro.sample(
                "theta", dist.LogNormal(theta_loc, theta_scale).to_event(1))

    def beta(self):
        # beta matrix elements are the weights of the FC layer on the decoder
        return self.recognition_net.beta.weight.cpu().detach().T


if __name__ == '__main__':
    # Loading data
    documents, categories = load_document_file("data/2017_data.json")

    vectorizer = TfidfVectorizer(max_df=0.5, min_df=20)
    docs = torch.from_numpy(vectorizer.fit_transform(list(documents.values())[:1000]).toarray())

    vocab = pd.DataFrame(columns=['word', 'index'])
    vocab['word'] = vectorizer.get_feature_names()
    vocab['index'] = vocab.index

    print('Dictionary size: %d' % len(vocab))
    print('Corpus size: {}'.format(docs.shape))

    # setting global variables
    seed = 0
    torch.manual_seed(seed)
    pyro.set_rng_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    num_topics = 20
    docs = docs.float().to(device)
    batch_size = 32
    learning_rate = 1e-3
    num_epochs = 50

    # training
    pyro.clear_param_store()

    prodLDA = ProdLDA(
        vocab_size=docs.shape[1],
        num_topics=num_topics,
        hidden=100,
        dropout=0.2
    )
    prodLDA.to(device)

    optimizer = pyro.optim.Adam({"lr": learning_rate})
    svi = SVI(prodLDA.model, prodLDA.guide, optimizer, loss=TraceMeanField_ELBO())
    num_batches = int(math.ceil(docs.shape[0] / batch_size))

    bar = trange(num_epochs)
    for epoch in bar:
        running_loss = 0.0
        for i in range(num_batches):
            batch_docs = docs[i * batch_size:(i + 1) * batch_size, :]
            loss = svi.step(batch_docs)
            running_loss += loss / batch_docs.size(0)

        bar.set_postfix(epoch_loss='{:.2e}'.format(running_loss))

    beta = prodLDA.beta()
    for i in range(beta.shape[0]):
        sorted_, indices = torch.sort(beta[i], descending=True)
        df = pd.DataFrame(indices[:20].numpy(), columns=['index'])
        print(pd.merge(df, vocab[['index', 'word']], how='left', on='index')['word'].values)