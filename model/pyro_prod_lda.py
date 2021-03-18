import logging
import math

import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyro.infer import SVI, TraceMeanField_ELBO
from tqdm import trange
from wordcloud import WordCloud

from preprocess.preprocessing import prepro_file_load

logging.basicConfig(format='%(relativeCreated) 9d %(message)s', level=logging.INFO)


class Encoder(nn.Module):
    # Base class for the encoder net, used in the guide
    def __init__(self, vocab_size, num_topics, hidden, dropout):
        super().__init__()
        self.drop = nn.Dropout(dropout)  # to avoid component collapse
        # Setup the linear transformations
        self.fc1 = nn.Linear(vocab_size, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fcmu = nn.Linear(hidden, num_topics)
        self.fclv = nn.Linear(hidden, num_topics)
        self.bnmu = nn.BatchNorm1d(num_topics)  # to avoid component collapse
        self.bnlv = nn.BatchNorm1d(num_topics)  # to avoid component collapse

    def forward(self, inputs):
        # Compute the hidden units
        h = F.softplus(self.fc1(inputs))
        h = F.softplus(self.fc2(h))
        h = self.drop(h)
        # Mean vector μ and covariance Σ are the outputs
        theta_loc = self.bnmu(self.fcmu(h))
        theta_scale = self.bnlv(self.fclv(h))
        theta_scale = (0.5 * theta_scale).exp()  # Enforces positivity
        return theta_loc, theta_scale


class Decoder(nn.Module):
    # Base class for the decoder net, used in the model
    def __init__(self, vocab_size, num_topics, dropout):
        super().__init__()
        # Setup the linear transformation
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
        # Create the encoder and decoder networks
        self.encoder = Encoder(vocab_size, num_topics, hidden, dropout)
        self.decoder = Decoder(vocab_size, num_topics, dropout)

    def model(self, docs=None):
        # Register PyTorch module `decoder` with Pyro
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

    def guide(self, docs=None):
        # Register PyTorch module `encoder` with Pyro
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


def plot_word_cloud(b, ax, vocab, n):
    sorted_, indices = torch.sort(b, descending=True)
    df = pd.DataFrame(indices[:100].numpy(), columns=['index'])
    words = pd.merge(df, vocab[['index', 'word']],
                     how='left', on='index')['word'].values.tolist()
    sizes = (sorted_[:100] * 1000).int().numpy().tolist()
    freqs = {words[i]: sizes[i] for i in range(len(words))}
    wc = WordCloud(background_color="white", width=800, height=500)
    wc = wc.generate_from_frequencies(freqs)
    ax.set_title('Topic %d' % (n + 1))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis("off")


def main():
    assert pyro.__version__.startswith('1.6.0')
    # Enable smoke test to test functionality
    smoke_test = False

    logging.info(f"CUDA available: {torch.cuda.is_available()}")

    # Loading data
    logging.info("Loading data...")
    docs = prepro_file_load("doc_word_matrix").to_dense()
    id2word = prepro_file_load("id2word")

    # Put vocab into dataframe for exploration of data
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
    num_categories = 0
    num_topics = 30 if not smoke_test else 3
    batch_size = 32
    learning_rate = 1e-3
    num_epochs = 50 if not smoke_test else 1

    # Training
    pyro.clear_param_store()

    prodLDA = ProdLDA(
        vocab_size=docs.shape[1],
        num_topics=num_topics,
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
            loss = svi.step(batch_docs)
            running_loss += loss / batch_docs.size(0)

        # Save and log losses
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
        plot_file_name = "../ProdLDA-loss-2017_categories-" + str(num_categories) + \
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
            logging.info(f"Topic {n}: {words}")

        # Word cloud plotting
        # beta = prodLDA.beta()
        # fig, axs = plt.subplots(7, 3, figsize=(14, 24))
        # for n in range(beta.shape[0]):
        #     i, j = divmod(n, 3)
        #     plot_word_cloud(beta[n], axs[i, j], vocab, n)
        # axs[-1, -1].axis('off')
        # plt.savefig("../wordcloud.png")
        # plt.show()


if __name__ == '__main__':
    main()
