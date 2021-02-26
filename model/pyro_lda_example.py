import argparse
import functools
import logging

import torch
from torch import nn
from torch.distributions import constraints

import pyro
import pandas as pd
import matplotlib.pyplot as plt
import pyro.distributions as dist
from pyro.infer import SVI, JitTraceEnum_ELBO, TraceEnum_ELBO, Predictive
from pyro.optim import ClippedAdam
from tqdm import tqdm

from processing.preprocessing import preprocessing

logging.basicConfig(format='%(relativeCreated) 9d %(message)s', level=logging.INFO)


# This is a fully generative model of a batch of documents.
# data is a [num_words_per_doc, num_documents] shaped array of word ids
# (specifically it is not a histogram). We assume in this simple example
# that all documents have the same number of words.
def model(data=None, args=None):
    # Globals.
    with pyro.plate("topics", args.num_topics):
        topic_weights = pyro.sample("topic_weights", dist.Gamma(1. / args.num_topics, 1.))
        topic_words = pyro.sample("topic_words",
                                  dist.Dirichlet(torch.ones(args.num_words) / args.num_words))

    # Locals.
    with pyro.plate("documents", args.num_docs) as ind:
        if data is not None:
            # with pyro.util.ignore_jit_warnings():
            #     assert data.shape == (args.num_words_per_doc, args.num_docs)
            data = data[:, ind]
        doc_topics = pyro.sample("doc_topics", dist.Dirichlet(topic_weights))
        with pyro.plate("words", args.num_words_per_doc):
            # The word_topics variable is marginalized out during inference,
            # achieved by specifying infer={"enumerate": "parallel"} and using
            # TraceEnum_ELBO for inference. Thus we can ignore this variable in
            # the guide.
            word_topics = pyro.sample("word_topics", dist.Categorical(doc_topics),
                                      infer={"enumerate": "parallel"})
            data = pyro.sample("doc_words", dist.Categorical(topic_words[word_topics]),
                               obs=data)

    return topic_weights, topic_words, data


# We will use amortized inference of the local topic variables, achieved by a
# multi-layer perceptron. We'll wrap the guide in an nn.Module.
def make_predictor(args):
    layer_sizes = ([args.num_words] +
                   [int(s) for s in args.layer_sizes.split('-')] +
                   [args.num_topics])
    logging.info('Creating MLP with sizes {}'.format(layer_sizes))
    layers = []
    for in_size, out_size in zip(layer_sizes, layer_sizes[1:]):
        layer = nn.Linear(in_size, out_size)
        layer.weight.data.normal_(0, 0.001)
        layer.bias.data.normal_(0, 0.001)
        layers.append(layer)
        layers.append(nn.Sigmoid())
    layers.append(nn.Softmax(dim=-1))
    return nn.Sequential(*layers)


def parametrized_guide(predictor, data, args):
    # Use a conjugate guide for global variables.
    topic_weights_posterior = pyro.param(
        "topic_weights_posterior",
        lambda: torch.ones(args.num_topics),
        constraint=constraints.positive)
    topic_words_posterior = pyro.param(
        "topic_words_posterior",
        lambda: torch.ones(args.num_topics, args.num_words),
        constraint=constraints.greater_than(0.5))
    with pyro.plate("topics", args.num_topics):
        pyro.sample("topic_weights", dist.Gamma(topic_weights_posterior, 1.))
        pyro.sample("topic_words", dist.Dirichlet(topic_words_posterior))

    # Use an amortized guide for local variables.
    if data is None:
        args.batch_size = None
    pyro.module("predictor", predictor)
    with pyro.plate("documents", args.num_docs, args.batch_size) as ind:
        data = data[:, ind]
        # The neural network will operate on histograms rather than word
        # index vectors, so we'll convert the raw data to a histogram.
        counts = (torch.zeros(args.num_words, ind.size(0))
                  .scatter_add(0, data, torch.ones(data.shape)))
        doc_topics = predictor(counts.transpose(0, 1))
        pyro.sample("doc_topics", dist.Delta(doc_topics, event_dim=1))


def print_top_topic_words(data, vocab, args):
    predictive = Predictive(model, num_samples=100, return_sites=['topic_words'])

    i = 0
    batch_size = 32
    idx = torch.arange(i * batch_size,
                       min((i + 1) * batch_size, len(data)))
    batch_docs = data[idx, :].cpu()
    samples = predictive(batch_docs, args)

    beta = samples['topic_words'].mean(dim=0).squeeze().detach()

    for i in range(beta.shape[0]):
        sorted_, indices = torch.sort(beta[i], descending=True)
        df = pd.DataFrame(indices[:20].numpy(), columns=['index'])
        print(f"Topic {i}: {[vocab[x] for x in list(df.values[:,0])]}\n")


def main(args):
    logging.info('Generating data')
    pyro.set_rng_seed(0)
    pyro.clear_param_store()

    # We can generate synthetic data directly by calling the model.
    # true_topic_weights, true_topic_words, pre_data = model(args=args)

    corpora, documents = preprocessing()
    data = [torch.tensor(list(filter(lambda a: a != -1, corpora.doc2idx(doc))), dtype=torch.int64) for doc in documents]
    big_len = max([len(x) for x in data])
    data = [torch.cat((x, torch.ones(big_len - len(x), dtype=torch.int64))) for x in data]
    args.num_words_per_doc = big_len
    args.num_words = len(corpora)
    args.num_docs = len(data)
    data = torch.stack(data).T

    # We'll train using SVI.
    logging.info('-' * 40)
    logging.info('Training on {} documents'.format(args.num_docs))
    predictor = make_predictor(args)
    guide = functools.partial(parametrized_guide, predictor)
    Elbo = JitTraceEnum_ELBO if args.jit else TraceEnum_ELBO
    elbo = Elbo(max_plate_nesting=2)
    optim = ClippedAdam({'lr': args.learning_rate})
    svi = SVI(model, guide, optim, elbo)
    logging.info('Step\tLoss')
    losses = []
    for step in tqdm(range(args.num_steps)):
        loss = svi.step(data, args=args)
        losses.append(loss)
        if step % 10 == 0:
            logging.info('{: >5d}\t{}'.format(step, loss))
    plt.plot(losses)
    plt.title("ELBO")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.show()
    print_top_topic_words(data, corpora, args)


if __name__ == '__main__':
    assert pyro.__version__.startswith('1.5.2')
    parser = argparse.ArgumentParser(description="Amortized Latent Dirichlet Allocation")
    parser.add_argument("-t", "--num-topics", default=20, type=int)
    parser.add_argument("-w", "--num-words", default=1024, type=int)
    parser.add_argument("-d", "--num-docs", default=1000, type=int)
    parser.add_argument("-wd", "--num-words-per-doc", default=64, type=int)
    parser.add_argument("-n", "--num-steps", default=1000, type=int)
    parser.add_argument("-l", "--layer-sizes", default="100-100")
    parser.add_argument("-lr", "--learning-rate", default=0.01, type=float)
    parser.add_argument("-b", "--batch-size", default=1, type=int)
    parser.add_argument('--jit', action='store_true')
    args = parser.parse_args()
    main(args)
