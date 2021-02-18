import argparse
import functools
import logging

import torch
from torch import nn
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, JitTraceEnum_ELBO, TraceEnum_ELBO
from pyro.optim import ClippedAdam

import numpy as np
from tqdm import tqdm

logging.basicConfig(format='%(relativeCreated) 9d %(message)s', level=logging.INFO)


# This is a fully generative model of a batch of documents.
# data is a [num_words_per_doc, num_documents] shaped array of word ids
# (specifically it is not a histogram). We assume in this simple example
# that all documents have the same number of words.
def model(doc_word_data=None, category_data=None, args=None, batch_size=None):
    # Globals.
    with pyro.plate("topics", args.num_topics):
        # topic_weights does not seem to come from the usual LDA plate notation, but seems to give an indication of
        # the importance of topics. It might be from the amortized LDA paper.
        topic_weights = pyro.sample("topic_weights", dist.Gamma(1. / args.num_topics, 1.))
        topic_words = pyro.sample("topic_words",
                                  dist.Dirichlet(torch.ones(args.num_words) / args.num_words))

    with pyro.plate("categories", args.num_categories):
        category_weights = pyro.sample("category_weights", dist.Gamma(1. / args.num_categories, 1.))
        category_topics = pyro.sample("category_topics", dist.Dirichlet(topic_weights))

    # Locals.
    with pyro.plate("documents", args.num_docs) as ind:
        if doc_word_data is not None:
            with pyro.util.ignore_jit_warnings():
                assert doc_word_data.shape == (args.num_words_per_doc, args.num_docs)  # Forces the 64x1000 shape
            doc_word_data = doc_word_data[:, ind]

        if category_data is not None:
            category_data = category_data[ind]

        category_data = pyro.sample("doc_categories", dist.Categorical(category_weights), obs=category_data)
        # doc_topics = pyro.sample("doc_topics", dist.Dirichlet(topic_weights))
        with pyro.plate("words", args.num_words_per_doc):
            # The word_topics variable is marginalized out during inference,
            # achieved by specifying infer={"enumerate": "parallel"} and using
            # TraceEnum_ELBO for inference. Thus we can ignore this variable in
            # the guide.
            word_topics = pyro.sample("word_topics", dist.Categorical(category_topics[category_data]),
                                      infer={"enumerate": "parallel"})
            doc_word_data = pyro.sample("doc_words", dist.Categorical(topic_words[word_topics]),
                                        obs=doc_word_data)

    return topic_weights, topic_words, doc_word_data, category_data


# We will use amortized inference of the local topic variables, achieved by a
# multi-layer perceptron. We'll wrap the guide in an nn.Module.
def make_predictor(args):
    layer_sizes = ([args.num_words] +
                   [int(s) for s in args.layer_sizes.split('-')] +
                   [args.num_topics])
    logging.info('Creating MLP with sizes {}'.format(layer_sizes))  # Default layers: 1024 -> 100 -> 100 -> 8
    layers = []
    for in_size, out_size in zip(layer_sizes, layer_sizes[1:]):
        layer = nn.Linear(in_size, out_size)
        layer.weight.data.normal_(0, 0.001)
        layer.bias.data.normal_(0, 0.001)
        layers.append(layer)
        layers.append(nn.Sigmoid())
    layers.append(nn.Softmax(dim=-1))
    return nn.Sequential(*layers)


def parametrized_guide(predictor, doc_word_data, category_data, args, batch_size=None):
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

    category_weights_posterior = pyro.param(
        "category_weights_posterior",
        lambda: torch.ones(args.num_categories),
        constraint=constraints.positive)
    category_topics_posterior = pyro.param(
        "category_topics_posterior",
        lambda: torch.ones(args.num_categories, args.num_topics),
        constraint=constraints.greater_than(0.5))
    with pyro.plate("categories", args.num_categories):
        pyro.sample("category_weights", dist.Gamma(category_weights_posterior, 1.))
        pyro.sample("category_topics", dist.Dirichlet(category_topics_posterior))

    doc_category_posterior = pyro.param(
        "doc_category_posterior",
        lambda: torch.ones(args.num_topics),
        constraint=constraints.positive)
    with pyro.plate("documents", args.num_docs, batch_size) as ind:
        pyro.sample("doc_categories", dist.Categorical(doc_category_posterior))

    # Use an amortized guide for local variables.
    # pyro.module("predictor", predictor)
    # with pyro.plate("documents", args.num_docs, batch_size) as ind:
    #     doc_word_data = doc_word_data[:, ind]
    #     category_data = category_data[ind]
    #     # The neural network will operate on histograms rather than word
    #     # index vectors, so we'll convert the raw data to a histogram.
    #     counts = (torch.zeros(ind.size(0))
    #               .scatter_add(0, category_data, torch.ones(category_data.shape)))
    #     doc_categories = predictor(counts.transpose(0, 1))
    #     pyro.sample("doc_categories", dist.Delta(doc_categories, event_dim=1))

        # original:
        # data = doc_word_data[:, ind]
        # # The neural network will operate on histograms rather than word
        # # index vectors, so we'll convert the raw data to a histogram.
        # counts = (torch.zeros(args.num_words, ind.size(0))
        #           .scatter_add(0, data, torch.ones(data.shape)))
        # doc_topics = predictor(counts.transpose(0, 1))
        # pyro.sample("doc_topics", dist.Delta(doc_topics, event_dim=1))


def main(args):
    logging.info(f"CUDA enabled: {torch.cuda.is_available()}")
    logging.info('Generating data')
    pyro.set_rng_seed(0)
    pyro.clear_param_store()
    pyro.enable_validation(__debug__)

    # We can generate synthetic data directly by calling the model.
    true_topic_weights, true_topic_words, doc_word_data, category_data = model(args=args)

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
    for step in tqdm(range(args.num_steps)):
        loss = svi.step(doc_word_data=doc_word_data, category_data=category_data, args=args, batch_size=args.batch_size)
        if step % 10 == 0:
            logging.info('{: >5d}\t{}'.format(step, loss))
    loss = elbo.loss(model, guide, doc_word_data=doc_word_data, category_data=category_data, args=args)
    logging.info('final loss = {}'.format(loss))
    print("debug string")


if __name__ == '__main__':
    assert pyro.__version__.startswith('1.5.2')
    parser = argparse.ArgumentParser(description="Amortized Latent Dirichlet Allocation")
    parser.add_argument("-c", "--num-categories", default=4, type=int)
    parser.add_argument("-t", "--num-topics", default=8, type=int)
    parser.add_argument("-w", "--num-words", default=1024, type=int)
    parser.add_argument("-d", "--num-docs", default=1000, type=int)
    parser.add_argument("-wd", "--num-words-per-doc", default=64, type=int)
    parser.add_argument("-n", "--num-steps", default=1000, type=int)
    parser.add_argument("-l", "--layer-sizes", default="100-100")
    parser.add_argument("-lr", "--learning-rate", default=0.01, type=float)
    parser.add_argument("-b", "--batch-size", default=32, type=int)
    parser.add_argument('--jit', action='store_true')
    args = parser.parse_args()
    main(args)
