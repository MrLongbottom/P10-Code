import argparse
import functools
import logging

import pyro
import pyro.distributions as dist
import torch
from pyro.infer import SVI, JitTraceEnum_ELBO, Trace_ELBO, Predictive
from pyro.optim import ClippedAdam
from pyro.params.param_store import ParamStoreDict
from torch import nn
from torch.distributions import constraints
from tqdm import tqdm

from preprocessing import preprocessing

logging.basicConfig(format='%(relativeCreated) 9d %(message)s', level=logging.INFO)


# This is a fully generative model of a batch of documents.
# data is a [num_words_per_doc, num_documents] shaped array of word ids
# (specifically it is not a histogram). We assume in this simple example
# that all documents have the same number of words.
def model(data=None, num_words_per_doc=None, args=None):
    # Globals.
    with pyro.plate("topics", args.num_topics):
        topic_weights = pyro.sample("topic_weights", dist.Gamma(1. / args.num_topics, 1.))
        topic_words = pyro.sample("topic_words",
                                  dist.Dirichlet(torch.ones(args.num_words) / args.num_words))
        # Changed here to from vector(with) to iteration to support varying number
        # of words (num_words_per_doc).
        # with pyro.plate("documents", args.num_docs) as ind:
    for doc in pyro.plate("documents", args.num_docs):
        doc_topics = pyro.sample("doc_topics_{}".format(doc), dist.Dirichlet(topic_weights))
        with pyro.plate("words_{}".format(doc), num_words_per_doc[doc]):
            word_topics = pyro.sample("word_topics_{}".format(doc), dist.Categorical(doc_topics))
            pyro.sample("doc_words_{}".format(doc), dist.Categorical(topic_words[word_topics]), obs=data[doc])
    return topic_weights, topic_words


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


def parametrized_guide(predictor, data, num_words_per_doc, args):
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
    pyro.module("predictor", predictor)
    for doc in pyro.plate("documents", args.num_docs, args.batch_size):
        # data = data[:, ind]
        # The neural network will operate on histograms rather than word
        # index vectors, so we'll convert the raw data to a histogram.
        counts = torch.zeros(args.num_words, 1)
        for i in data[doc]: counts[i] += 1
        #    .scatter_add(0, data[doc], torch.ones(data[doc].shape)))
        doc_topics = predictor(counts.transpose(0, 1))
        pyro.sample("doc_topics_{}".format(doc), dist.Delta(doc_topics, event_dim=1))
        # added this part since
        with pyro.plate("words_{}".format(doc), num_words_per_doc[doc]):
            word_topics = pyro.sample("word_topics_{}".format(doc), dist.Categorical(doc_topics))


def main(args):
    pyro.set_rng_seed(0)
    pyro.clear_param_store()
    pyro.enable_validation(__debug__)

    # Loading data
    corpora, documents = preprocessing()
    data = [torch.tensor(list(filter(lambda a: a != -1, corpora.doc2idx(doc))), dtype=torch.int64) for doc in documents]
    N = list(map(len, data))
    args.num_words = len(corpora)
    args.num_docs = len(data)

    # We'll train using SVI.
    logging.info('Training on {} documents'.format(args.num_docs))
    predictor = make_predictor(args)
    guide = functools.partial(parametrized_guide, predictor)
    Elbo = JitTraceEnum_ELBO if args.jit else Trace_ELBO
    elbo = Elbo(max_plate_nesting=2)
    optim = ClippedAdam({'lr': args.learning_rate})
    svi = SVI(model, guide, optim, elbo)

    logging.info('Step\tLoss')
    for step in tqdm(range(args.num_steps)):
        loss = svi.step(data, N, args=args)
        if step % 10 == 0:
            # logging.info('{: >5d}\t{}'.format(step, loss))
            logging.info(f"Loss: {loss}")
    loss = elbo.loss(model, guide, data, N, args=args)
    logging.info('final loss = {}'.format(loss))
    # save model
    torch.save({"model": predictor.state_dict(), "guide": guide}, "mymodel.pt")
    pyro.get_param_store().save("mymodelparams.pt")

    # load model
    # saved_model_dict = torch.load("mymodel.pt")
    # predictor.load_state_dict(saved_model_dict['model'])
    # guide = saved_model_dict['guide']
    # pyro.get_param_store().load("mymodelparams.pt")


if __name__ == '__main__':
    assert pyro.__version__.startswith('1.5.2')
    parser = argparse.ArgumentParser(description="Amortized Latent Dirichlet Allocation")
    parser.add_argument("-t", "--num-topics", default=30, type=int)
    parser.add_argument("-w", "--num-words", default=1024, type=int)
    parser.add_argument("-d", "--num-docs", default=1000, type=int)
    parser.add_argument("-n", "--num-steps", default=1000, type=int)
    parser.add_argument("-l", "--layer-sizes", default="100-100")
    parser.add_argument("-lr", "--learning-rate", default=0.01, type=float)
    parser.add_argument("-b", "--batch-size", default=32, type=int)
    parser.add_argument('--jit', action='store_true')
    args = parser.parse_args()
    main(args)
