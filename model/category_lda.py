import argparse
import logging
import functools
import re

import torch
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.infer import SVI, JitTraceEnum_ELBO, TraceEnum_ELBO, Trace_ELBO
from pyro.optim import ClippedAdam

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from preprocess.preprocessing import preprocessing, prepro_file_load

logging.basicConfig(format='%(relativeCreated) 9d %(message)s', level=logging.INFO)


# This is a fully generative model of a batch of documents.
# Documents do not need to include the same number of words.
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
        # TODO category weights might not be necessary in our model
        category_topics = pyro.sample("category_topics", dist.Dirichlet(topic_weights))

    doc_category_list = []
    doc_word_list = []

    # Locals.
    for index, doc in enumerate(pyro.plate("documents", args.num_docs)):
        if doc_word_data is not None:
            cur_doc_word_data = doc_word_data[doc]
        else:
            cur_doc_word_data = None

        if category_data is not None:
            cur_category_data = category_data[doc]
        else:
            cur_category_data = None

        doc_category_list.append(
            pyro.sample("doc_categories_{}".format(doc), dist.Categorical(category_weights), obs=cur_category_data))

        with pyro.plate("words_{}".format(doc), args.num_words_per_doc[doc]):
            word_topics = pyro.sample("word_topics_{}".format(doc),
                                      dist.Categorical(category_topics[int(doc_category_list[index].item())]))
            # TODO Enum parallel/sequential optimizing?

            doc_word_list.append(pyro.sample("doc_words_{}".format(doc), dist.Categorical(topic_words[word_topics]),
                                             obs=cur_doc_word_data))

    results = {"topic_weights": topic_weights,
               "topic_words": topic_words,
               "doc_word_data": doc_word_list,
               "category_weights": category_weights,
               "category_topics": category_topics,
               "category_data": doc_category_list}

    return results


def parametrized_guide(doc_word_data, category_data, args, batch_size=None):
    # Use a conjugate guide for global variables.
    topic_weights_posterior = pyro.param(
        "topic_weights_posterior",
        lambda: torch.ones(args.num_topics),
        constraint=constraints.positive)
    topic_words_posterior = pyro.param(
        "topic_words_posterior",
        lambda: torch.ones(args.num_topics, args.num_words),
        constraint=constraints.greater_than(0.5))  # TODO constraint may be too restrictive
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
        lambda: torch.ones(args.num_categories),
        constraint=constraints.less_than(args.num_categories))
    for doc in pyro.plate("documents", args.num_docs, batch_size):
        pyro.sample("doc_categories_{}".format(doc), dist.Delta(doc_category_posterior, event_dim=1),
                    infer={'is_auxiliary': True})  # TODO might be worth finding out why is_auxilliary is necessary here
        # TODO word_topics param might be necessary now, without the enum = parallel in the model


def main(args):
    logging.info(f"CUDA available: {torch.cuda.is_available()}")
    logging.info('Generating data')
    pyro.set_rng_seed(0)
    pyro.clear_param_store()
    pyro.enable_validation(True)

    # Debugging the trace of the model. For showing the shapes of the tensors through the model
    # tracemodel = functools.partial(model, args=args)
    # trace = poutine.trace(tracemodel).get_trace()
    # trace.compute_log_prob()  # optional, but allows printing of log_prob shapes
    # print(trace.format_shapes())

    # We can generate synthetic data directly by calling the model.
    data = model(args=args)

    gen_doc_word_data = data["doc_word_data"]
    gen_category_data = data["category_data"]

    # Loading data
    corpora = prepro_file_load("corpora")
    documents = list(prepro_file_load("id2pre_text").values())
    documents = [re.sub("[\[\]',]", "", doc).split() for doc in documents]
    category_list = [[cat] for cat in list(prepro_file_load("id2category").values())]
    category_corpora = prepro_file_load("category_corpora")

    doc_word_data = [torch.tensor(list(filter(lambda a: a != -1, corpora.doc2idx(doc))), dtype=torch.int64)
                     for doc in documents]
    doc_category_data = [torch.tensor(next(filter(lambda a: a != -1, category_corpora.doc2idx(cat))), dtype=torch.int64)
                         for cat in category_list]
    # TODO X check if there are differences in this date and model generated data

    # Slice data to only use data from the first n documents
    data_slice = None
    if data_slice is not None:
        doc_word_data = doc_word_data[:data_slice]
        doc_category_data = doc_category_data[:data_slice]

    # Setting the new args
    args.num_words_per_doc = list(map(len, doc_word_data))
    args.num_words = len(corpora)
    args.num_docs = len(doc_word_data)
    args.num_categories = len(category_corpora)
    args.num_topics = args.num_categories * 2  # TODO X test different amounts of topics

    # We'll train using SVI.
    logging.info('-' * 40)
    logging.info('Training on {} documents'.format(args.num_docs))
    Elbo = JitTraceEnum_ELBO if args.jit else TraceEnum_ELBO  # TODO test TraceEnum_ vs Trace_
    elbo = Elbo(max_plate_nesting=2)  # TODO Changing the max plate nesting value might be worth looking at
    optim = ClippedAdam({'lr': args.learning_rate})  # TODO X try different learning rates
    svi = SVI(model, parametrized_guide, optim, elbo)

    # If generating data from the model, turn category list to tensor
    # gen_category_data = torch.Tensor(gen_category_data)

    losses = []

    # Training for num_steps iterations
    logging.info('Step\tLoss')
    for step in tqdm(range(args.num_steps)):
        loss = svi.step(doc_word_data=doc_word_data, category_data=doc_category_data, args=args,
                        batch_size=args.batch_size)
        losses.append(loss)
        if step % 10 == 0:
            logging.info('{: >5d}\t{}'.format(step, loss))
    loss = elbo.loss(model, parametrized_guide, doc_word_data=doc_word_data, category_data=doc_category_data, args=args,
                     batch_size=args.batch_size)
    logging.info('final loss = {}'.format(loss))

    # Print params after training
    print('topic_weights_posterior = ', pyro.param("topic_weights_posterior"))
    print('topic_words_posterior = ', pyro.param("topic_words_posterior"))
    print('category_weights_posterior = ', pyro.param("category_weights_posterior"))
    print('category_topics_posterior = ', pyro.param("category_topics_posterior"))
    print('doc_category_posterior = ', pyro.param("doc_category_posterior"))

    # Plot loss over iterations
    plt.plot(losses)
    plt.title("ELBO")
    plt.xlabel("step")
    plt.ylabel("loss")
    plot_file_name = "../loss-2017_categories-" + str(args.num_categories) + \
                     "_topics-" + str(args.num_topics) + \
                     "_batch-" + str(args.batch_size) + \
                     "_lr-" + str(args.learning_rate) + \
                     "_data-size-" + str(data_slice) + \
                     ".png"
    plt.savefig(plot_file_name)
    plt.show()

    # save model
    pyro.get_param_store().save("mymodelparams.pt")

    # load model
    # saved_model_dict = torch.load("mymodel.pt")
    # predictor.load_state_dict(saved_model_dict['model'])
    # guide = saved_model_dict['guide']
    # pyro.get_param_store().load("mymodelparams.pt")


if __name__ == '__main__':
    assert pyro.__version__.startswith('1.5.2')

    # Default args
    parser = argparse.ArgumentParser(description="Amortized Latent Dirichlet Allocation")
    parser.add_argument("-c", "--num-categories", default=32, type=int)
    parser.add_argument("-t", "--num-topics", default=64, type=int)
    parser.add_argument("-w", "--num-words", default=1024, type=int)
    parser.add_argument("-wd", "--num-words-per-doc", default=np.random.randint(low=5, high=300, size=1000))
    parser.add_argument("-d", "--num-docs", default=1000, type=int)
    parser.add_argument("-n", "--num-steps", default=500, type=int)
    parser.add_argument("-l", "--layer-sizes", default="100-100")
    parser.add_argument("-lr", "--learning-rate", default=0.01, type=float)
    parser.add_argument("-b", "--batch-size", default=32, type=int)  # TODO try different batch sizes
    parser.add_argument('--jit', action='store_true')
    args = parser.parse_args()
    main(args)
