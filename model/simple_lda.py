import argparse
import logging

import torch
import torch.distributions.constraints as constraints

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, JitTraceEnum_ELBO, TraceEnum_ELBO, config_enumerate, Trace_ELBO
from pyro.optim import ClippedAdam
from tqdm import tqdm


def model(data, args):
    gamma = torch.ones(args.num_words) / args.num_words
    with pyro.plate('topics', args.num_topics) as k:
        beta = pyro.sample('beta', dist.Dirichlet(gamma))
        assert beta.shape == (args.num_topics, args.num_words)

    alpha = torch.ones(args.num_topics) / args.num_topics
    with pyro.plate('documents', args.num_docs) as d:
        theta = pyro.sample('theta', dist.Dirichlet(alpha))
        assert theta.shape == (args.num_docs, 100, args.num_topics)

        with pyro.plate('words', args.num_words) as n:
            z = pyro.sample('z', dist.Categorical(theta))
            # assert z.shape == (args.num_words, args.num_docs)
            w = pyro.sample('w', dist.Categorical(beta[z]), obs=data[d, :])
            assert w.shape == (args.num_docs, args.num_words)


def guide(data, args):
    with pyro.plate('topics', args.num_topics) as k:
        gamma_q = pyro.param('gamma_q', torch.ones(args.num_words), constraint=constraints.positive)
        beta_q = pyro.sample('beta', dist.Dirichlet(gamma_q))
        assert beta_q.shape == (args.num_topics, args.num_words)

    with pyro.plate('documents', args.num_docs, dim=-2):
        theta = pyro.sample('theta', dist.Dirichlet(alpha))
        assert theta.shape == (args.num_docs, 100, args.num_topics)

        with pyro.plate('words', args.num_words, dim=-1):
            z = pyro.sample('z', dist.Categorical(theta))
            # assert z.shape == (args.num_docs, args.num_words)
            w = pyro.sample('w', dist.Categorical(beta[z]), obs=data)
            assert w.shape == (args.num_docs, args.num_words)


def main(args):
    pyro.clear_param_store()
    optimizer = ClippedAdam({'lr': 0.01})
    Elbo = JitTraceEnum_ELBO if args.jit else Trace_ELBO
    elbo = Elbo(max_plate_nesting=2)
    svi = SVI(model, config_enumerate(guide, 'parallel'), optimizer, loss=elbo)

    # Data
    z = [torch.zeros(args.num_words, dtype=torch.long) for i in range(args.num_docs)]
    data = [torch.zeros(args.num_words) for i in range(args.num_docs)]
    for d in tqdm(range(args.num_docs)):
        for n in range(args.num_words):
            z[d][n] = dist.Categorical(theta[d, :]).sample()
            data[d][n] = dist.Categorical(beta[z[d][n], :]).sample()
    data = torch.stack(data)

    for step in tqdm(range(args.num_steps)):
        loss = svi.step(data, args=args)
        if step % 10 == 0:
            print(loss)
    loss = elbo.loss(model, guide, data, args=args)
    logging.info('final loss = {}'.format(loss))


if __name__ == '__main__':
    assert pyro.__version__.startswith('1.5.2')
    parser = argparse.ArgumentParser(description="Amortized Latent Dirichlet Allocation")
    parser.add_argument("-t", "--num-topics", default=30, type=int)
    parser.add_argument("-w", "--num-words", default=100, type=int)
    parser.add_argument("-d", "--num-docs", default=100, type=int)
    parser.add_argument("-n", "--num-steps", default=500, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.01, type=float)
    parser.add_argument("-b", "--batch-size", default=32, type=int)
    parser.add_argument('--jit', action='store_true')
    args = parser.parse_args()
    # Hyperparams
    alpha = torch.zeros([args.num_docs, args.num_topics]) + 0.5
    gamma = torch.zeros([args.num_topics, args.num_words]) + 0.01

    # Priors
    theta = dist.Dirichlet(alpha).sample()
    beta = dist.Dirichlet(gamma).sample()
    main(args)
