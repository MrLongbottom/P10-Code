import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
import torch


def model(data):
    theta = pyro.sample('theta', dist.Dirichlet(torch.ones(6)))
    with pyro.plate('data', len(data)):
        pyro.sample('obs', dist.Categorical(theta), obs=data)


data = torch.tensor([5, 4, 2, 5, 6, 5, 3, 3, 1, 5, 5, 3, 5, 3, 5, 3, 5, 5, 3, 5, 5, 3, 1, 5, 3, 3, 6, 5, 5, 6])

nuts_kernel = NUTS(model)
mcmc = MCMC(nuts_kernel, num_samples=1000, warmup_steps=200)
mcmc.run(data - 1)  # -1: we need to work with indices [0, 5] instead of [1, 6]
hmc_samples = {k: v.detach().cpu().numpy()
               for k, v in mcmc.get_samples().items()}


means = hmc_samples['theta'].mean(axis=0)
stds = hmc_samples['theta'].std(axis=0)
print('Inferred dice probabilities from the data (68% confidence intervals):')
for i in range(6):
    print('%d: %.2f ± %.2f' % (i + 1, means[i], stds[i]))