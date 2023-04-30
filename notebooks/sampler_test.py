import clrs
import numpy as np
import jax
import jax.numpy as jnp

import pprint

rng = np.random.RandomState(1234)
rng_key = jax.random.PRNGKey(rng.randint(2 ** 32))

train_sampler, spec = clrs.build_sampler(
    name = 'auction_matching',
    num_samples = 1,
    length = 4,
    weighted = True)  # number of nodes

pprint.pprint(spec)  # spec is the algorithm specification, all the probes


def _iterate_sampler(sampler, batch_size):
    while True:
        yield sampler.next(batch_size)


train_sampler = _iterate_sampler(train_sampler, batch_size = 1)


# %%
# Bipartite graph is weighted
print(np.triu(next(train_sampler).features.inputs[1].data[0]))

