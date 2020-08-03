"""
Test-agnostic functions and classes that simplify testing.
"""
import functools
import logging

from genomatnn import sim

logging.disable(logging.CRITICAL)


# Just for memoizing basic_sim(sample_counts=HashableDict(...)).
# This is a bad idea in general, so don't use outside these tests.
class HashableDict(dict):
    def __hash__(self):
        return tuple(self).__hash__()


@functools.lru_cache(maxsize=32)
def basic_sim(sample_counts, sequence_length=int(1e5), min_allele_frequency=0.05):
    modelspec = "HomSap/PapuansOutOfAfrica_10J19/Neutral/msprime"
    model = sim.get_demog_model(modelspec)
    ts = sim.sim(
        modelspec,
        sample_counts=sample_counts,
        sequence_length=sequence_length,
        min_allele_frequency=min_allele_frequency,
        seed=1234,
    )
    return ts, model


def reorder_indices(model, sample_counts):
    """
    Return new indices to change model ordering to the sample_counts ordering.
    """
    pop_id = {pop.id: j for j, pop in enumerate(model.populations)}
    new_indices = {}
    offset = 0
    for i, (pop, count) in enumerate(sample_counts.items()):
        j = pop_id[pop]
        new_indices[j] = offset
        offset += count
    return new_indices
