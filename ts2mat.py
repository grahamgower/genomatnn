import numpy as np


def ts2mat(ts, num_rows):
    """
    Extract genotype matrix from ``ts``, and resize to ``num_rows`` rows.
    """
    m = np.zeros((num_rows, ts.num_samples), dtype=np.int8)
    sequence_length = ts.sequence_length
    for variant in ts.variants():
        j = int(num_rows * variant.site.position / sequence_length)
        m[j, :] += variant.genotypes
    return m


if __name__ == "__main__":
    import msprime
    import vcf2mat

    num_rows = 32

    # check that ts2mat() and vcf2mat.resize() are equivalent
    ts = msprime.simulate(
        random_seed=1234,
        sample_size=100, Ne=10000, length=1e5,
        mutation_rate=1e-8, recombination_rate=1e-8)
    m1 = ts2mat(ts, num_rows)
    pos = [site.position for site in ts.sites()]
    m2 = vcf2mat.resize(pos, ts.genotype_matrix(), ts.sequence_length, num_rows)
    np.testing.assert_array_equal(m1, m2)
