#!/usr/bin/env python3

import logging
import collections
import functools
import concurrent.futures

import numpy as np
import zarr
import tskit


logger = logging.getLogger(__name__)


def _sort_similarity(A, c):
    """
    Sort the columns of 2d-array A by Euclidean distance to vector c.
    """
    dist = np.sum((A - c)**2, axis=0)
    idx = np.argsort(dist, kind="stable")
    return A[:, idx]


def _reorder_and_sort(A, counts, from_, to, ref_pop):
    """
    Reorder populations and sort haplotypes within the populations.
    """
    assert len(from_) == len(to) == len(counts)

    # Average over the columns of the ref population. If the genotype
    # matrix were not resized, these would be the ref pop's allele frequencies.
    a = from_[ref_pop]
    b = a + counts[ref_pop]
    ref_vec = np.mean(A[:, a:b], axis=1, keepdims=True)

    B = np.empty_like(A)
    for pop, n in counts.items():
        a = from_[pop]
        b = a + n
        d = to[pop]
        e = d + n
        B[:, d:e] = _sort_similarity(A[:, a:b], ref_vec)
    return B


def ts_pop_counts_indices(ts):
    """
    Get the per-population sample counts, and indices that partition the
    samples into populations.
    """
    pops = [ts.get_population(sample) for sample in ts.samples()]
    # Ordering of items in collections.Counter() should match order added.
    counts = collections.Counter(pops)
    indices = np.cumsum(list(counts.values()))
    # Record the starting indices.
    indices = [0] + list(indices[:-1])
    return counts, dict(zip(counts.keys(), indices))


def _ts2mat(ts, num_rows, maf_thres, rng, exclude_mut_with_metadata=True):
    """
    Extract genotype matrix from ``ts``, and resize to ``num_rows`` rows.
    """
    ac_thres = maf_thres * ts.num_samples
    A = np.zeros((num_rows, ts.num_samples), dtype=np.int8)
    sequence_length = ts.sequence_length
    for variant in ts.variants():
        if exclude_mut_with_metadata:
            # We wish to exclude the mutation added in by SLiM that's under
            # selection in our selection scenarios. It's possible that a CNN
            # could learn to classify based on this mutation alone, which
            # is problematic given that we added it in a biologically
            # unrealistic way (i.e. at a fixed position).
            # XXX: need a less fragile detection/removal of the drawn mutation.
            for mut in variant.site.mutations:
                if len(mut.metadata) > 0:
                    continue
        genotypes = variant.genotypes
        allele_counts = collections.Counter(genotypes)
        if allele_counts[0] < ac_thres or allele_counts[1] < ac_thres:
            continue
        # Polarise 0 and 1 in genotype matrix by major allele frequency.
        # If allele counts are the same, randomly choose a major allele.
        if allele_counts[1] > allele_counts[0] or \
                (allele_counts[1] == allele_counts[0] and rng.random() > 0.5):
            genotypes = genotypes ^ 1
        j = int(num_rows * variant.site.position / sequence_length)
        A[j, :] += genotypes
    return A


def ts_genotype_matrix(
        ts_file, pop_indices, ref_pop, num_rows, num_cols, maf_thres, rng):
    """
    Return a genotype matrix from ``ts``, shrunk to ``num_rows``,
    with populations ordered according to ``pop_indices`` and haplotypes
    within populations sorted left-to-right by similarity to ``ref_pop``.
    ``ref_pop`` should be the beneficial mutation donor population.

    XXX: Don't put the beneficial-mutation recipient population to the
         left of the donor population!
    TODO: sort haplotypes right-to-left if the population sits to the
          left of ``ref_pop``.
    """
    assert ref_pop in pop_indices
    ts = tskit.load(ts_file)
    A = _ts2mat(ts, num_rows, maf_thres, rng)
    ts_counts, ts_pop_indices = ts_pop_counts_indices(ts)
    if sum(ts_counts.values()) != num_cols:
        raise RuntimeError(
                f"{ts_file}: found {sum(ts_counts.values())} samples, "
                f"but expected {num_cols}")
    return _reorder_and_sort(A, ts_counts, ts_pop_indices, pop_indices, ref_pop)


def _prepare_training_data(
        path, tranche, pop_indices, ref_pop, num_rows, num_cols, rng,
        parallelism, maf_thres, train_frac=0.9):
    """
    Load and label the data, then split into training and validation sets.
    """
    files = []
    labels = []  # integer labels
    if len(tranche) < 2:
        raise RuntimeError("Must specify at least two tranches.")
    for i, (tr_id, tr_list) in enumerate(tranche.items()):
        n_tranche = 0
        for tr in tr_list:
            source = path / tr
            logger.debug(f"Looking for {tr_id} simulations under {source}...")
            _files = list(source.rglob("*.trees"))
            n_tranche += len(_files)
            files.extend(_files)
        if n_tranche == 0:
            raise RuntimeError(f"No *.trees found for tranche {tr_id}.")
        logger.debug(f"Found {n_tranche} trees for tranche {tr_id}.")
        labels.extend([i] * n_tranche)

    n = len(files)
    indexes = list(range(n))
    rng.shuffle(indexes)
    files = [files[i] for i in indexes]
    labels = np.fromiter((labels[i] for i in indexes), dtype=np.int8)
    n_train = round(n * train_frac)

    logger.debug(
            f"Converting {n} tree sequence files to genotype matrices with "
            f"shape ({num_rows}, {num_cols})...")

    # XXX: parallelism here breaks fixed-seed rng determinism.
    gt_func = functools.partial(
            ts_genotype_matrix, pop_indices=pop_indices, ref_pop=ref_pop,
            num_rows=num_rows, num_cols=num_cols, maf_thres=maf_thres, rng=rng)

    with concurrent.futures.ProcessPoolExecutor(parallelism) as ex:
        data = list(ex.map(gt_func, files, chunksize=10))
    data = np.array(data, dtype=np.int8)

    train_data, val_data = data[:n_train], data[n_train:]
    train_labels, val_labels = labels[:n_train], labels[n_train:]

    return train_data, train_labels, val_data, val_labels


def _check_data(data, tranche, num_rows, num_cols):
    """
    A sprinkling of paranoia.
    """
    n_ids = len(tranche)
    assert n_ids >= 2, "Must specify at least two tranches."
    train_data, train_labels, val_data, val_labels = data
    for d in (train_data, val_data):
        assert len(d.shape) == 3, "Data has too many dimensions."
        assert d.shape[1:3] == (num_rows, num_cols), \
            f"Data has shape {d.shape[1:3]}, but ({num_rows}, {num_cols}) " \
            "was expected."
    assert train_data.shape[2] == val_data.shape[2], \
        "Training and validation data have different shapes."
    assert train_data.shape[0] == train_labels.shape[0], \
        "The number of data instances doesn't match the number of labels."
    assert val_data.shape[0] == val_labels.shape[0], \
        "The number of data instances doesn't match the number of labels."
    for i, l in enumerate((train_labels, val_labels)):
        which = "training" if i == 0 else "validation"
        n_unique_ids = len(np.unique(l))
        assert n_unique_ids == n_ids, \
            f"The {which} data has {n_unique_ids} unique label(s), " \
            f"but {n_ids} tranches were specified"


def prepare_training_data(
        path, tranche, pop_indices, ref_pop, num_rows, num_cols, rng,
        parallelism, maf_thres, cache):
    """
    Wrapper for _prepare_training_data() that maintains an on-disk zarr cache.
    """
    cache_keys = ("train/data", "train/labels", "val/data", "val/labels")
    if cache.exists():
        logger.debug(f"Loading data from {cache}.")
        store = zarr.load(str(cache))
        data = tuple(store[k] for k in cache_keys)
    else:
        # Data are not cached, load them up.
        data = _prepare_training_data(
                path, tranche, pop_indices, ref_pop, num_rows, num_cols, rng,
                parallelism, maf_thres)
        logger.debug(f"Caching data to {cache}.")
        data_kwargs = {k: zarr.array(v, chunks=False)
                       for k, v in zip(cache_keys, data)}
        zarr.save(str(cache), **data_kwargs)
    _check_data(data, tranche, num_rows, num_cols)
    return data