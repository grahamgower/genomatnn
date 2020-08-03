#!/usr/bin/env python3

import logging
import collections
import functools
import concurrent.futures

import numpy as np
import zarr
import tskit

from genomatnn import provenance

logger = logging.getLogger(__name__)


def sort_similarity(A, c):
    """
    Sort the columns of 2d-array A by Euclidean distance to vector c.
    """
    assert len(A.shape) == 2
    assert len(c.shape) == 2 and c.shape[1] == 1
    dist = np.sum((A - c) ** 2, axis=0)
    idx = np.argsort(dist, kind="stable")
    return A[:, idx]


def verify_partition(indexes, counts, total):
    seen = set()
    for a, n in zip(indexes, counts):
        if a + n > total:
            raise ValueError(f"{a}:{a+n} out of bounds for 0:{total}")
        itv = set(range(a, a + n))
        if len(seen & itv) != 0:
            raise ValueError(f"overlapping interval at {a}:{a+n}")
        seen.update(itv)

    if len(seen) != total:
        raise ValueError(f"partition is not an exact covering of 0:{total}")


def reorder_and_sort(A, counts, from_, to, ref_pop):
    """
    Reorder populations and sort haplotypes within the populations.
    """
    assert len(from_) == len(to) == len(counts)
    idlist = list(counts.keys())
    verify_partition([from_[id] for id in idlist], counts.values(), A.shape[1])
    verify_partition([to[id] for id in idlist], counts.values(), A.shape[1])

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
        B[:, d:e] = sort_similarity(A[:, a:b], ref_vec)
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


def ts2mat(
    ts, num_rows, maf_thres, rng, pop_intervals=None, exclude_mut_with_metadata=True
):
    """
    Returns a resized and MAF filtered genotype matrix, plus the allele
    frequencies of the drawn mutation.

    The genotype matrix is extracted from ``ts``, and resized to ``num_rows``.
    The allele frequency is calculated for each population defined by the
    population indexes ``pop_intervals``.
    """
    ac_thres = maf_thres * ts.num_samples
    A = np.zeros((num_rows, ts.num_samples), dtype=np.int8)
    sequence_length = ts.sequence_length
    if pop_intervals is None:
        pop_intervals = [(0, ts.num_samples)]
    af = [0] * len(pop_intervals)  # Allele frequency of the drawn mutation.
    for variant in ts.variants():
        genotypes = variant.genotypes

        # We wish to exclude the mutation added in by SLiM that's under
        # selection in our selection scenarios. It's possible that a CNN
        # could learn to classify based on this mutation alone, which
        # is problematic given that we added it in a biologically
        # unrealistic way (i.e. at a fixed position).
        # XXX: need a less fragile detection/removal of the drawn mutation.
        skip_variant = False
        for mut in variant.site.mutations:
            if len(mut.metadata) > 0:
                assert all(np.array(af) == 0)
                af = [genotypes[a:b].sum() / (b - a) for a, b in pop_intervals]
                if exclude_mut_with_metadata:
                    skip_variant = True
        if skip_variant:
            continue

        ac1 = np.sum(genotypes)
        ac0 = len(genotypes) - ac1

        if min(ac0, ac1) < ac_thres:
            continue
        # Polarise 0 and 1 in genotype matrix by major allele frequency.
        # If allele counts are the same, randomly choose a major allele.
        if ac1 > ac0 or (ac1 == ac0 and rng.random() > 0.5):
            genotypes ^= 1
        j = int(num_rows * variant.site.position / sequence_length)
        A[j, :] += genotypes
    return A, af


def ts_genotype_matrix(
    ts_file, *, pop_indices, ref_pop, num_rows, num_cols, maf_thres, rng
):
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
    ts_counts, ts_pop_indices = ts_pop_counts_indices(ts)
    pop_partition = [
        (j, j + n) for j, n in zip(ts_pop_indices.values(), ts_counts.values())
    ]
    A, af = ts2mat(ts, num_rows, maf_thres, rng, pop_intervals=pop_partition)
    if sum(ts_counts.values()) != num_cols:
        raise RuntimeError(
            f"{ts_file}: found {sum(ts_counts.values())} samples, "
            f"but expected {num_cols}"
        )
    B = reorder_and_sort(A, ts_counts, ts_pop_indices, pop_indices, ref_pop)
    params = provenance.load_parameters(ts)
    # TODO remove the s/4G19/4G20/ renaming
    params["modelspec"] = params["modelspec"].replace(
        "HomininComposite_4G19", "HomininComposite_4G20"
    )
    params["AF"] = np.array(af)
    metadata = tuple(params[k] for k in ("modelspec", "T_mut", "T_sel", "s", "AF"))
    return B, metadata


def _prepare_data(
    *,
    path,
    tranche,
    pop_indices,
    ref_pop,
    num_rows,
    num_cols,
    rng,
    parallelism,
    maf_thres,
):
    """
    Load and label the data.
    """
    files = []
    labels = []  # integer labels
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

    logger.debug(
        f"Converting {n} tree sequence files to genotype matrices with "
        f"shape ({num_rows}, {num_cols})..."
    )

    # XXX: parallelism here breaks fixed-seed rng determinism.
    gt_func = functools.partial(
        ts_genotype_matrix,
        pop_indices=pop_indices,
        ref_pop=ref_pop,
        num_rows=num_rows,
        num_cols=num_cols,
        maf_thres=maf_thres,
        rng=rng,
    )

    with concurrent.futures.ProcessPoolExecutor(parallelism) as ex:
        res = list(ex.map(gt_func, files, chunksize=10))
        data, metadata = zip(*res)
    data = np.array(data, dtype=np.int8)

    max_modelspec_len = functools.reduce(max, (len(md[0]) for md in metadata))
    af_len = len(metadata[0][-1])
    metadata_dtype = [
        ("modelspec", f"U{max_modelspec_len}"),
        ("T_mut", float),
        ("T_sel", float),
        ("s", float),
        ("AF", (float, af_len)),
    ]
    metadata = np.fromiter(metadata, dtype=metadata_dtype)
    return data, labels, metadata


def _prepare_training_data(
    *,
    path,
    tranche,
    pop_indices,
    ref_pop,
    num_rows,
    num_cols,
    rng,
    parallelism,
    maf_thres,
    train_frac,
):
    """
    Load and label the data, then split into training and validation sets.
    """
    if len(tranche) < 2:
        raise RuntimeError("Must specify at least two tranches.")
    data, labels, metadata = _prepare_data(
        path=path,
        tranche=tranche,
        pop_indices=pop_indices,
        ref_pop=ref_pop,
        num_rows=num_rows,
        num_cols=num_cols,
        rng=rng,
        parallelism=parallelism,
        maf_thres=maf_thres,
    )

    n = len(data)
    n_train = round(n * train_frac)
    train_data, val_data = data[:n_train], data[n_train:]
    train_labels, val_labels = labels[:n_train], labels[n_train:]
    train_metadata, val_metadata = metadata[:n_train], metadata[n_train:]

    return (
        train_data,
        train_labels,
        train_metadata,
        val_data,
        val_labels,
        val_metadata,
    )


def check_data(data, tranche, num_rows, num_cols):
    """
    A sprinkling of paranoia.
    """
    n_ids = len(tranche)
    assert n_ids >= 2, "Must specify at least two tranches."
    (
        train_data,
        train_labels,
        train_metadata,
        val_data,
        val_labels,
        val_metadata,
    ) = data
    for d in (train_data, val_data):
        assert len(d.shape) == 3, "Data has too many dimensions."
        assert d.shape[1:3] == (num_rows, num_cols), (
            f"Data has shape {d.shape[1:3]}, but ({num_rows}, {num_cols}) "
            "was expected."
        )
    assert (
        train_data.shape[2] == val_data.shape[2]
    ), "Training and validation data have different shapes."
    assert (
        train_data.shape[0] == train_labels.shape[0]
    ), "The number of training data instances doesn't match the number of labels."
    assert (
        train_data.shape[0] == train_metadata.shape[0]
    ), "The number of training data instances doesn't match the metadata."
    assert (
        val_data.shape[0] == val_labels.shape[0]
    ), "The number of validataion data instances doesn't match the number of labels."
    assert (
        val_data.shape[0] == val_metadata.shape[0]
    ), "The number of validation data instances doesn't match the metadata."
    for i, l in enumerate((train_labels, val_labels)):
        which = "training" if i == 0 else "validation"
        n_unique_ids = len(np.unique(l))
        assert n_unique_ids == n_ids, (
            f"The {which} data has {n_unique_ids} unique label(s), "
            f"but {n_ids} tranches were specified"
        )


_cache_keys = (
    "train/data",
    "train/labels",
    "train/metadata",
    "val/data",
    "val/labels",
    "val/metadata",
)


def load_data_cache(cache, cache_keys=_cache_keys):
    if not cache.exists():
        raise RuntimeError(f"{cache} doesn't exist")
    logger.debug(f"Loading data from {cache}.")
    store = zarr.load(str(cache))
    return tuple(store[k] for k in cache_keys)


def save_data_cache(cache, data, cache_keys=_cache_keys):
    logger.debug(f"Caching data to {cache}.")
    data_kwargs = dict()
    max_chunk_size = 2 ** 30  # 1 Gb
    for k, v in zip(cache_keys, data):
        shape = list(v.shape)
        size = v.size * v.itemsize
        if size > max_chunk_size:
            shape[0] = int(shape[0] * max_chunk_size / size)
        data_kwargs[k] = zarr.array(v, chunks=shape)
    zarr.save(str(cache), **data_kwargs)


def prepare_training_data(
    *,
    path,
    tranche,
    pop_indices,
    ref_pop,
    num_rows,
    num_cols,
    rng,
    parallelism,
    maf_thres,
    cache,
    train_frac,
    filter_pop,
    filter_modelspec,
    filter_AF,
):
    """
    Wrapper for _prepare_training_data() that maintains an on-disk zarr cache.
    """
    if cache.exists():
        data = load_data_cache(cache)
    else:
        # Data are not cached, load them up.
        data = _prepare_training_data(
            path=path,
            tranche=tranche,
            pop_indices=pop_indices,
            ref_pop=ref_pop,
            num_rows=num_rows,
            num_cols=num_cols,
            rng=rng,
            parallelism=parallelism,
            maf_thres=maf_thres,
            train_frac=train_frac,
        )
        if filter_pop is not None and filter_modelspec is not None:
            data = filter_by_af(data, filter_pop, filter_modelspec, filter_AF)
        save_data_cache(cache, data)
    check_data(data, tranche, num_rows, num_cols)
    return data


def prepare_extra(
    *,
    path,
    tranche,
    pop_indices,
    ref_pop,
    num_rows,
    num_cols,
    rng,
    parallelism,
    maf_thres,
    cache,
):
    cache_keys = (
        "extra/data",
        "extra/labels",
        "extra/metadata",
    )
    if cache.exists():
        data = load_data_cache(cache, cache_keys=cache_keys)
    else:
        # Data are not cached, load them up.
        data = _prepare_data(
            path=path,
            tranche=tranche,
            pop_indices=pop_indices,
            ref_pop=ref_pop,
            num_rows=num_rows,
            num_cols=num_cols,
            rng=rng,
            parallelism=parallelism,
            maf_thres=maf_thres,
        )
        save_data_cache(cache, data, cache_keys=cache_keys)
    # TODO fix check_data to work with n_tranches != 2
    # check_data(data, tranche, num_rows, num_cols)
    return data


def filter_by_af(data, pop, modelspec, af):
    def _filt(data, labels, metadata, pop, af):
        idx = np.where(
            np.bitwise_or(
                np.char.find(metadata["modelspec"], modelspec) == -1,
                metadata["AF"][:, pop] >= af,
            )
        )
        data = data[idx]
        labels = labels[idx]
        metadata = metadata[idx]
        return data, labels, metadata

    (
        train_data,
        train_labels,
        train_metadata,
        val_data,
        val_labels,
        val_metadata,
    ) = data
    train_data, train_labels, train_metadata = _filt(
        train_data, train_labels, train_metadata, pop, af
    )
    val_data, val_labels, val_metadata = _filt(
        val_data, val_labels, val_metadata, pop, af
    )
    data = (
        train_data,
        train_labels,
        train_metadata,
        val_data,
        val_labels,
        val_metadata,
    )
    return data
