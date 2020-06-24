import numpy as np
import msprime


def reachable_through_migration(mig_mat, lineages):
    # get connected populations via migration
    k = len(lineages)
    M = np.eye(k) + mig_mat + lineages
    reachable = (np.linalg.matrix_power(M, k) > 0) * 1
    return (reachable.dot(lineages) > 0) * 1


def reachability_matrix(ddb):
    """
    A lineage-reachability matrix generator. One matrix is yielded for each
    epoch in the demography debugger `ddb`.
    """
    lineages = np.eye(ddb.num_populations)
    for epoch in ddb.epochs:
        for de in epoch.demographic_events:
            if de.type == "mass_migration":
                for lineage in lineages:
                    if lineage[de.source]:
                        lineage[de.dest] = 1
                        if de.proportion == 1:
                            lineage[de.source] = 0
        lineages = reachable_through_migration(epoch.migration_matrix, lineages)
        yield lineages


def min_coalescence_time(ddb, samples):
    """
    Minimum coalescent time of `samples`.
    """
    samples = sorted(samples, key=lambda sample: sample.time)
    sample_index = 0
    oldest_sample = max(sample.time for sample in samples)
    zero_pops = {sample.population for sample in samples}
    pops = set()

    for epoch, lineages in zip(ddb.epochs, reachability_matrix(ddb)):

        for sample in samples[sample_index:]:
            if sample.time > epoch.start_time:
                break
            zero_pops.remove(sample.population)
            pops.add(sample.population)
            sample_index += 1

        # XXX: modifies the mutable matrix from the lineages generator
        for pop in zero_pops:
            lineages[pop] = np.zeros(ddb.num_populations)
            lineages[pop][pop] = 1

        for sample in samples[sample_index:]:
            if sample.time >= epoch.end_time:
                break
            zero_pops.remove(sample.population)
            pops.add(sample.population)
            sample_index += 1

        if epoch.end_time < oldest_sample:
            continue

        intersect = np.ones(ddb.num_populations)
        for pop in pops:
            np.logical_and(lineages[pop], intersect, out=intersect)
        if any(intersect):
            return max(epoch.start_time, oldest_sample)
    return None


def tmrca(model, *pops):
    """
    Minimum coalescent time between populations with indexes in `pops`.
    """
    samples = [msprime.Sample(time=0, population=pop) for pop in pops]
    ddb = model.get_demography_debugger()
    return min_coalescence_time(ddb, samples)


def split_time(model, p1, p2):
    """
    Split time of p1 and p2 (or their parent lineages).
    """
    if p1 == p2:
        raise ValueError("p1 and p2 are the same")
    for p in (p1, p2):
        if p >= len(model.populations):
            raise ValueError(f"{p} not valid for model")
    pops = sorted((p1, p2))
    last_time = 0
    for de in model.demographic_events:
        if de.time < last_time:
            raise ValueError("demographic_events not sorted in time-ascending order")
        last_time = de.time
        if isinstance(de, msprime.MassMigration):
            pp = sorted((de.source, de.dest))
            if pops == pp:
                if de.proportion == 1:
                    return de.time
            # ascend into parent lineage
            if de.proportion == 1:
                if de.source == pops[0]:
                    pops = sorted((de.dest, pops[1]))
                elif de.source == pops[1]:
                    pops = sorted((pops[0], de.dest))
    return None
