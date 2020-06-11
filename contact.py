import numpy as np
import msprime
import stdpopsim


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


def _5pop_test_demog(N=1000):
    populations = [stdpopsim.Population(f"pop{i}", f"Population {i}") for i in range(5)]
    pop_config = [
        msprime.PopulationConfiguration(
            initial_size=N, metadata=populations[i].asdict()
        )
        for i in range(len(populations))
    ]
    mig_mat = [
        [0, 0, 0, 0, 0],
        [0, 0, 1e-5, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]
    dem_events = [
        msprime.MassMigration(time=100, source=0, destination=1, proportion=0.1),
        msprime.MassMigration(time=200, source=3, destination=2),
        msprime.MigrationRateChange(time=200, rate=0),
        msprime.MassMigration(time=300, source=1, destination=0),
        msprime.MassMigration(time=400, source=2, destination=4, proportion=0.1),
        msprime.MassMigration(time=600, source=2, destination=0),
        msprime.MassMigration(time=700, source=4, destination=0),
    ]
    return stdpopsim.DemographicModel(
        id="_5pop_test",
        description="_5pop_test",
        long_description="_5pop_test",
        populations=populations,
        generation_time=1,
        population_configurations=pop_config,
        demographic_events=dem_events,
        migration_matrix=mig_mat,
    )


if __name__ == "__main__":
    model = _5pop_test_demog()
    ddb = msprime.DemographyDebugger(
        demographic_events=model.demographic_events,
        population_configurations=model.population_configurations,
        migration_matrix=model.migration_matrix,
    )

    assert tmrca(model, 0, 1) == 100
    assert tmrca(model, 0, 2) == 100
    assert tmrca(model, 1, 2) == 0
    assert tmrca(model, 0, 3) == 200
    assert tmrca(model, 1, 3) == 200
    assert tmrca(model, 2, 3) == 200
    assert tmrca(model, 0, 4) == 400
    assert tmrca(model, 1, 4) == 400
    assert tmrca(model, 2, 4) == 400
    assert tmrca(model, 3, 4) == 400

    # ancient samples
    assert (
        min_coalescence_time(
            ddb,
            [
                msprime.Sample(time=150, population=0),
                msprime.Sample(time=0, population=1),
            ],
        )
        == 300
    )
    assert (
        min_coalescence_time(
            ddb,
            [
                msprime.Sample(time=150, population=0),
                msprime.Sample(time=0, population=4),
            ],
        )
        == 700
    )
    assert (
        min_coalescence_time(
            ddb,
            [
                msprime.Sample(time=0, population=0),
                msprime.Sample(time=450, population=4),
            ],
        )
        == 450
    )
