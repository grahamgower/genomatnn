#!/usr/bin/env python3
import os.path
import random
import argparse
import collections

import attr
import stdpopsim
import msprime


__version__ = "0.1"


@attr.s(frozen=True, kw_only=True)
class MyContig(stdpopsim.Contig):
    """
    Extend stdpopsim.Contig with an origin attribute. The attribute cannot be
    set on a regular stdpopsim.Contig() instance, becase the class is frozen.
    """
    origin = attr.ib(default=None, type=str)


def simple_chrom_name(chrom):
    if chrom is None:
        return chrom
    if chrom.startswith("chr"):
        chrom = chrom[3:]
    return chrom.lower()


def random_autosomal_chunk(species, genetic_map, length):
    """
    Returns a `length` sized recombination map from the given `species`' `genetic_map`.

    The chromosome is drawn from available autosomes in proportion to their
    length. The position is drawn uniformly from the autosome, excluding any
    flanking regions with zero recombination rate (telomeres).
    """

    if not hasattr(msprime.RecombinationMap, "slice"):
        raise RuntimeError("recombination map slicing requires msprime >= 1.0")

    chromosomes = species.genome.chromosomes
    w = []
    for i, ch in enumerate(chromosomes):
        wl = 0
        if ch.length >= length and simple_chrom_name(ch.id) not in ("x", "y", "m"):
            wl = ch.length
        w.append(wl if i == 0 else wl + w[i-1])
    if w[-1] == 0:
        raise ValueError(f"No chromosomes long enough for length {length}.")
    chrom = random.choices(chromosomes, cum_weights=w)[0]

    gm = species.get_genetic_map(genetic_map)
    recomb_map = gm.get_chromosome_map(chrom.id)
    positions = recomb_map.get_positions()
    rates = recomb_map.get_rates()

    # Get indices for the start and end of the recombination map.
    j = 0
    while j < len(rates) and rates[j] == 0:
        j += 1
    k = len(positions) - 1
    while k > 1 and rates[k-1] == 0:
        k -= 1

    assert j <= k
    if positions[k] - positions[j] < length:
        # TODO: should really check this when we enumerate chromosomes
        raise ValueError(
                f"{chrom.id} was sampled, but its recombination map is "
                f"shorter than {length}.")

    pos = random.randrange(positions[j], positions[k] - length)
    new_map = recomb_map[pos:pos+length]  # requires msprime >= 1.0
    origin = f"{chrom.id}:{pos}-{pos+length}"
    contig = MyContig(
            recombination_map=new_map, mutation_rate=chrom.mutation_rate,
            genetic_map=gm, origin=origin)

    return contig


def homsap_papuans_model(length, sample_counts):
    species = stdpopsim.get_species("HomSap")
    model = species.get_demographic_model("PapuansOutOfAfrica_10J19")
    contig = random_autosomal_chunk(species, "HapMapII_GRCh37", length)
    samples = model.get_samples(
        *[sample_counts.get(p.id, 0) for p in model.populations])
    return species, model, contig, samples


def homsap_papuans_Neutral(model, contig, samples, seed, verbosity, **kwargs):
    engine = stdpopsim.get_engine("msprime")
    ts = engine.simulate(model, contig, samples, seed=seed)
    return ts, (contig.origin, 0, 0, 0)


def homsap_papuans_DFE(model, contig, samples, seed, verbosity, **kwargs):
    mutation_types = stdpopsim.ext.KimDFE()
    engine = stdpopsim.get_engine("slim")
    ts = engine.simulate(
            model, contig, samples,
            seed=seed,
            verbosity=verbosity,
            mutation_types=mutation_types,
            extended_events=[],
            slim_no_recapitation=True,
            )
    return ts, (contig.origin, 0, 0, 0)


def homsap_papuans_AI_Den1_to_Papuan(
        model, contig, samples, seed, verbosity, dfe=False, slim_script=False,
        **kwargs):
    rng = random.Random(seed)

    mutation_types = []
    if dfe:
        mutation_types.extend(stdpopsim.ext.KimDFE())
    positive = stdpopsim.ext.MutationType(convert_to_substitution=False)
    mutation_types.append(positive)
    mut_id = len(mutation_types)

    # TODO: get these from the model itself
    T_Den_Nea_split = 15090
    T_DenA_Den1_split = 9750
    # T_DenA_Den2_split = 12500
    T_Den1_Papuan_mig = 29.8e3 / model.generation_time
    # T_Den2_Papuan_mig = 45.7e3 / model.generation_time

    # should flip a coin to choose Den1 or Den2.
    T_Den_split = T_DenA_Den1_split
    T_mig = T_Den1_Papuan_mig

    T_mut = rng.uniform(T_Den_split, T_Den_Nea_split)
    T_sel = rng.uniform(1e3 / model.generation_time, T_mig)
    s = rng.uniform(0.001, 0.1)

    pop = {p.id: i for i, p in enumerate(model.populations)}
    coordinate = round(contig.recombination_map.get_length() / 2)

    extended_events = [
        # Draw mutation in Denisovans.
        stdpopsim.ext.DrawMutation(
                time=T_mut, mutation_type_id=mut_id, population_id=pop["DenA"],
                coordinate=coordinate,
                # Save state before the mutation is introduced
                save=True),
        # Mutation is positively selected in Papuans
        stdpopsim.ext.ChangeMutationFitness(
                start_time=T_sel, end_time=0,
                mutation_type_id=mut_id, population_id=pop["Papuan"],
                selection_coeff=s, dominance_coeff=0.5),
        # Allele frequency conditioning. If the condition is not met, we
        # restore to the most recent save point.
        stdpopsim.ext.ConditionOnAlleleFrequency(
                start_time=stdpopsim.ext.GenerationAfter(T_mut),
                end_time=T_Den_split,
                mutation_type_id=mut_id, population_id=pop["DenA"],
                op=">", allele_frequency=0),
        stdpopsim.ext.ConditionOnAlleleFrequency(
                start_time=stdpopsim.ext.GenerationAfter(T_Den_split),
                end_time=T_mig,
                mutation_type_id=mut_id, population_id=pop["Den1"],
                op=">", allele_frequency=0,
                # Update save point at start_time.
                save=True),
        stdpopsim.ext.ConditionOnAlleleFrequency(
                start_time=stdpopsim.ext.GenerationAfter(T_mig), end_time=0,
                mutation_type_id=mut_id, population_id=pop["Papuan"],
                op=">", allele_frequency=0,
                # Update save point at start_time.
                save=True),
        stdpopsim.ext.ConditionOnAlleleFrequency(
                start_time=0, end_time=0,
                mutation_type_id=mut_id, population_id=pop["Papuan"],
                op=">", allele_frequency=0.1),
        ]

    engine = stdpopsim.get_engine("slim")
    ts = engine.simulate(
            model, contig, samples,
            seed=rng.randrange(1, 2**32),
            verbosity=verbosity,
            mutation_types=mutation_types,
            extended_events=extended_events,
            slim_script=slim_script,
            slim_no_recapitation=dfe,
            # slim_no_burnin=True,
            )

    return ts, (
            contig.origin, T_mut*model.generation_time,
            T_sel*model.generation_time, s)


def homsap_papuans_Sweep_Papuan(
        model, contig, samples, seed, verbosity, dfe=False, slim_script=False,
        **kwargs):
    rng = random.Random(seed)

    mutation_types = []
    if dfe:
        mutation_types.extend(stdpopsim.ext.KimDFE())
    mutation_types = stdpopsim.ext.KimDFE()
    positive = stdpopsim.ext.MutationType(convert_to_substitution=False)
    mutation_types.append(positive)
    mut_id = len(mutation_types)

    T_Papuan_Ghost_split = 1784
    T_sel = rng.uniform(1e3 / model.generation_time, T_Papuan_Ghost_split)
    T_mut = rng.uniform(T_sel, T_Papuan_Ghost_split)
    s = rng.uniform(0.001, 0.1)

    pop = {p.id: i for i, p in enumerate(model.populations)}
    coordinate = round(contig.recombination_map.get_length() / 2)

    extended_events = [
        # Draw mutation.
        stdpopsim.ext.DrawMutation(
                time=T_mut, mutation_type_id=mut_id, population_id=pop["Papuan"],
                coordinate=coordinate,
                # Save state before the mutation is introduced.
                save=True),
        # Mutation is positively selected at time T_sel.
        stdpopsim.ext.ChangeMutationFitness(
                start_time=T_sel, end_time=0,
                mutation_type_id=mut_id, population_id=pop["Papuan"],
                selection_coeff=s, dominance_coeff=0.5),
        # Allele frequency conditioning. If the condition is not met, we
        # restore to the save point.
        stdpopsim.ext.ConditionOnAlleleFrequency(
                start_time=stdpopsim.ext.GenerationAfter(T_mut), end_time=0,
                mutation_type_id=mut_id, population_id=pop["Papuan"],
                op=">", allele_frequency=0),
        stdpopsim.ext.ConditionOnAlleleFrequency(
                start_time=0, end_time=0,
                mutation_type_id=mut_id, population_id=pop["Papuan"],
                op=">", allele_frequency=0.1),
        ]

    engine = stdpopsim.get_engine("slim")
    ts = engine.simulate(
            model, contig, samples,
            seed=rng.randrange(1, 2**32),
            verbosity=verbosity,
            mutation_types=mutation_types,
            extended_events=extended_events,
            slim_script=slim_script,
            slim_no_recapitation=dfe,
            # slim_no_burnin=True,
            )

    return ts, (
            contig.origin, T_mut*model.generation_time,
            T_sel*model.generation_time, s)


toai_simulations = {
    "HomSap/PapuansOutOfAfrica_10J19": {
        # Model kwargs come first and must be prefixed with an underscore.
        "_sample_counts": {"YRI": 200, "Papuan": 200, "DenA": 2, "NeaA": 2},

        # Base demographic model.
        "demographic_model": homsap_papuans_model,

        # Various mutation models to stack on top of the demographic model.
        "Neutral": homsap_papuans_Neutral,
        "DFE": homsap_papuans_DFE,
        "AI/Den1_to_Papuan": homsap_papuans_AI_Den1_to_Papuan,
        "Sweep/Papuan": homsap_papuans_Sweep_Papuan,
        },
}


def toai_models(mdict=toai_simulations):
    models = {}
    kwargs = {}
    for key, val in mdict.items():
        if key[0] == '_':
            key = key[1:]
            kwargs[key] = val
        elif key == "demographic_model":
            kwargs[key] = val
        elif callable(val):
            models[key] = (val, kwargs)
        else:
            children = (
                    (f"{key}/{m}", (f, kw))
                    for m, (f, kw) in toai_models(val).items())
            models.update(children)
    return models


def parse_args():
    prefixes = "\n".join(toai_simulations.keys())

    parser = argparse.ArgumentParser(description="Run a simulation.")
    parser.add_argument(
            "-v", "--verbose", default=False, action="store_true",
            help="Show verbose output from SLiM script.")
    parser.add_argument(
            "-s", "--seed", type=int, default=None,
            help="Seed for the random number generator.")
    parser.add_argument(
            "-o", "--output-dir", default=".",
            help="Output directory for the tree sequence files.")
    parser.add_argument(
            "-l", "--length", default=100*1000, type=int,
            help="Length of the genomic region to simulate. [default=%(default)s]")
    parser.add_argument(
            "--slim-script", default=False, action="store_true",
            help="Print SLiM script to stdout. Don't run the simulation.")
    parser.add_argument(
            "modelspec",
            help="Model specification. This is either a full model "
                 "specification, or a prefix. If a prefix is used, all model "
                 "specifications with that prefix are printed. "
                 "Available modelspec prefixes:\n"
                 f"{prefixes}")
    args = parser.parse_args()

    models = toai_models()
    for model, (func, kwargs) in models.items():
        if args.modelspec == model:
            args.modelspec = model, (func, kwargs)
            break
    else:
        found = False
        for model, (func, kwargs) in models.items():
            if model.startswith(args.modelspec):
                found = True
                print(model)
                for kw, arg in kwargs.items():
                    if callable(arg):
                        continue
                    print(f"\t\t{kw}: {arg}")
        if found:
            exit(1)
        else:
            parser.error(f"No models matching spec `{args.modelspec}`.")

    args.verbosity = 0
    if args.verbose:
        # Debug verbosity level.
        args.verbosity = 2

    return args


if __name__ == "__main__":
    args = parse_args()
    if args.seed is not None:
        seed = args.seed
    else:
        random.seed()
        seed = random.randrange(1, 2**32)

    modelspec, (sim_func, kwargs) = args.modelspec
    odir = f"{args.output_dir}/{modelspec}"
    os.makedirs(odir, exist_ok=True)

    demographic_model_func = kwargs.get("demographic_model")
    assert demographic_model_func is not None
    del kwargs["demographic_model"]

    sample_counts = kwargs.get("sample_counts")
    assert sample_counts is not None
    del kwargs["sample_counts"]

    species, model, contig, samples = demographic_model_func(
            args.length, sample_counts)

    ts, (origin, T_mut, T_sel, s) = sim_func(
            model, contig, samples, seed, args.verbosity,
            slim_script=args.slim_script, **kwargs)

    if ts is None:
        assert args.slim_script, "ts is None, but no --slim-script requested"
        exit(0)

    ts = stdpopsim.ext.dedup_slim_provenances(ts)

    save_info = dict(
            seed=seed, modelspec=modelspec, origin=origin,
            T_mut=T_mut, T_sel=T_sel, s=s)
    if len(kwargs) > 0:
        save_info["extra_kwargs"] = kwargs

    ts = stdpopsim.ext.save_ext(ts, "toai", __version__, **save_info)

    popid = {i: p.id for i, p in enumerate(model.populations)}
    observed_counts = collections.Counter(
            [popid[ts.get_population(i)] for i in ts.samples()])
    assert observed_counts == sample_counts, f"{observed_counts} != {sample_counts}"

    ts.dump(f"{odir}/{seed}.trees")
