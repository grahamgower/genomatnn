#!/usr/bin/env python3
import math
import random
import collections
import functools
import bisect
import logging

import attr
import msprime
import stdpopsim

from genomatnn import (
    provenance,
    contact,
)


logger = logging.getLogger(__name__)


@attr.s(kw_only=True)
class MyContig(stdpopsim.Contig):
    """
    Extend stdpopsim.Contig with an origin attribute that records the chromosome
    and position from whence the contig was obtained. The attribute cannot be
    set on a regular stdpopsim.Contig() instance, becase the class is frozen.
    """

    origin = attr.ib(default=None, type=str)


def simple_chrom_name(chrom):
    if chrom is None:
        return chrom
    if chrom.startswith("chr"):
        chrom = chrom[3:]
    return chrom.lower()


def recomb_slice(recomb_map, start=None, end=None):
    """
    Returns a subset of this recombination map between the specified end
    points. If start is None, it defaults to 0. If end is None, it defaults
    to the end of the map.
    """
    if hasattr(msprime.RecombinationMap, "slice"):
        raise RuntimeError("Using msprime >= 1.0. Should use recombination map slicing")

    positions = recomb_map.get_positions()
    rates = recomb_map.get_rates()

    if start is None:
        i = 0
        start = 0
    if end is None:
        end = positions[-1]
        j = len(positions)

    if (
        start < 0
        or end < 0
        or start > positions[-1]
        or end > positions[-1]
        or start > end
    ):
        raise IndexError(f"Invalid subset: start={start}, end={end}")

    if start != 0:
        i = bisect.bisect_left(positions, start)
        if start < positions[i]:
            i -= 1
    if end != positions[-1]:
        j = bisect.bisect_right(positions, end, lo=i)

    new_positions = list(positions[i:j])
    new_rates = list(rates[i:j])
    new_positions[0] = start
    if end > new_positions[-1]:
        new_positions.append(end)
        new_rates.append(0)
    else:
        new_rates[-1] = 0
    new_positions = [pos - start for pos in new_positions]

    return msprime.RecombinationMap(new_positions, new_rates)


def random_autosomal_chunk(species, genetic_map, length, seed):
    """
    Returns a `length` sized recombination map from the given `species`' `genetic_map`.

    The chromosome is drawn from available autosomes in proportion to their
    length. The position is drawn uniformly from the autosome, excluding any
    flanking regions with zero recombination rate (telomeres).
    """
    chromosomes = species.genome.chromosomes
    w = []
    for i, ch in enumerate(chromosomes):
        wl = 0
        if ch.length >= length and simple_chrom_name(ch.id) not in ("x", "y", "m"):
            wl = ch.length
        w.append(wl if i == 0 else wl + w[i - 1])
    if w[-1] == 0:
        raise ValueError(f"No chromosomes long enough for length {length}.")
    rng = random.Random(seed)
    chrom = rng.choices(chromosomes, cum_weights=w)[0]

    gm = species.get_genetic_map(genetic_map)
    recomb_map = gm.get_chromosome_map(chrom.id)
    positions = recomb_map.get_positions()
    rates = recomb_map.get_rates()

    # Get indices for the start and end of the recombination map.
    j = 0
    while j < len(rates) and rates[j] == 0:
        j += 1
    k = len(positions) - 1
    while k > 1 and rates[k - 1] == 0:
        k -= 1

    assert j <= k
    if positions[k] - positions[j] < length:
        # TODO: should really check this when we enumerate chromosomes
        raise ValueError(
            f"{chrom.id} was sampled, but its recombination map is "
            f"shorter than {length}."
        )

    pos = rng.randrange(positions[j], positions[k] - length)
    # new_map = recomb_map[pos:pos+length]  # requires msprime >= 1.0
    new_map = recomb_slice(recomb_map, start=pos, end=pos + length)
    origin = f"{chrom.id}:{pos}-{pos+length}"
    contig = MyContig(
        recombination_map=new_map,
        mutation_rate=chrom.mutation_rate,
        genetic_map=gm,
        origin=origin,
    )

    return contig


def KimDFE():
    """
    Return neutral and negative MutationType()s representing a human DFE.
    Kim et al. (2018), p.23, http://doi.org/10.1371/journal.pgen.1007741
    """
    neutral = stdpopsim.ext.MutationType(weight=1.0)
    gamma_shape = 0.186  # shape
    gamma_mean = -0.01314833  # expected value
    h = 0.5/(1-7071.07*gamma_mean)  # dominance coefficient
    negative = stdpopsim.ext.MutationType(
            weight=2.31, dominance_coeff=h, distribution_type="g",
            distribution_args=[gamma_mean, gamma_shape])
    return [neutral, negative]


def hominin_composite():
    id = "HomininComposite_4G20"
    description = "Four population out of Africa with Neandertal admixture"
    long_description = """
                A composite of demographic parameters from multiple sources
                """
    # samples:
    # T_Altai = 115e3
    # T_Vindija = 55e3
    # n_YRI = 108
    # n_CEU = 99

    populations = [
        stdpopsim.Population(id="YRI", description="1000 Genomes YRI (Yorubans)"),
        stdpopsim.Population(
            id="CEU",
            description=(
                "1000 Genomes CEU (Utah Residents (CEPH) with Northern and "
                "Western European Ancestry"
            ),
        ),
        stdpopsim.Population(id="Nea", description="Neandertal lineage"),
        stdpopsim.Population(
            id="Anc", description="Ancestral hominins", sampling_time=None
        ),
    ]
    pop = {p.id: i for i, p in enumerate(populations)}

    citations = [
        stdpopsim.Citation(
            author="Kuhlwilm et al.",
            year=2016,
            doi="https://doi.org/10.1038/nature16544",
        ),
        stdpopsim.Citation(
            author="Prüfer et al.",
            year=2017,
            doi="https://doi.org/10.1126/science.aao1887",
        ),
        stdpopsim.Citation(
            author="Ragsdale and Gravel",
            year=2019,
            doi="https://doi.org/10.1371/journal.pgen.1008204",
        ),
    ]

    generation_time = 29

    # Kuhlwilm et al. 2016
    N_YRI = 27000
    N_Nea = 3400
    N_Anc = 18500

    # Ragsdale & Gravel 2019
    N_CEU0 = 1450
    r_CEU = 0.00202
    T_CEU_exp = 31.9e3 / generation_time
    N_CEU = N_CEU0 * math.exp(r_CEU * T_CEU_exp)
    T_YRI_CEU_split = 65.7e3 / generation_time
    N_ooa_bottleneck = 1080

    # Prüfer et al. 2017
    T_Nea_human_split = 550e3 / generation_time
    T_Nea_CEU_mig = 55e3 / generation_time
    m_Nea_CEU = 0.0225

    pop_meta = (p.asdict() for p in populations)
    population_configurations = [
        msprime.PopulationConfiguration(initial_size=N_YRI, metadata=next(pop_meta)),
        msprime.PopulationConfiguration(
            initial_size=N_CEU, growth_rate=r_CEU, metadata=next(pop_meta)
        ),
        msprime.PopulationConfiguration(initial_size=N_Nea, metadata=next(pop_meta)),
        msprime.PopulationConfiguration(initial_size=N_Anc, metadata=next(pop_meta)),
    ]

    demographic_events = [
        # out-of-Africa bottleneck
        msprime.PopulationParametersChange(
            time=T_CEU_exp,
            initial_size=N_ooa_bottleneck,
            growth_rate=0,
            population_id=pop["CEU"],
        ),
        # Neandertal -> CEU admixture
        msprime.MassMigration(
            time=T_Nea_CEU_mig,
            proportion=m_Nea_CEU,
            source=pop["CEU"],
            destination=pop["Nea"],
        ),
        # population splits
        msprime.MassMigration(
            time=T_YRI_CEU_split, source=pop["CEU"], destination=pop["Anc"]
        ),
        msprime.MassMigration(
            time=T_YRI_CEU_split, source=pop["YRI"], destination=pop["Anc"]
        ),
        msprime.MassMigration(
            time=T_Nea_human_split, source=pop["Nea"], destination=pop["Anc"]
        ),
    ]

    return stdpopsim.DemographicModel(
        id=id,
        description=description,
        long_description=long_description,
        populations=populations,
        citations=citations,
        generation_time=generation_time,
        population_configurations=population_configurations,
        demographic_events=demographic_events,
    )


def homsap_composite_model(length, sample_counts, seed):
    if "Nea" in sample_counts and sample_counts["Nea"] != 4:
        raise RuntimeError(
            "Must have one sample each for the Vindija and Altai Neanderthals"
        )
    species = stdpopsim.get_species("HomSap")
    model = hominin_composite()
    contig = random_autosomal_chunk(species, "HapMapII_GRCh37", length, seed)
    samples = model.get_samples(
        *[sample_counts.get(p.id, 0) if p.id != "Nea" else 0 for p in model.populations]
    )
    if "Nea" in sample_counts:
        # Altai and Vindija Neanderthal dates from Prüfer et al. 2017.
        T_Altai = 115e3 / model.generation_time
        T_Vindija = 55e3 / model.generation_time
        pop = {p.id: i for i, p in enumerate(model.populations)}
        samples.extend(
            [
                msprime.Sample(pop["Nea"], T_Altai),
                msprime.Sample(pop["Nea"], T_Altai),
                msprime.Sample(pop["Nea"], T_Vindija),
                msprime.Sample(pop["Nea"], T_Vindija),
            ]
        )
    return species, model, contig, samples


def homsap_composite_Nea_to_CEU(
    model,
    contig,
    samples,
    seed,
    dfe=False,
    slim_script=False,
    min_allele_frequency=0,
    logunif_s=True,
    s_lo=0.001,
    s_hi=0.1,
    **kwargs,
):
    rng = random.Random(seed)
    pop = {p.id: i for i, p in enumerate(model.populations)}

    mutation_types = []
    if dfe:
        mutation_types.extend(KimDFE())
    positive = stdpopsim.ext.MutationType(convert_to_substitution=False)
    mutation_types.append(positive)
    mut_id = len(mutation_types)

    T_Nea_human_split = contact.split_time(model, pop["Nea"], pop["CEU"])
    T_Nea_CEU_mig = contact.tmrca(model, pop["Nea"], pop["CEU"])

    t_delta = 1e3 / model.generation_time
    T_mut = rng.uniform(T_Nea_CEU_mig + t_delta, T_Nea_human_split)
    T_sel = rng.uniform(t_delta, T_Nea_CEU_mig)
    if logunif_s:
        # uniform on a log scale
        s = math.exp(rng.uniform(math.log(s_lo), math.log(s_hi)))
    else:
        s = rng.uniform(s_lo, s_hi)

    coordinate = round(contig.recombination_map.get_length() / 2)

    extended_events = [
        # Draw mutation in Neanderthals.
        stdpopsim.ext.DrawMutation(
            time=T_mut,
            mutation_type_id=mut_id,
            population_id=pop["Nea"],
            coordinate=coordinate,
            # Save state before the mutation is introduced
            save=True,
        ),
        # Mutation is positively selected in CEU
        stdpopsim.ext.ChangeMutationFitness(
            start_time=T_sel,
            end_time=0,
            mutation_type_id=mut_id,
            population_id=pop["CEU"],
            selection_coeff=s,
            dominance_coeff=0.5,
        ),
        # Allele frequency conditioning. If the condition is not met, we
        # restore to the most recent save point.
        stdpopsim.ext.ConditionOnAlleleFrequency(
            # FIXME: GenerationAfter(T_mut) < T_Nea_CEU_mig
            start_time=stdpopsim.ext.GenerationAfter(T_mut),
            end_time=T_Nea_CEU_mig,
            mutation_type_id=mut_id,
            population_id=pop["Nea"],
            op=">",
            allele_frequency=0,
        ),
        stdpopsim.ext.ConditionOnAlleleFrequency(
            start_time=stdpopsim.ext.GenerationAfter(T_Nea_CEU_mig),
            end_time=0,
            mutation_type_id=mut_id,
            population_id=pop["CEU"],
            op=">",
            allele_frequency=0,
            # Update save point at start_time.
            save=True,
        ),
        stdpopsim.ext.ConditionOnAlleleFrequency(
            start_time=0,
            end_time=0,
            mutation_type_id=mut_id,
            population_id=pop["CEU"],
            op=">",
            allele_frequency=min_allele_frequency,
        ),
    ]

    engine = stdpopsim.get_engine("slim")
    ts = engine.simulate(
        model,
        contig,
        samples,
        seed=seed,
        mutation_types=mutation_types,
        extended_events=extended_events,
        slim_script=slim_script,
        slim_burn_in=10 if dfe else 0.1,
        slim_scaling_factor=10,
    )

    return (
        ts,
        (
            contig.origin,
            T_mut * model.generation_time,
            T_sel * model.generation_time,
            s,
        ),
    )


def homsap_composite_Sweep_CEU(
    model,
    contig,
    samples,
    seed,
    dfe=False,
    slim_script=False,
    min_allele_frequency=0,
    logunif_s=True,
    s_lo=0.001,
    s_hi=0.1,
    **kwargs,
):
    rng = random.Random(seed)
    pop = {p.id: i for i, p in enumerate(model.populations)}

    mutation_types = []
    if dfe:
        mutation_types.extend(KimDFE())
    positive = stdpopsim.ext.MutationType(convert_to_substitution=False)
    mutation_types.append(positive)
    mut_id = len(mutation_types)

    T_YRI_CEU_split = contact.split_time(model, pop["YRI"], pop["CEU"])

    T_sel = rng.uniform(1e3 / model.generation_time, T_YRI_CEU_split)
    T_mut = rng.uniform(T_sel, T_YRI_CEU_split)
    if logunif_s:
        # uniform on a log scale
        s = math.exp(rng.uniform(math.log(s_lo), math.log(s_hi)))
    else:
        s = rng.uniform(s_lo, s_hi)

    coordinate = round(contig.recombination_map.get_length() / 2)

    extended_events = [
        # Draw mutation.
        stdpopsim.ext.DrawMutation(
            time=T_mut,
            mutation_type_id=mut_id,
            population_id=pop["CEU"],
            coordinate=coordinate,
            # Save state before the mutation is introduced.
            save=True,
        ),
        # Mutation is positively selected at time T_sel.
        stdpopsim.ext.ChangeMutationFitness(
            start_time=T_sel,
            end_time=0,
            mutation_type_id=mut_id,
            population_id=pop["CEU"],
            selection_coeff=s,
            dominance_coeff=0.5,
        ),
        # Allele frequency conditioning. If the condition is not met, we
        # restore to the save point.
        stdpopsim.ext.ConditionOnAlleleFrequency(
            start_time=stdpopsim.ext.GenerationAfter(T_mut),
            end_time=0,
            mutation_type_id=mut_id,
            population_id=pop["CEU"],
            op=">",
            allele_frequency=0,
        ),
        stdpopsim.ext.ConditionOnAlleleFrequency(
            start_time=0,
            end_time=0,
            mutation_type_id=mut_id,
            population_id=pop["CEU"],
            op=">",
            allele_frequency=min_allele_frequency,
        ),
    ]

    engine = stdpopsim.get_engine("slim")
    ts = engine.simulate(
        model,
        contig,
        samples,
        seed=seed,
        mutation_types=mutation_types,
        extended_events=extended_events,
        slim_script=slim_script,
        slim_burn_in=10 if dfe else 0.1,
        slim_scaling_factor=10,
    )

    return (
        ts,
        (
            contig.origin,
            T_mut * model.generation_time,
            T_sel * model.generation_time,
            s,
        ),
    )


def homsap_papuans_model(length, sample_counts, seed):
    species = stdpopsim.get_species("HomSap")
    model = species.get_demographic_model("PapuansOutOfAfrica_10J19")
    contig = random_autosomal_chunk(species, "HapMapII_GRCh37", length, seed)
    samples = model.get_samples(
        *[sample_counts.get(p.id, 0) for p in model.populations]
    )
    return species, model, contig, samples


def generic_Neutral(model, contig, samples, seed, engine="slim", **kwargs):
    engine = stdpopsim.get_engine(engine)
    ts = engine.simulate(
        model,
        contig,
        samples,
        seed=seed,
        slim_burn_in=0.1,
        slim_scaling_factor=10,
    )
    return ts, (contig.origin, 0, 0, 0)


def homsap_DFE(model, contig, samples, seed, **kwargs):
    mutation_types = KimDFE()
    engine = stdpopsim.get_engine("slim")
    ts = engine.simulate(
        model,
        contig,
        samples,
        seed=seed,
        mutation_types=mutation_types,
        slim_burn_in=10,
        slim_scaling_factor=10,
    )
    return ts, (contig.origin, 0, 0, 0)


def homsap_papuans_AI_Den_to_Papuan(
    model,
    contig,
    samples,
    seed,
    dfe=False,
    Den="Den1",
    slim_script=False,
    min_allele_frequency=0,
    logunif_s=True,
    s_lo=0.001,
    s_hi=0.1,
    **kwargs,
):
    rng = random.Random(seed)

    if Den not in ("Den1", "Den2"):
        raise ValueError("Source population Den must be either Den1 or Den2.")

    pop = {p.id: i for i, p in enumerate(model.populations)}

    mutation_types = []
    if dfe:
        mutation_types.extend(KimDFE())
    positive = stdpopsim.ext.MutationType(convert_to_substitution=False)
    mutation_types.append(positive)
    mut_id = len(mutation_types)

    T_Den_Nea_split = contact.tmrca(model, pop["DenA"], pop["NeaA"])
    T_DenA_Den1_split = contact.tmrca(model, pop["DenA"], pop["Den1"])
    T_DenA_Den2_split = contact.tmrca(model, pop["DenA"], pop["Den2"])
    T_Den1_Papuan_mig = contact.tmrca(model, pop["Papuan"], pop["Den1"])
    T_Den2_Papuan_mig = contact.tmrca(model, pop["Papuan"], pop["Den2"])

    if Den == "Den1":
        T_Den_split = T_DenA_Den1_split
        T_mig = T_Den1_Papuan_mig
    else:
        T_Den_split = T_DenA_Den2_split
        T_mig = T_Den2_Papuan_mig

    t_delta = 1e3 / model.generation_time
    T_mut = rng.uniform(T_Den_split + t_delta, T_Den_Nea_split)
    T_sel = rng.uniform(t_delta, T_mig)
    if logunif_s:
        # uniform on a log scale
        s = math.exp(rng.uniform(math.log(s_lo), math.log(s_hi)))
    else:
        s = rng.uniform(s_lo, s_hi)

    coordinate = round(contig.recombination_map.get_length() / 2)

    extended_events = [
        # Draw mutation in Denisovans.
        stdpopsim.ext.DrawMutation(
            time=T_mut,
            mutation_type_id=mut_id,
            population_id=pop["DenA"],
            coordinate=coordinate,
            # Save state before the mutation is introduced
            save=True,
        ),
        # Mutation is positively selected in Papuans
        stdpopsim.ext.ChangeMutationFitness(
            start_time=T_sel,
            end_time=0,
            mutation_type_id=mut_id,
            population_id=pop["Papuan"],
            selection_coeff=s,
            dominance_coeff=0.5,
        ),
        # Allele frequency conditioning. If the condition is not met, we
        # restore to the most recent save point.
        stdpopsim.ext.ConditionOnAlleleFrequency(
            # FIXME: GenerationAfter(T_mut) < T_Den_split
            start_time=stdpopsim.ext.GenerationAfter(T_mut),
            end_time=T_Den_split,
            mutation_type_id=mut_id,
            population_id=pop["DenA"],
            op=">",
            allele_frequency=0,
        ),
        stdpopsim.ext.ConditionOnAlleleFrequency(
            start_time=stdpopsim.ext.GenerationAfter(T_Den_split),
            end_time=T_mig,
            mutation_type_id=mut_id,
            population_id=pop[Den],
            op=">",
            allele_frequency=0,
            # Update save point at start_time.
            save=True,
        ),
        stdpopsim.ext.ConditionOnAlleleFrequency(
            start_time=stdpopsim.ext.GenerationAfter(T_mig),
            end_time=0,
            mutation_type_id=mut_id,
            population_id=pop["Papuan"],
            op=">",
            allele_frequency=0,
            # Update save point at start_time.
            save=True,
        ),
        stdpopsim.ext.ConditionOnAlleleFrequency(
            start_time=0,
            end_time=0,
            mutation_type_id=mut_id,
            population_id=pop["Papuan"],
            op=">",
            allele_frequency=min_allele_frequency,
        ),
    ]

    engine = stdpopsim.get_engine("slim")
    ts = engine.simulate(
        model,
        contig,
        samples,
        seed=seed,
        mutation_types=mutation_types,
        extended_events=extended_events,
        slim_script=slim_script,
        slim_burn_in=10 if dfe else 0.1,
        slim_scaling_factor=10,
    )

    return (
        ts,
        (
            contig.origin,
            T_mut * model.generation_time,
            T_sel * model.generation_time,
            s,
        ),
    )


def homsap_papuans_Sweep_Papuan(
    model,
    contig,
    samples,
    seed,
    dfe=False,
    slim_script=False,
    min_allele_frequency=0,
    logunif_s=True,
    s_lo=0.001,
    s_hi=0.1,
    **kwargs,
):
    rng = random.Random(seed)

    pop = {p.id: i for i, p in enumerate(model.populations)}

    mutation_types = []
    if dfe:
        mutation_types.extend(KimDFE())
    positive = stdpopsim.ext.MutationType(convert_to_substitution=False)
    mutation_types.append(positive)
    mut_id = len(mutation_types)

    T_Papuan_Ghost_split = contact.split_time(model, pop["Papuan"], pop["Ghost"])

    T_sel = rng.uniform(1e3 / model.generation_time, T_Papuan_Ghost_split)
    T_mut = rng.uniform(T_sel, T_Papuan_Ghost_split)
    if logunif_s:
        # uniform on a log scale
        s = math.exp(rng.uniform(math.log(s_lo), math.log(s_hi)))
    else:
        s = rng.uniform(s_lo, s_hi)

    coordinate = round(contig.recombination_map.get_length() / 2)

    extended_events = [
        # Draw mutation.
        stdpopsim.ext.DrawMutation(
            time=T_mut,
            mutation_type_id=mut_id,
            population_id=pop["Papuan"],
            coordinate=coordinate,
            # Save state before the mutation is introduced.
            save=True,
        ),
        # Mutation is positively selected at time T_sel.
        stdpopsim.ext.ChangeMutationFitness(
            start_time=T_sel,
            end_time=0,
            mutation_type_id=mut_id,
            population_id=pop["Papuan"],
            selection_coeff=s,
            dominance_coeff=0.5,
        ),
        # Allele frequency conditioning. If the condition is not met, we
        # restore to the save point.
        stdpopsim.ext.ConditionOnAlleleFrequency(
            start_time=stdpopsim.ext.GenerationAfter(T_mut),
            end_time=0,
            mutation_type_id=mut_id,
            population_id=pop["Papuan"],
            op=">",
            allele_frequency=0,
        ),
        stdpopsim.ext.ConditionOnAlleleFrequency(
            start_time=0,
            end_time=0,
            mutation_type_id=mut_id,
            population_id=pop["Papuan"],
            op=">",
            allele_frequency=min_allele_frequency,
        ),
    ]

    engine = stdpopsim.get_engine("slim")
    ts = engine.simulate(
        model,
        contig,
        samples,
        seed=seed,
        mutation_types=mutation_types,
        extended_events=extended_events,
        slim_script=slim_script,
        slim_burn_in=10 if dfe else 0.1,
        slim_scaling_factor=10,
    )

    return (
        ts,
        (
            contig.origin,
            T_mut * model.generation_time,
            T_sel * model.generation_time,
            s,
        ),
    )


@attr.s(kw_only=True)
class ModelSpec:
    # Base demographic model.
    model_func = attr.ib()
    # Various mutation models to stack on top of the demographic model.
    # TODO: these probably need a description field.
    scenarios = attr.ib()


_simulations = {
    "HomSap/PapuansOutOfAfrica_10J19": ModelSpec(
        model_func=homsap_papuans_model,
        scenarios={
            # Various mutation models to stack on top of the demographic model.
            "Neutral/slim": functools.partial(generic_Neutral, engine="slim"),
            "Neutral/msprime": functools.partial(generic_Neutral, engine="msprime"),
            "DFE": homsap_DFE,
            "AI/Den1_to_Papuan": functools.partial(
                homsap_papuans_AI_Den_to_Papuan, Den="Den1"
            ),
            "AI/Den2_to_Papuan": functools.partial(
                homsap_papuans_AI_Den_to_Papuan, Den="Den2"
            ),
            "Sweep/Papuan": homsap_papuans_Sweep_Papuan,
        },
    ),
    "HomSap/HomininComposite_4G20": ModelSpec(
        model_func=homsap_composite_model,
        scenarios={
            "Neutral/slim": functools.partial(generic_Neutral, engine="slim"),
            "Neutral/msprime": functools.partial(generic_Neutral, engine="msprime"),
            "DFE": homsap_DFE,
            "AI/Nea_to_CEU": functools.partial(homsap_composite_Nea_to_CEU, s_lo=1e-4),
            "Sweep/CEU": homsap_composite_Sweep_CEU,
        },
    ),
}


def _models(mdict=_simulations):
    models = {}
    for prefix, spec in mdict.items():
        assert callable(spec.model_func)
        for scenario, sim_func in spec.scenarios.items():
            assert callable(sim_func)
            modelspec = f"{prefix}/{scenario}"
            models[modelspec] = (spec.model_func, sim_func)
    return models


def get_demog_model(modelspec, sequence_length=100000):
    models = _models()
    for model, (model_func, sim_func) in models.items():
        if modelspec == model:
            break
    else:
        raise ValueError(f"{modelspec} not found")

    _, model, _, _ = model_func(sequence_length, {}, 1234)
    return model


def sim(
    modelspec,
    *,
    sequence_length,
    min_allele_frequency,
    sample_counts,
    seed=None,
    slim_script=False,
    command=None,
):
    models = _models()
    for model, (model_func, sim_func) in models.items():
        if modelspec == model:
            break
    else:
        raise ValueError(f"{modelspec} not found")

    # Do simulation.
    species, model, contig, samples = model_func(sequence_length, sample_counts, seed)
    ts, (origin, T_mut, T_sel, s) = sim_func(
        model,
        contig,
        samples,
        seed,
        slim_script=slim_script,
        min_allele_frequency=min_allele_frequency,
    )
    if ts is None:
        return None

    popid = {i: p.id for i, p in enumerate(model.populations)}
    observed_counts = collections.Counter(
        [popid[ts.get_population(i)] for i in ts.samples()]
    )
    assert observed_counts == sample_counts, f"{observed_counts} != {sample_counts}"

    # Add provenance.
    ts = provenance.dedup_slim_provenances(ts)
    params = dict(
        seed=seed,
        modelspec=modelspec,
        origin=origin,
        T_mut=T_mut,
        T_sel=T_sel,
        s=s,
        min_allele_frequency=min_allele_frequency,
    )
    if command is not None:
        params["command"] = command
    ts = provenance.save_parameters(ts, **params)

    return ts
