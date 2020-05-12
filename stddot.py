#!/usr/bin/env python3

import sys

import networkx as nx
import msprime
import stdpopsim


def digraph(model):
    """
    Construct a networkx digraph from a stdpopsim.DemographicModel object.
    """
    G = nx.DiGraph()

    for pop in model.populations:
        G.add_node(pop.id, time=pop.sampling_time, color="red", shape="rect")

    parent = dict()

    def top_of_lineage(x):
        while x in parent:
            x = parent[x]
        return x

    prev_event_time = 0
    for i, event in enumerate(model.demographic_events):
        if event.time < prev_event_time:
            raise ValueError(
                    "demographic events must be sorted in time-ascending order")
        prev_event_time = event.time

        if isinstance(event, msprime.MassMigration):
            source = top_of_lineage(model.populations[event.source].id)
            dest = top_of_lineage(model.populations[event.dest].id)

            t_source = G.nodes[source].get("time")
            if t_source is None:
                t_source = event.time
                G.nodes[source].update(
                        time=t_source, label=f"{source}\ntime={t_source:.0f}")
            t_dest = G.nodes[dest].get("time")
            if t_dest is None:
                t_dest = event.time
                G.nodes[dest].update(
                        time=t_dest, label=f"{dest}\ntime={t_dest:.0f}")

            if event.proportion == 1:
                if t_dest < event.time:
                    new = dest + "/^"
                    G.add_node(
                            new, time=event.time,
                            label=f"{new}\ntime={event.time:.0f}")
                    G.add_edge(new, dest)
                    parent[dest] = new
                    dest = new
                G.add_edge(dest, source)
                parent[source] = dest
            else:
                if t_source < event.time:
                    new = source + "/^"
                    G.add_node(new, time=event.time, shape="point")
                    G.add_edge(new, source)
                    parent[source] = new
                    source = new
                if t_dest < event.time:
                    new = dest + "/^"
                    G.add_node(new, time=event.time, shape="point")
                    G.add_edge(new, dest)
                    parent[dest] = new
                    dest = new
                G.add_edge(
                        dest, source, style="dotted", color="red",
                        label=f"{event.proportion:.3g}")

    assert nx.is_directed_acyclic_graph(G), \
        "Cycle detected. Please report this bug, and include the " \
        "demographic model that triggered this error."

    return G


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} modelspec", file=sys.stderr)
        exit(1)

    modelspec = sys.argv[1]
    assert modelspec.count("/") == 1, f"Unexpected modelspec {modelspec}"
    species_str, model_str = modelspec.split("/")
    species = stdpopsim.get_species(species_str)
    model = species.get_demographic_model(model_str)

    G = digraph(model)

    # Put all extant populations at the same 'rank'.
    A = nx.nx_agraph.to_agraph(G)
    extant = [p.id for p in model.populations if p.sampling_time == 0]
    A.add_subgraph(extant, rank="same")
    A.write(sys.stdout)
