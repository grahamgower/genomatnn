import json

import kastore


def save_parameters(ts, name, version, **parameters):
    """
    Save parameters to a tskit provenance record.
    """
    provenance = {
        "schema_version": "1.0.0",
        "software": {"name": name, "version": version},
        "parameters": parameters,
        # "environment": tskit.provenance.get_environment()
    }

    tables = ts.dump_tables()
    # tskit.validate_provenance(provenance)
    tables.provenances.add_row(json.dumps(provenance))
    ts = tables.tree_sequence()
    return ts


def load_parameters(ts, name):
    """
    Load parameters from `name` in the provenance record.
    """
    for p in ts.provenances():
        d = json.loads(p.record)
        if d["software"]["name"] == name:
            return d.get("parameters")
    return None


def load_parameters_from_file(filename, name):
    """
    Load parameters from `name` in the provenance record of `filename`.

    Using tskit can be slow when loading provenance records for a lot of files.
    We deliberately bypass the tskit API here, so that we don't unnecessarily
    load and parse all of the treesequence tables.
    """
    ka = kastore.load(filename)
    record_offset = ka["provenances/record_offset"]
    i = record_offset[0]
    for j in record_offset[1:]:
        record = ka["provenances/record"][i:j].tostring()
        d = json.loads(record)
        if d["software"]["name"] == name:
            return d.get("parameters")
        i = j
    return None


def dedup_slim_provenances(ts):
    """
    Remove redundant copies of SLiM's provenance records.

    Duplicate SLiM provenance records may result from sim.treeSeqOutput()
    followed by sim.readFromPopulationFile(), as is common when ensuring that
    an introduced mutation is not lost.
    """
    tables = ts.dump_tables()
    tables.provenances.clear()

    # Find the last SLiM provenance entry.
    i = -1
    for p in ts.provenances():
        d = json.loads(p.record)
        if d["software"]["name"] == "SLiM":
            i += 1

    if i == -1:
        # print("SLiM provenance not found", file=sys.stderr)
        return ts

    for j, p in enumerate(ts.provenances()):
        if j < i:
            continue
        tables.provenances.add_row(p.record, p.timestamp)

    return tables.tree_sequence()
