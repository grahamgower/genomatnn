import gzip
import itertools
import operator
import functools
import urllib.parse

import numpy as np
import ncls


class Intervals:
    """
    A class for manipulating GFF3-like intervals.
    This is a wrapper around a numpy record array, with some convenience
    functions for doing set operations on closed integer intervals.
    """

    def __init__(self, data):
        if not (
            isinstance(data, np.recarray)
            and hasattr(data, "start")
            and hasattr(data, "end")
        ):
            raise ValueError(
                "data must be a numpy recarray with start and end attributes"
            )
        self.data = data
        self.data_fields = set(data.dtype.names)
        self._ncls = None

    def __len__(self):
        return len(self.data)

    def __getattr__(self, key):
        # Use attributes of self.data as if they were attributes of this class.
        return getattr(self.data, key)

    def __iter__(self):
        # Iterate over self.data.
        return iter(self.data)

    def __getitem__(self, key):
        # Use indexes of self.data as if they were indexes of this class.
        return self.data[key]

    @property
    def ncls(self):
        """
        Nested Containment List for the intervals.
        """
        if self._ncls is None:
            start = self.start
            end = self.end
            # arrays must be c-contiguous for ncls, so copy if they're not
            if not start.flags["C_CONTIGUOUS"]:
                start = np.array(start)
            if not end.flags["C_CONTIGUOUS"]:
                end = np.array(end)
            self._ncls = ncls.NCLS(start, end, np.arange(len(start)))
        return self._ncls

    def overlap(self, start, end, yield_all=False):
        """
        Find all intervals that overlap the query interval(s) ``[start, end]``.
        ``start`` and ``end`` may be single numbers, or numpy arrays.
        Returns a generator yielding a tuple (qi, record_idx), where qi is the
        index of the query interval and record_idx is a list of indexes record
        indexes that overlaps the query interval. If ``yield_all=True``, empty
        lists will also be generated for query intervals with no overlaps.
        """
        try:
            len(start)
        except TypeError:
            start = np.array([start])
            end = np.array([end])
        assert len(start) == len(end)
        # arrays must be c-contiguous for ncls, so copy if they're not
        if not start.flags["C_CONTIGUOUS"]:
            start = np.array(start)
        if not end.flags["C_CONTIGUOUS"]:
            end = np.array(end)
        indexes = np.arange(len(start))
        query_idx, self_idx = self.ncls.all_overlaps_both(start, end, indexes)
        last_qi = 0
        for qi, groups in itertools.groupby(
            zip(query_idx, self_idx), operator.itemgetter(0)
        ):
            if yield_all:
                while last_qi != qi:
                    yield last_qi, []
                    last_qi += 1
            yield qi, [idx[1] for idx in groups]
        if yield_all:
            qi = len(start) - 1
            while last_qi != qi:
                yield last_qi, []
                last_qi += 1

    def _iter_merge(self):
        """
        Generator of ``[start, end]`` coordinates for merged intervals.

        This only does the correct thing for closed intervals.
        """
        if len(self) == 0:
            return
        start = None
        end = -np.inf
        pending = True
        for record in sorted(self, key=operator.itemgetter("start")):
            if record.start - 1 <= end:
                # intersects (or is contiguous with) the previous record(s)
                end = max(end, record.end)
                pending = True
            else:
                # this record does not intersect with the previous record(s)
                if end > 0:
                    yield start, end
                    pending = False
                start, end = record.start, record.end
        if pending:
            yield start, end

    def merge(self):
        """
        Returns a new instance containing merged intervals.

        This only does the correct thing for closed intervals.
        """
        records = list(self._iter_merge())
        recarray = np.rec.fromrecords(records, dtype=[("start", int), ("end", int)])
        return self.__class__(recarray)

    def subset(self, start=None, end=None, **kwargs):
        """
        Returns a new instance containing a subset of intervals.
        If the ``start`` and/or ``end``
        coordinates are specified, only intervals in the ``[start, end]``
        genomic window are generated. Intevals overlapping this window
        are clipped to the window boundaries.
        Additional keyword arguments can be specified to obtain subsets
        by filtering for exact matches on other data fields. E.g.
            >>> gff.subset(source="ensembl", type="exon")
        """
        if start is None:
            start = 1
        if end is None:
            end = np.inf
        records = self.data
        if len(kwargs) > 0:
            inc_filters = []
            for key, val in kwargs.items():
                if key not in self.data_fields:
                    raise ValueError(f"unknown record field '{key}'")
                inc_filters.append(self.data[key] == val)
            indices = np.where(np.all(inc_filters, axis=0))
            records = self.data[indices]
        indices = np.where(np.logical_and(records.end >= start, records.start <= end))
        records = records[indices]
        indices = np.where(np.logical_or(records.start < start, records.end > end))
        records[indices] = records[indices].copy()
        records[indices].start = np.maximum(records[indices].start, start)
        records[indices].end = np.minimum(records[indices].end, end)
        return self.__class__(records)

    def dump(self, file):
        """
        Print tab-separated fields to the file object ``file``.
        """
        for record in self:
            print(*record, sep="\t", file=file)

    @classmethod
    def fromrecords(cls, records):
        """
        Returns a new instance from ``records``, which is a list of 2-tuples.
        """
        recarray = np.rec.fromrecords(records, dtype=[("start", int), ("end", int)])
        return cls(recarray)


def parse_gff3_attributes(attr):
    """
    Split semicolon delimited attributes into a dictionary.
    """
    d = dict()
    fields = attr.split(";")
    for field in fields:
        subfields = field.split("=")
        if len(subfields) == 1:
            d[field] = True
        elif len(subfields) == 2:
            k, v = subfields
            d[k] = urllib.parse.unquote(v)
        else:
            raise ValueError(f"unknown field {field} in annotation {attr}")
    return d


def load_gff3(filename, **kwargs):
    """
    Load a GFF3 file, filtering on equality of provided keyword args (if any).
    E.g. to load only ensembl_havana annotations for chromosome 1:
        >>> load_gff3("/path/to/file.gff", source="ensembl_havana", seqid="1")

    GFF3 format references:
    https://www.ensembl.org/info/website/upload/gff.html
    http://gmod.org/wiki/GFF3
    """
    field_types = [
        # field name, numpy dtype
        ("seqid", object),
        ("source", object),
        ("type", object),
        ("start", int),
        ("end", int),
        ("score", object),
        ("strand", object),
        ("phase", object),
        ("attributes", object),
    ]
    field_names = set([name for name, _ in field_types])
    field_names -= set(["start", "end"])  # TODO support these? via tabix?
    keep = lambda record: True  # noqa:E731
    if len(kwargs) > 0:
        for key in kwargs:
            if key not in field_names:
                raise ValueError(f"unsupported filter key '{key}'")
        indices = [i for i, (key, _) in enumerate(field_types) if key in kwargs]
        getter = operator.itemgetter(*indices)
        values = tuple(kwargs[field_types[i][0]] for i in indices)
        if len(values) == 1:
            values = values[0]
        keep = lambda record: getter(record) == values  # noqa:E731

    xopen = functools.partial(gzip.open, mode="rt")
    try:
        xopen(filename).read(1)
    except OSError:
        xopen = open
    records = []
    with xopen(filename) as f:
        for lineno, line in enumerate(f, 1):
            line = line.rstrip()
            if line[0] == "#":
                continue
            record = line.split("\t")
            if len(record) != len(field_types):
                raise RuntimeError(
                    f"{filename}: {lineno}: expected {len(field_types)} "
                    f"tab-separated columns, but found {len(record)}"
                )
            record[3] = int(record[3])
            record[4] = int(record[4])
            if keep(record):
                records.append(tuple(record))
    records = np.rec.fromrecords(records, dtype=field_types)
    return Intervals(records)


def parse_predictions(pred_file, max_chr_name_len=128):
    chr_dtype = f"U{max_chr_name_len}"
    with open(pred_file) as f:
        # header is: chrom, start, end, Pr(X), ...
        header = next(f).split()
        dtype = [(header[0], chr_dtype), (header[1], int), (header[2], int)] + [
            (h, float) for h in header[3:]
        ]
        preds = np.loadtxt(f, dtype=dtype)
    records = np.rec.array(preds)
    return header, records


def annotate(*, gff_file, pred_file, file, x=None, n=None, pad=100000):
    if x is None and n is None:
        raise ValueError("Specify either top x proportion of hits or top n hits")

    genes = load_gff3(gff_file, source="ensembl_havana", type="gene")
    preds_header, preds = parse_predictions(pred_file)

    # sort predictions by p-value
    p = preds_header[3]  # Pr(X) label
    idx = np.argsort(preds[p])[::-1]

    # take only top predictions
    if x is not None:
        # annotate top x proportion of hits
        n = int(len(idx) * x)
    top_preds = preds[idx[:n]]

    # extend in both directions by `pad`
    top_preds.start -= pad
    top_preds.end += pad

    chroms = np.unique(top_preds.chrom).tolist()
    for chrom in sorted(chroms, key=int):
        chr_preds = top_preds[np.where(top_preds.chrom == chrom)]
        records = Intervals(chr_preds).merge()
        chr_genes = genes.subset(seqid=chrom)

        for j, gene_idx in chr_genes.overlap(records.start, records.end):
            record = records[j]
            gene_names = []
            for j in gene_idx:
                attr = parse_gff3_attributes(chr_genes[j].attributes)
                gene_names.append(attr.get("Name"))
            print(
                chrom,
                record.start,
                record.end,
                "; ".join(gene_names),
                sep="\t",
                file=file,
            )


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 4:
        print(f"usage: {sys.argv[0]} file.gff3 predictions.txt x")
        exit(1)

    gff_file = sys.argv[1]
    pred_file = sys.argv[2]
    x = float(sys.argv[3])

    if x <= 0:
        raise ValueError("x must be greater than zero")
    if x < 1:
        # annotate top x proportion of hits
        n = None
    else:
        # annotate top n hits
        n = round(x)
        x = None

    annotate(gff_file=gff_file, pred_file=pred_file, file=sys.stdout, n=n, x=x)
