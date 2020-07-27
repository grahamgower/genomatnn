import itertools
import subprocess
import collections
import tempfile

import numpy as np


def sample_phasing(vcf, sample_list):
    """
    Determine if the genotypes in ``vcf`` are phased for the ``sample_list``.
    """
    phasing_unknown = set(range(len(sample_list)))
    phasing = [True] * len(sample_list)

    with tempfile.TemporaryDirectory() as tmpdir:
        samples_file = f"{tmpdir}/samples.txt"
        with open(samples_file, "w") as f:
            print(*sample_list, file=f, sep="\n")

        cmd = ["bcftools", "query", "-S", samples_file, "-f", "[%GT\t]\\n", vcf]

        with subprocess.Popen(
            cmd,
            bufsize=1,
            universal_newlines=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ) as p:
            for line in p.stdout:
                gt_str_list = line.split()
                for i, gt_str in enumerate(gt_str_list):
                    if gt_str[0] == ".":
                        # phasing unknown
                        continue
                    if gt_str[1] == "/":
                        phasing[i] = False
                    elif gt_str[1] == "|":
                        phasing[i] = True
                    phasing_unknown.discard(i)
                if len(phasing_unknown) == 0:
                    break
            p.terminate()
            stderr = p.stderr.read()

    if stderr:
        raise RuntimeError(f"{vcf}: {stderr}")
    if len(phasing_unknown) > 0:
        raise RuntimeError(f"{vcf}: couldn't determine phasing for all samples.")
    return phasing


def contig_lengths(vcf):
    contigs = []
    cmd = ["bcftools", "view", "-h", vcf]
    with subprocess.Popen(
        cmd, bufsize=1, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    ) as p:
        for line in p.stdout:
            if line.startswith("##contig="):
                line = line.rstrip()
                line = line[len("##contig=<") : -1]
                cline = dict()
                for field in line.split(","):
                    try:
                        key, val = field.split("=")
                    except ValueError:
                        print(f"{vcf}: line={line}")
                        raise
                    cline[key] = val
                if "ID" not in cline or "length" not in cline:
                    raise ValueError(f"{vcf}: parse error: line={line}")
                try:
                    length = int(cline["length"])
                except ValueError:
                    print(f"{vcf}: line={line}")
                    raise
                contigs.append((cline["ID"], length))
        p.terminate()
        stderr = p.stderr.read()

    if stderr:
        raise RuntimeError(f"{vcf}: {stderr}")
    return contigs


def vcf2mat(
    vcf,
    samples_file,
    chrom,
    start,
    end,
    rng,
    max_missing_thres=None,
    maf_thres=None,
    unphase=True,
):
    """
    Extract a genotype matrix from ``vcf`` for the samples in file ``samples``
    and the genomic region defined by ``chrom:start-end``. ``rng`` is a random
    number generator object with a :meth:`random` method.
    If ``unphase`` is true, unphased genotypes are randomly assigned to the
    first or second chromosome of the individual.
    Returns a (pos, gt) tuple, where ``pos`` is a vector of positions and
    ``gt`` is the genotype matrix. Columns of the genotype matrix are
    haplotypes and rows correspond to the genomic coordinates in ``pos``.
    The major allele (frequency > 0.5) is coded as 0, the minor allele as 1,
    and missing genotypes as -1. When counts for both alleles are equal, the
    0 and 1 coding is randomly assigned.
    """

    cmd = [
        "bcftools",
        "query",
        "-S",
        samples_file,
        "-r",
        f"{chrom}:{start}-{end}",
        "-e",
        "TYPE!='snp'",  # Exclude non-SNPs.
        "-f",
        "%POS\\t%ALT[\\t%GT]\\n",
        vcf,
    ]

    def g2i(gt_str):
        """
        Convert diploid GT string to tuple of integers.
        """
        a0 = int(gt_str[0]) if gt_str[0] != "." else -1
        a1 = int(gt_str[2]) if gt_str[2] != "." else -1
        # If genotypes are unphased, we switch them with probability 0.5.
        if unphase and gt_str[1] == "/" and rng.random() > 0.5:
            a0, a1 = a1, a0
        return a0, a1

    flip = {0: 1, 1: 0, -1: -1}
    pos = []
    gt = []

    with subprocess.Popen(
        cmd, bufsize=1, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    ) as p:
        for line in p.stdout:
            fields = line.split()
            _pos = int(fields[0])
            _alt = fields[1]

            if len(_alt) != 1:
                # Not a biallelic SNP.
                # Filtering here shouldn't be needed, because we filter with
                # the `-e` bcftools parameter. However, it has happened that
                # the filtering I've asked for has not always been as strict
                # as I wanted. But checking the length of the ALT allele always
                # cleans up what bcftools lets through.
                continue

            _gt = list(itertools.chain(*map(g2i, fields[2:])))
            allele_counts = collections.Counter(_gt)

            if max_missing_thres is not None and allele_counts[
                -1
            ] > max_missing_thres * len(_gt):
                continue

            if maf_thres is not None:
                n = allele_counts[0] + allele_counts[1]
                minor_ac = min(allele_counts[0], allele_counts[1])
                if minor_ac < maf_thres * n:
                    continue

            # Polarise 0 and 1 in genotype matrix by major allele frequency.
            # If allele counts are the same, randomly choose a major allele.
            if allele_counts[1] > allele_counts[0] or (
                allele_counts[1] == allele_counts[0] and rng.random() > 0.5
            ):
                _gt = [flip[g] for g in _gt]

            pos.append(_pos)
            gt.append(_gt)

        p.terminate()
        stderr = p.stderr.read()

    if stderr:
        raise RuntimeError(f"{vcf}: {stderr}")

    pos = np.array(pos, dtype=np.uint32)
    gt = np.array(gt, dtype=np.int8)
    return pos, gt


def resize(pos, gt, sequence_length, num_rows):
    """
    Resize genotype matrix ``gt``, that spans ``sequence_length`` base pairs,
    to have ``num_rows`` rows. The vector ``pos`` should contain the position
    of each row in ``gt``. Genotypes in the original matrix are mapped to a new
    position by retaining their relative offset in (0, sequence_length),
    and summing values with the same destination offset.
    """
    A = np.zeros((num_rows, gt.shape[1]), dtype=gt.dtype)
    for _pos, _gt in zip(pos, gt):
        j = int(num_rows * _pos / sequence_length)
        np.add(A[j, :], _gt, out=A[j, :], where=_gt != -1)
    return A


def coordinates(vcf_files, chr_list, window, step, one_based=True, closed=True):
    """
    Return chrom:start-end coordinates for the given vcf_files, chr_list,
    window, and step size.
    The default one-based closed intervals are compatible with `bcftools -r`,
    so the first two intervals are chrom:1-window, chrom:(step+1)-(window+step).
    For BED-like intervals (one_based=False, closed=False), we instead get
    chrom:0-window, chrom:step-(window+step).
    """
    assert step <= window
    coords = []
    last_vcf = None
    for vcf, chrom in zip(vcf_files, chr_list):
        if vcf != last_vcf:
            contigs = dict(contig_lengths(vcf))
        last_vcf = vcf
        chrlen = contigs.get(str(chrom))
        if chrlen is None:
            raise RuntimeError(f"{vcf}: couldn't find chromosome '{chrom}'")

        starts = np.arange(0, chrlen - window, step) + (1 * one_based)
        ends = starts + window - (1 * closed)
        coords.extend((vcf, chrom, start, end) for start, end in zip(starts, ends))
    return coords
