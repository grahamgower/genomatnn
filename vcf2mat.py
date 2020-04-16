import random
import itertools
import subprocess
import collections

import numpy as np


def vcf2mat(
        vcf, samples, chrom, start, end, rng,
        max_missing_thres=None, maf_thres=None, unphase=True):
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

    cmd = ["bcftools", "query"]
    if samples is not None:
        cmd.extend(["-S", samples])
    cmd.extend([
            "-r", f"{chrom}:{start}-{end}",
            "-e", "TYPE!='snp'",  # Exclude non-SNPs.
            "-f", "%POS\\t%ALT[\\t%GT]\\n",
            vcf
            ])

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

    with subprocess.Popen(cmd, bufsize=1, text=True, stdout=subprocess.PIPE) as p:
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

            if max_missing_thres is not None and \
                    allele_counts[-1] > max_missing_thres * len(_gt):
                continue

            if maf_thres is not None:
                n = allele_counts[0] + allele_counts[1]
                minor_ac = min(allele_counts[0], allele_counts[1])
                if minor_ac < maf_thres * n:
                    continue

            # Polarise 0 and 1 in genotype matrix by major allele frequency.
            # If allele counts are the same, randomly choose a major allele.
            if allele_counts[1] > allele_counts[0] or \
                    (allele_counts[1] == allele_counts[0] and rng.random() > 0.5):
                _gt = [flip[g] for g in _gt]

            pos.append(_pos)
            gt.append(_gt)

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
    m = np.zeros((num_rows, gt.shape[1]), dtype=gt.dtype)
    for _pos, _gt in zip(pos, gt):
        j = int(num_rows * _pos / sequence_length)
        np.add(m[j, :], _gt, out=m[j, :], where=_gt != -1)
    return m


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(
            description="Extract genotype matrix from a vcf.")
    parser.add_argument("--chrom", type=str, required=True, help="Chromosome")
    parser.add_argument("--start", type=int, required=True, help="Start coordinate")
    parser.add_argument("--end", type=int, required=True, help="End coordinate")
    parser.add_argument(
            "--samples", metavar="samples.txt",
            help="File containing list of samples.")
    parser.add_argument(
            "--maf-thres", metavar="MAF", type=float, default=0.05,
            help="Exclude SNPs with minor allele frequency < MAF [%(default)s].")
    parser.add_argument(
            "--num-rows", type=int, default=32,
            help="Resize genotype matrixes to have this many rows [%(default)s].")
    parser.add_argument("vcf", metavar="in.vcf", help="Input vcf/bcf.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    rng = random.Random(1234)
    pos, gt = vcf2mat(
            vcf=args.vcf, samples=args.samples, chrom=args.chrom,
            start=args.start, end=args.end, rng=rng,
            max_missing_thres=0.1, maf_thres=args.maf_thres,
            )
    relative_pos = pos - args.start
    sequence_length = args.end - args.start
    gt = resize(relative_pos, gt, sequence_length, args.num_rows)
    print(gt.shape)
