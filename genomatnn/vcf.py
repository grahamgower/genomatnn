import subprocess
import functools
import concurrent.futures

import numpy as np

from genomatnn import convert


def bcftools_query(
    fmt, vcf_file, exclude=None, samples_file=None, regions_file=None, regions=None
):
    cmd = ["bcftools", "query", "-f", fmt]
    if exclude is not None:
        cmd.extend(["-e", exclude])
    if samples_file is not None:
        cmd.extend(["-S", samples_file])
    if regions_file is not None:
        cmd.extend(["-R", regions_file])
    if regions is not None:
        cmd.extend(["-r", regions])
    cmd.append(vcf_file)

    with subprocess.Popen(
        cmd,
        bufsize=1,
        universal_newlines=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ) as p:
        for line in p.stdout:
            yield line
        p.terminate()
        stderr = p.stderr.read()

    if stderr:
        raise RuntimeError(f"{vcf_file}: {stderr}")


def contig_lengths(vcf):
    contigs = []
    cmd = ["bcftools", "view", "-h", vcf]
    with subprocess.Popen(
        cmd,
        bufsize=1,
        universal_newlines=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
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


def genotypes(
    vcf,
    *,
    vcf_pop_intervals,
    samples_file=None,
    regions=None,
    max_missing_thres=None,
    maf_thres=None,
    rng=None,
):
    """
    Generate vectors of genotypes for sites in ``vcf``.
    ``samples_file`` is a file with one sample name per line.
    ``rng`` is a random number generator object with a :meth:`random` method.
    Yields a (chrom, pos, gt) tuple, where ``gt`` is the vector of genotypes.
    The major allele (frequency > 0.5) is coded as 0, the minor allele as 1,
    and missing genotypes as 2. When counts for both alleles are equal, the
    0 and 1 coding is randomly assigned.
    """

    from genomatnn.misc import gt_bytes2vec

    if max_missing_thres is None:
        max_missing_thres = 1
    if maf_thres is None:
        maf_thres = 0
    if rng is None:
        rng = np.random.default_rng()

    for line in bcftools_query(
        "%CHROM\\t%POS\\t%ALT[\\t%GT]\\n",
        vcf,
        exclude="TYPE!='snp'",
        samples_file=samples_file,
        regions=regions,
    ):
        fields = line.split(maxsplit=3)
        chrom = fields[0]
        pos = int(fields[1])
        alt = fields[2]
        gt_str = fields[3]

        if len(alt) != 1:
            # Not a biallelic SNP.
            # Filtering here shouldn't be needed, because we filter with
            # the `-e` bcftools parameter. However, it has happened that
            # the filtering I've asked for has not always been as strict
            # as I wanted. But checking the length of the ALT allele always
            # cleans up what bcftools lets through.
            continue

        gt, ac = gt_bytes2vec(gt_str.encode(encoding="ascii"))

        if ac[2] > max_missing_thres * len(gt):
            continue

        n = ac[0] + ac[1]
        minor_ac = min(ac[0], ac[1])
        if minor_ac < maf_thres * n:
            continue

        # Check missingness for each population.
        skip = False
        for a, b in vcf_pop_intervals:
            if np.count_nonzero(gt[a:b] == 2) > max_missing_thres * (b - a):
                skip = True
                break
        if skip:
            continue

        # Polarise 0 and 1 in genotype matrix by major allele frequency.
        # If allele counts are the same, randomly choose a major allele.
        if ac[1] > ac[0] or (ac[1] == ac[0] and rng.random() > 0.5):
            np.bitwise_xor(gt, 1, out=gt, where=gt != 2)

        yield chrom, pos, gt


def accumulate_matrices(
    vcf,
    *,
    winsize,
    winstep,
    vcf_pop_intervals,
    samples_file=None,
    regions=None,
    min_seg_sites=None,
    max_missing_thres=None,
    maf_thres=None,
    rng=None,
):
    """
    Site-wise accumulation of genotype vectors into matrices, where the span of
    sites for each matrix may overlap.
    """
    if winsize < winstep:
        raise ValueError("Must have winsize >= winstep")
    if min_seg_sites is None:
        min_seg_sites = 1

    start = None
    end = None
    prev_chrom = None
    gt_list = []
    gt_pos_list = []

    for chrom, pos, gt in genotypes(
        vcf,
        vcf_pop_intervals=vcf_pop_intervals,
        samples_file=samples_file,
        regions=regions,
        max_missing_thres=max_missing_thres,
        maf_thres=maf_thres,
        rng=rng,
    ):
        if start is None:
            start = pos - (pos % winstep) + 1
            end = start + winsize - 1

        if chrom != prev_chrom:
            if len(gt_list) >= min_seg_sites:
                M = np.array(gt_list, dtype=gt_list[0].dtype)
                yield prev_chrom, start, end, M, gt_pos_list
            gt_list = []
            gt_pos_list = []
            start = pos - (pos % winstep) + 1
            end = start + winsize - 1
        else:
            while pos > end:
                i = 0
                for i, _pos in enumerate(gt_pos_list):
                    if _pos >= start:
                        break
                if i > 0:
                    gt_pos_list = gt_pos_list[i:]
                    gt_list = gt_list[i:]

                if len(gt_list) >= min_seg_sites:
                    M = np.array(gt_list, dtype=gt_list[0].dtype)
                    yield chrom, start, end, M, gt_pos_list

                if len(gt_list) == 0:
                    start = pos - (pos % winstep) + 1
                else:
                    start += winstep
                end = start + winsize - 1

        gt_list.append(gt)
        gt_pos_list.append(pos)
        prev_chrom = chrom

    i = 0
    for i, _pos in enumerate(gt_pos_list):
        if _pos >= start:
            break
    if i > 0:
        gt_pos_list = gt_pos_list[i:]
        gt_list = gt_list[i:]

    if len(gt_list) >= min_seg_sites:
        M = np.array(gt_list, dtype=gt_list[0].dtype)
        yield chrom, start, end, M, gt_pos_list


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
        np.add(A[j, :], _gt, out=A[j, :], where=_gt != 2)
    return A


def _matrix_batch_1(
    args,
    *,
    winsize,
    winstep,
    num_rows,
    counts,
    indices,
    haploid_counts,
    haploid_indices,
    ref_pop,
    samples_file,
    min_seg_sites,
    max_missing_thres,
    maf_thres,
    phased,
    ploidy,
):
    (vcf, chrom, start, end), seed = args
    rng = np.random.default_rng(seed)
    region = f"{chrom}:{start}-{end}"
    vcf_pop_intervals = [
        (j, j + n) for j, n in zip(haploid_indices.values(), haploid_counts.values())
    ]
    coords = []
    B = []
    for chrom, start, end, M, gt_pos_list in accumulate_matrices(
        vcf,
        winsize=winsize,
        winstep=winstep,
        vcf_pop_intervals=vcf_pop_intervals,
        samples_file=samples_file,
        regions=region,
        min_seg_sites=min_seg_sites,
        max_missing_thres=max_missing_thres,
        maf_thres=maf_thres,
        rng=rng,
    ):
        relative_pos = np.array(gt_pos_list, dtype=int) - start
        A = resize(relative_pos, M, winsize, num_rows)
        if not phased:
            A = convert.collapse_unphased(A, ploidy)
        A = convert.reorder_and_sort(A, counts, indices, indices, ref_pop)
        A = A[np.newaxis, :, :, np.newaxis]
        coords.append((chrom, start, end))
        B.append(A)
    if len(B) > 0:
        B = np.concatenate(B)
    return coords, B


def matrix_batches(
    vcf_files,
    chr_list,
    *,
    # required keyword args
    winsize,
    winstep,
    num_rows,
    counts,
    indices,
    haploid_counts,
    haploid_indices,
    ref_pop,
    phased,
    ploidy,
    # optional keyword args
    coordinates=None,
    samples_file=None,
    min_seg_sites=None,
    max_missing_thres=None,
    maf_thres=None,
    vcf_chunk_size=10 * 1000 * 1000,  # 10 mb
    parallelism=1,
    rng=None,
):
    assert winsize >= winstep
    assert vcf_chunk_size >= winsize
    assert vcf_chunk_size % winsize == 0
    if rng is None:
        rng = np.random.default_rng()

    def coords_generator():
        prev_vcf = None
        for vcf, chrom in zip(vcf_files, chr_list):
            if vcf != prev_vcf:
                contigs = dict(contig_lengths(vcf))
            prev_vcf = vcf
            chrlen = contigs.get(str(chrom))
            if chrlen is None:
                raise RuntimeError(f"{vcf}: couldn't find chromosome '{chrom}'")

            for start in range(1, chrlen, vcf_chunk_size):
                # Note that the end coordinate overlaps the next chunk by
                # (winsize - winstep) positions.
                end = min(chrlen, start + vcf_chunk_size + winsize - winstep - 1)
                yield str(vcf), chrom, start, end
                rng.randrange(1, 2 ** 32)

    if coordinates is None:
        coordinates = coords_generator()
    args = ((coord, rng.randrange(1, 2 ** 32)) for coord in coordinates)

    batch_f = functools.partial(
        _matrix_batch_1,
        winsize=winsize,
        winstep=winstep,
        num_rows=num_rows,
        counts=counts,
        indices=indices,
        haploid_counts=haploid_counts,
        haploid_indices=haploid_counts,
        ref_pop=ref_pop,
        samples_file=samples_file,
        min_seg_sites=min_seg_sites,
        max_missing_thres=max_missing_thres,
        maf_thres=maf_thres,
        phased=phased,
        ploidy=ploidy,
    )

    with concurrent.futures.ProcessPoolExecutor(parallelism) as ex:
        for coords, B in ex.map(batch_f, args):
            if len(B) > 0:
                yield coords, B
