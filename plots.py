import itertools
import operator

import numpy as np
import matplotlib
matplotlib.use('Agg')  # NOQA  # don't try to use $DISPLAY
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import vcf


def predictions_all_chr(pred_by_chr):
    pass


def predictions_one_chr(fig, ax, header, chrom, preds, chrlen, p_thres=0.9):
    assert len(header) == 4, "Non-binary predictions not yet supported."

    col = plt.get_cmap("tab20").colors
    sym = "ods"
    sym2 = "dso"
    ec = [col[2], col[4], col[2]]
    ec2 = [col[0], col[4], col[2]]
    fc = [col[3], col[5], col[3]]
    fc2 = [col[1], col[5], col[3]]
    sz1 = [10, 10, 10]
    sz2 = [20, 10, 10]

    ax.hlines(p_thres, -0, chrlen, linestyle='--', color="gray", lw=0.5)
    ax.grid(linestyle='-', color="gray", lw=0.1)

    start = header[1]
    end = header[2]
    label = header[3]
    midpos = (preds[start] + preds[end]) // 2
    p = preds[label]
    idx1 = np.where(p > p_thres)
    idx2 = np.where(np.logical_not(p > p_thres))
    i = 0
    ax.scatter(
            midpos[idx1], p[idx1], edgecolor=ec[i], facecolor=fc[i],
            marker=sym2[i], label=None, s=sz2[i], lw=0.5)
    ax.scatter(
            midpos[idx2], p[idx2], edgecolor=ec2[i], facecolor=fc2[i],
            marker=sym[i], label=label, s=sz1[i], lw=0.5)
    ax.set_ylabel(label)

    xt_interval = 1e7
    if chrlen > 1e8:
        xt_interval *= 2
    if chrlen > 2e8:
        xt_interval *= 2
    xticks = np.arange(xt_interval, chrlen, xt_interval)
    xlabels = [str(int(x/1e6))+" mbp" for x in xticks]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)
    ax.set_xlim([0, chrlen])

    if not chrom.startswith("chr"):
        chrom = "chr" + chrom
    ax.set_title(chrom)


def load_predictions(pred_file, max_chr_name_len=128):
    chr_dtype = f"U{max_chr_name_len}"
    with open(pred_file) as f:
        # header is: chrom, start, end, Pr(X), ...
        header = next(f).split()
        dtype = [(header[0], chr_dtype), (header[1], int), (header[2], int)] + \
                [(h, float) for h in header[3:]]
        preds = np.loadtxt(f, dtype=dtype)
    preds_by_chr = dict()
    for k, g in itertools.groupby(preds, operator.itemgetter(0)):
        preds_by_chr[k] = np.fromiter(g, dtype=dtype)
    return header, preds_by_chr


def predictions(conf, pred_file, aspect=9/16, scale=1.0):
    header, preds_by_chr = load_predictions(pred_file)
    chrlen = dict(vcf.contig_lengths(conf.file[0]))

    for chrom in preds_by_chr.keys():
        if chrom not in chrlen:
            raise RuntimeError(
                    f"{pred_file}: chromosome '{chrom}' not found in vcf header "
                    f"of {conf.file[0]}")

    plot_cfg = conf.apply.get("plot")
    if plot_cfg is not None:
        if "aspect" in plot_cfg:
            aspect = plot_cfg["aspect"]
        if "scale" in plot_cfg:
            scale = plot_cfg["scale"]

    pdf_file = conf.nn_hdf5_file[:-len(".hdf5")] + ".pdf"
    pdf = PdfPages(pdf_file)
    fig_w, fig_h = plt.figaspect(aspect)
    fig_w *= scale
    fig_h *= scale

    for chrom, preds in preds_by_chr.items():
        fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))
        predictions_one_chr(fig, ax, header, chrom, preds, chrlen[chrom])
        fig.tight_layout()
        pdf.savefig(figure=fig)
        plt.close(fig)
    pdf.close()
