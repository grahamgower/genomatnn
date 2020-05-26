import itertools
import operator

import numpy as np
import matplotlib

# Don't try to use X11.
matplotlib.use('Agg')

import matplotlib.pyplot as plt  # NOQA: E402
from matplotlib.backends.backend_pdf import PdfPages  # NOQA: E402
from matplotlib.lines import Line2D  # NOQA: E402
from mpl_toolkits.axes_grid1.inset_locator \
        import zoomed_inset_axes, mark_inset  # NOQA: E402

import vcf  # NOQA: E402


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


def predictions(conf, pred_file, pdf_file, aspect=9/16, scale=1.0):
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


def longest_common_prefix(string_list):
    min_length = min(len(s) for s in string_list)
    for i in range(min_length):
        s_i = set(s[i] for s in string_list)
        if len(s_i) != 1:
            break
    return string_list[0][:i]


def esf(a, q, norm=True):
    """
    Empirical survival function.

    Returns the proportion of 'a' which is greater than each quantile in 'q'.
    """
    b = np.sort(a)
    sf = np.empty_like(q)
    n = 0
    for i, x in enumerate(q):
        while n < len(b) and x > b[n]:
            n += 1
        sf[i] = len(b) - n
    if norm:
        sf /= len(b)
    return sf


def roc(
        conf, val_labels, val_pred, val_metadata, pdf_file,
        aspect=10/16, scale=1.5, inset=False):

    assert val_labels.shape == val_pred.shape
    assert len(val_labels) == len(val_metadata)

    ((false_tranche, false_modelspecs),
     (true_tranche, true_modelspecs)) = list(conf.tranche.items())

    tp = val_pred[np.where(val_labels == 1)]
    fp_list = []
    for fp_modelspec in false_modelspecs:
        fp = val_pred[np.where(val_metadata["modelspec"] == fp_modelspec)]
        fp_list.append(fp)

    roc_cfg = conf.eval.get("roc")
    if roc_cfg is not None:
        plot_cfg = roc_cfg.get("plot")
        if plot_cfg is not None:
            if "aspect" in plot_cfg:
                aspect = plot_cfg["aspect"]
            if "scale" in plot_cfg:
                scale = plot_cfg["scale"]
            # TODO: get inset from config

    pdf = PdfPages(pdf_file)
    fig_w, fig_h = plt.figaspect(aspect)
    fig_w *= scale
    fig_h *= scale

    fig, axs = plt.subplots(1, 3, figsize=(fig_w, fig_h))

    if len(false_modelspecs) > 1:
        fp_pfx = longest_common_prefix(false_modelspecs)
        fp_labels = [spec[len(fp_pfx):] for spec in false_modelspecs]
    else:
        fp_labels = false_modelspecs

    if inset:
        ax0_inset = zoomed_inset_axes(axs[0], 4, loc="center")
        # ax0_inset.set_xlim([0, 0.25])
        ax0_inset.set_xlim([0, 0.1])
        ax0_inset.set_ylim([0.8, 1])

    q = np.linspace(0, 1, 101)
    tpr = esf(tp, q)

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colours = prop_cycle.by_key()['color']
    linestyles = ["-", "--", ":", "-."]
    markers = ".1s*P"

    for fp, label, m, c, ls in zip(fp_list, fp_labels, markers, colours, linestyles):
        # ROC
        fpr = esf(fp, q)
        axs[0].plot(fpr, tpr, color=c, linestyle=ls, label=label)
        if inset:
            ax0_inset.plot(fpr, tpr, color=c, linestyle=ls)
        for i, ch in zip((50, 90), ('o', 'x')):
            if ch == 'x':
                ec = "none"
                fc = c
            else:
                ec = c
                fc = "none"
            axs[0].scatter(fpr[i], tpr[i], marker=ch, facecolor=fc, edgecolor=ec)
            if inset:
                ax0_inset.scatter(fpr[i], tpr[i], marker=ch, facecolor=fc, edgecolor=ec)

        # PR
        recall = tpr
        precision = tpr / (tpr + fpr)

        # We should do a scatter plot, not a line plot, because linear
        # interpolation is not appropriate for precision-recall curves.
        # See Davis & Goadrich (2006), http://doi.org/10.1145/1143844.1143874
        # But our curves are very linear, and scatter plots are hard to see.
        axs[1].plot(recall, precision, color=c, linestyle=ls, label=label)
        # axs[1].scatter(
        #        recall, precision, marker=m, facecolor=c, edgecolor=c, label=label)

        if len(axs) >= 3:
            tnr = 1 - fpr  # true negative rate (specificity)
            fnr = 1 - tpr  # false negative rate
            npv = tnr / (tnr + fnr)  # negative predictive value

            axs[2].plot(tnr, npv, color=c, linestyle=ls, label=label)
            # axs[2].scatter(fnr, _for, marker=m, facecolor=c, edgecolor=c, label=label)

    handles, _ = axs[0].get_legend_handles_labels()
    handles.extend(
            [Line2D(
                [0], [0], marker='o', label='p > 0.50', c="none",
                markerfacecolor='none', markeredgecolor='k', markersize=10),
             Line2D(
                 [0], [0], marker='x', label='p > 0.90', c="none",
                 markerfacecolor='k', markeredgecolor='k', markersize=10)
             ])
    axs[0].legend(
            title="Condition negative:", handles=handles, framealpha=1,
            loc="lower right")

    for ax in axs:
        # xlim = ax.get_xlim()
        # ylim = ax.get_ylim()
        # ax.set_xlim([min(xlim[0],0),max(xlim[1],1)])
        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim([-0.05, 1.05])

        itv = 0.05
        ax.set_xticks(np.arange(0, 1, itv), minor=True)
        ax.set_yticks(np.arange(0, 1, itv), minor=True)
        ax.grid(which="both", linestyle='-', color="gray", lw=0.1)

    if inset:
        ax0_inset.set_xticks(np.arange(*ax0_inset.get_xlim(), itv))
        ax0_inset.set_yticks(np.arange(*ax0_inset.get_ylim(), itv))
        ax0_inset.grid(which="major", linestyle='-', color="gray", lw=0.1)
        plt.setp(ax0_inset.get_xticklabels(), visible=False)
        plt.setp(ax0_inset.get_yticklabels(), visible=False)

        # draw a bbox of the region of the inset axes in the parent axes and
        # connecting lines between the bbox and the inset axes area
        mark_inset(axs[0], ax0_inset, loc1=1, loc2=3, fc="none", ec="0.5")

    axs[0].set_title("ROC")
    axs[0].set_xlabel("FPR: FP/(FP+TN)")  # (not-AI predicted to be AI)
    axs[0].set_ylabel("TPR: TP/(TP+FN)")  # (AI predicted to be AI)

    axs[1].set_title("Precision-Recall")
    axs[1].set_xlabel("Recall: TP/(TP+FN)")
    axs[1].set_ylabel("Precision: TP/(TP+FP)")

    if len(axs) >= 3:
        axs[2].set_title("Specificity vs. Negative Predictive Value")
        axs[2].set_xlabel("TNR: TN/(TN+FP)")
        axs[2].set_ylabel("NPV: TN/(TN+FN)")

    fig.tight_layout()
    pdf.savefig(figure=fig)
    plt.close(fig)
    pdf.close()
