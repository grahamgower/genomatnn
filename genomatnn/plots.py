import itertools
import operator
import collections
import copy

import numpy as np
import matplotlib

# Don't try to use X11.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.collections import PatchCollection  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from mpl_toolkits.axes_grid1.inset_locator import (  # noqa: E402
    zoomed_inset_axes,
    mark_inset,
)
from mpl_toolkits import axes_grid1  # noqa: E402

from genomatnn import (  # noqa: E402
    vcf,
    calibrate,
)


def predictions_all_chr(ax, header, preds_by_chr, lengths_by_chr):
    assert len(header) == 4, "Non-binary predictions not yet supported."

    start = header[1]
    end = header[2]
    label = header[3]

    colours = plt.get_cmap("tab10").colors
    chrpos = []
    chrmid = []
    chrnames = []
    lsum = 0
    i = 0
    for chrom, preds in preds_by_chr.items():
        midpos = lsum + (preds[start] + preds[end]) // 2
        ax.scatter(
            midpos,
            preds[label],
            color=colours[i],
            marker="o",
            s=1,
            rasterized=True,
        )

        chrlen = lengths_by_chr[chrom]
        chrpos.append(lsum + chrlen)
        chrmid.append(lsum + chrlen // 2)
        chrnames.append(chrom)
        lsum += chrlen
        i = (i + 1) % 2

    chrindexes = list(range(len(chrnames)))
    if False:
        # ugly workaround for humans so the labels can be read
        chrindexes = list(range(1, 10, 2)) + [14, 19]

    # ax.set_title("CNN predictions")
    ax.set_xlabel("Chromosome")
    ax.set_ylabel(label)
    ax.set_xticks([chrmid[i] for i in chrindexes])
    ax.set_xticklabels([chrnames[i] for i in chrindexes], rotation=90)
    # ax.set_ylim(-0.02, None)
    ax.set_ylim(-0.02, 1.02)

    # ugly hardcoded annotation of peaks
    bbox = dict(boxstyle="round", fc="lightblue", ec="black", lw=1, alpha=0.5)
    if False:  # european
        ax.annotate("BAZ2B", xy=(0.06 * lsum, 0.765), bbox=bbox)
        ax.annotate("ZBTB20", xy=(0.22 * lsum, 0.745), bbox=bbox)
        ax.annotate("TSNARE1", xy=(0.43 * lsum, 0.715), bbox=bbox)
        ax.annotate("BNC2", xy=(0.555 * lsum, 0.66), bbox=bbox)
        ax.annotate("ZNF486", xy=(0.84 * lsum, 0.76), bbox=bbox)
        ax.annotate("WDR88", xy=(0.84 * lsum, 0.71), bbox=bbox)
        ax.annotate("KCNQ2", xy=(0.97 * lsum, 0.715), bbox=bbox)
    if False:  # papuan
        ax.annotate("SLC30A9", xy=(0.15 * lsum, 0.283), bbox=bbox)
        ax.annotate("TNFAIP3", xy=(0.322 * lsum, 0.29), bbox=bbox)
        ax.annotate("SFRP4", xy=(0.455 * lsum, 0.215), bbox=bbox)
        ax.annotate("RBM19", xy=(0.63 * lsum, 0.19), bbox=bbox)
        ax.annotate("DGCR2", xy=(0.95 * lsum, 0.177), bbox=bbox)

    x1, x2 = ax.get_xlim()
    ax.set_xlim(x1, x2)


def predictions_one_chr(ax, header, chrom, preds, chrlen):
    assert len(header) == 4, "Non-binary predictions not yet supported."

    start = header[1]
    end = header[2]
    label = header[3]

    colours = plt.get_cmap("tab10").colors
    midpos = (preds[start] + preds[end]) // 2
    ax.scatter(
        midpos,
        preds[label],
        color=colours[0],
        marker="o",
        s=10,
        rasterized=True,
    )
    ax.set_ylabel(label)
    ax.set_xlabel("Genomic coordinate")

    xt_interval = 1e7
    if chrlen > 1e8:
        xt_interval *= 2
    if chrlen > 2e8:
        xt_interval *= 2
    xticks = np.arange(xt_interval, chrlen, xt_interval)
    xlabels = [str(int(x / 1e6)) + " mbp" for x in xticks]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)
    ax.set_ylim(-0.02, 1.02)

    x1, x2 = ax.get_xlim()
    ax.set_xlim(x1, x2)

    if chrom.startswith("chr"):
        chrom = chrom[len("chr") :]
    ax.set_title(f"CNN predictions: chromosome {chrom}")


def load_predictions(pred_file, max_chr_name_len=128):
    chr_dtype = f"U{max_chr_name_len}"
    with open(pred_file) as f:
        # header is: chrom, start, end, Pr(X), ...
        header = next(f).split()
        dtype = [(header[0], chr_dtype), (header[1], int), (header[2], int)] + [
            (h, float) for h in header[3:]
        ]
        preds = np.loadtxt(f, dtype=dtype)
    preds_by_chr = dict()
    for k, g in itertools.groupby(preds, operator.itemgetter(0)):
        preds_by_chr[k] = np.fromiter(g, dtype=dtype)
    return header, preds_by_chr


def predictions(conf, pred_file, pdf_file, aspect=9 / 16, scale=1.0, dpi=200):
    header, preds_by_chr = load_predictions(pred_file)
    chrlen = dict(vcf.contig_lengths(conf.file[0]))

    for chrom in preds_by_chr.keys():
        if chrom not in chrlen:
            raise RuntimeError(
                f"{pred_file}: chromosome '{chrom}' not found in vcf header "
                f"of {conf.file[0]}"
            )

    aspect = conf.get("apply.plot.aspect", aspect)
    scale = conf.get("apply.plot.scale", scale)

    pdf = PdfPages(pdf_file)
    fig_w, fig_h = plt.figaspect(aspect)
    fig_w *= scale
    fig_h *= scale

    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))
    predictions_all_chr(ax, header, preds_by_chr, chrlen)
    fig.tight_layout()
    pdf.savefig(figure=fig, dpi=dpi)
    plt.close(fig)

    # for chrom, preds in preds_by_chr.items():
    #    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))
    #    predictions_one_chr(ax, header, chrom, preds, chrlen[chrom])
    #    fig.tight_layout()
    #    pdf.savefig(figure=fig, dpi=dpi)
    #    plt.close(fig)

    pdf.close()


def longest_common_prefix(string_list):
    min_length = min(len(s) for s in string_list)
    for i in range(min_length):
        s_i = set(s[i] for s in string_list)
        if len(s_i) != 1:
            break
    return string_list[0][:i]


def mcc_unit(tp, fp):
    """
    Unit-normalised Matthews correlation coefficient (MCC).
    """
    fn = 1 - tp
    tn = 1 - fp
    mcc_num = tp * tn - fp * fn
    mcc_den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.true_divide(
        mcc_num, mcc_den, out=np.full_like(tp, np.nan), where=mcc_den != 0
    )
    return (mcc + 1) / 2


def f1_score(tp, fp):
    """
    F1 score: the harmonic mean of precision and recall.
    """
    fn = 1 - tp
    f1_num = 2 * tp
    f1_den = 2 * tp + fp + fn
    f1 = np.true_divide(f1_num, f1_den, out=np.full_like(tp, np.nan), where=f1_den != 0)
    return f1


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
    *,
    conf,
    labels,
    pred,
    metadata,
    extra_labels,
    extra_pred,
    extra_metadata,
    pdf_file,
    aspect=10 / 32,
    scale=1,
    inset=False,
):
    (
        (condition_negative, false_modelspecs),
        (condition_positive, true_modelspecs),
    ) = copy.deepcopy(list(conf.tranche.items()))

    tp = pred[np.where(labels == 1)]
    fp_list = []
    for fp_modelspec in false_modelspecs:
        fp = pred[np.where(metadata["modelspec"] == fp_modelspec)]
        fp_list.append(fp)

    extra_sims = conf.get("sim.extra")
    if extra_sims is not None and len(extra_sims) == 0:
        extra_sims = None
    if extra_sims is not None:
        assert extra_labels is not None
        assert extra_pred is not None
        assert extra_metadata is not None
        for mspec_label, extra_modelspecs in extra_sims.items():
            for modelspec in extra_modelspecs:
                fp = extra_pred[np.where(extra_metadata["modelspec"] == modelspec)]
                fp_list.append(fp)
                false_modelspecs.append(modelspec)

    aspect = conf.get("eval.plot.aspect", aspect)
    scale = conf.get("eval.plot.scale", scale)
    aspect = conf.get("eval.roc.plot.aspect", aspect)
    scale = conf.get("eval.roc.plot.scale", scale)
    # TODO: support the inset for each subplot
    # inset = conf.get("eval.roc.plot.inset", inset)

    pdf = PdfPages(pdf_file)
    fig_w, fig_h = plt.figaspect(aspect)
    fig, axs = plt.subplots(1, 3, figsize=(scale * fig_w, scale * fig_h))

    if len(false_modelspecs) > 1:
        fp_pfx = longest_common_prefix(false_modelspecs)
        fp_labels = [spec[len(fp_pfx) :] for spec in false_modelspecs]
    else:
        try:
            fp_labels = ["/".join(spec.split("/")[-2:]) for spec in false_modelspecs]
        except IndexError:
            fp_labels = false_modelspecs

    if inset:
        ax0_inset = zoomed_inset_axes(axs[0], 4, loc="center")
        # ax0_inset.set_xlim([0, 0.25])
        ax0_inset.set_xlim([0, 0.1])
        ax0_inset.set_ylim([0.8, 1])

    q = np.linspace(0, 1, 101)
    tpr = esf(tp, q)

    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colours = prop_cycle.by_key()["color"]
    linestyles = ["-", "--", ":", "-."]
    markers = ".1s*P"

    for fp, label, m, c, ls in zip(fp_list, fp_labels, markers, colours, linestyles):
        # ROC
        fpr = esf(fp, q)
        axs[0].plot(fpr, tpr, color=c, linestyle=ls, label=label)
        if inset:
            ax0_inset.plot(fpr, tpr, color=c, linestyle=ls)
        axs[0].scatter(fpr[50], tpr[50], marker="o", facecolor="none", edgecolor=c)
        if inset:
            ax0_inset.scatter(
                fpr[50], tpr[50], marker="o", facecolor="none", edgecolor=c
            )

        # Precision-Recall
        recall = tpr
        precision = np.true_divide(
            tpr, tpr + fpr, out=np.full_like(tpr, np.nan), where=tpr + fpr != 0
        )

        # We should do a scatter plot, not a line plot, because linear
        # interpolation is not appropriate for precision-recall curves.
        # See Davis & Goadrich (2006), http://doi.org/10.1145/1143844.1143874
        # But our curves are very linear, and scatter plots are hard to see.
        axs[1].plot(recall, precision, color=c, linestyle=ls, label=label)
        # axs[1].scatter(
        #        recall, precision, marker=m, facecolor=c, edgecolor=c, label=label)
        axs[1].scatter(
            recall[50], precision[50], marker="o", facecolor="none", edgecolor=c
        )

        # MCC-F1
        mcc = mcc_unit(tpr, fpr)
        f1 = f1_score(tpr, fpr)
        axs[2].plot(f1, mcc, color=c, linestyle=ls, label=label)
        axs[2].scatter(f1[50], mcc[50], marker="o", facecolor="none", edgecolor=c)

    handles, _ = axs[0].get_legend_handles_labels()
    handles.extend(
        [
            Line2D(
                [0],
                [0],
                marker="o",
                c="none",
                markerfacecolor="none",
                markeredgecolor="k",
                markersize=10,
                label=f"Pr[{condition_positive}] > 0.50",
            ),
        ]
    )
    axs[0].legend(
        title="Condition negative:", handles=handles, framealpha=1, loc="lower right"
    )

    for ax in axs:
        # xlim = ax.get_xlim()
        # ylim = ax.get_ylim()
        # ax.set_xlim([min(xlim[0],0),max(xlim[1],1)])
        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim([-0.05, 1.05])

        itv = 0.2
        ax.set_xticks(np.arange(0, 1 + itv, itv))
        ax.set_yticks(np.arange(0, 1 + itv, itv))
        itv = 0.1
        ax.set_xticks(np.arange(0, 1, itv), minor=True)
        ax.set_yticks(np.arange(0, 1, itv), minor=True)
        ax.grid(which="both", linestyle="-", color="lightgray", lw=0.5)

    if inset:
        ax0_inset.set_xticks(np.arange(*ax0_inset.get_xlim(), itv))
        ax0_inset.set_yticks(np.arange(*ax0_inset.get_ylim(), itv))
        ax0_inset.grid(which="major", linestyle="-", color="gray", lw=0.1)
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

    axs[2].set_title("MCC-$F_1$ curve")
    axs[2].set_xlabel("$F_1$")
    axs[2].set_ylabel("Unit-normalised MCC")

    fig.tight_layout()
    pdf.savefig(figure=fig)
    plt.close(fig)
    pdf.close()


def partition2d_logx(x, y, z, bins):
    x = np.log10(x + 1e-9)
    xmax = round(np.max(x))
    xmin = round(np.min(x))
    ymax = round(np.max(y), 3)
    ymin = round(np.min(y), 3)
    xitv = max((xmax - xmin) / bins, 1e-9)
    yitv = max((ymax - ymin) / bins, 1e-9)

    binned = collections.defaultdict(list)
    for xi, yi, zi in zip(x, y, z):
        _x = xi - (xi % xitv)
        _y = yi - (yi % yitv)
        binned[(_x, _y)].append(zi)

    ymin = ymin - (ymin % yitv)
    ymax = ymax - (ymax % yitv) + yitv
    # xmin = xmin - (xmin % xitv)
    # xmax = xmax - (xmax % xitv) + xitv
    # print(np.unique([xy[0] for xy in binned.keys()]))
    # print(f"xmin={xmin}, xmax={xmax}, xitv={xitv}")
    return binned, xitv, yitv, xmin, xmax, ymin, ymax


def accuracy(conf, labels, pred, metadata, pdf_file, aspect=3 / 4, scale=1, bins=15):
    aspect = conf.get("eval.plot.aspect", aspect)
    scale = conf.get("eval.plot.scale", scale)
    aspect = conf.get("eval.accuracy.plot.aspect", aspect)
    scale = conf.get("eval.accuracy.plot.scale", scale)
    bins = conf.get("eval.accuracy.plot.bins", bins)
    title = conf.get("eval.plot.title", "True positive rate")
    title = conf.get("eval.accuracy.plot.title", title)

    # Consider only "condition positive" cases.
    idx = np.where(labels == 1)
    labels = labels[idx]
    pred = pred[idx]
    metadata = metadata[idx]

    pdf = PdfPages(pdf_file)
    fig_w, fig_h = plt.figaspect(aspect)
    fig, ax = plt.subplots(1, 1, figsize=(scale * fig_w, scale * fig_h))

    x = metadata["s"]
    y = metadata["T_sel"] / 1000
    # z = 1 - np.abs(labels - pred)  # "accuracy" that works for both labels
    z = pred

    binned, xitv, yitv, xmin, xmax, ymin, ymax = partition2d_logx(x, y, z, bins)

    boxes = []
    colours = []
    for (xi, yi), zi_array in binned.items():
        r = Rectangle((xi, yi), xitv, yitv)
        boxes.append(r)
        colours.append(np.mean(zi_array))

    pc = PatchCollection(boxes, rasterized=True)
    pc.set_array(np.array(colours))
    # pc.set_clim(vmax=1)
    ax.add_collection(pc)

    if title:
        ax.set_title(title)
    ax.set_xlabel("$\\log_{10}s$")
    ax.set_ylabel("$T_{sel}$ (kya)")

    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])

    fig.colorbar(pc)
    fig.tight_layout()

    pdf.savefig(figure=fig)
    plt.close(fig)
    pdf.close()


def confusion1(ax, cm, xticklabels, yticklabels, cbar=True, annotate=True):
    im = ax.imshow(cm.T, origin="lower", rasterized=True, vmin=0, vmax=1, cmap="Blues")
    if cbar:
        ax.figure.colorbar(im, ax=ax, label="Pr(Truth | Prediction)")
        # cb.ax.yaxis.set_label_position('left')
        # cb.ax.yaxis.set_ticks_position('left')

    ax.set(
        xticks=np.arange(cm.shape[0]),
        yticks=np.arange(cm.shape[1]),
        xticklabels=xticklabels,
        yticklabels=yticklabels,
        # title="Confusion matrix",
        xlabel="Predicted label",
        ylabel="True label",
    )

    if annotate:
        # Add text annotations.
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    i,
                    j,
                    f"{cm[i, j]:.3f}",
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                )


def confusion(conf, labels, pred, metadata, pdf_file, aspect=1, scale=1):
    """
    Confusion matrix.
    """

    aspect = conf.get("eval.plot.aspect", aspect)
    scale = conf.get("eval.plot.scale", scale)
    title = conf.get("eval.plot.title", "Confusion matrix")
    aspect = conf.get("eval.confusion.plot.aspect", aspect)
    scale = conf.get("eval.confusion.plot.scale", scale)
    title = conf.get("eval.confusion.plot.title", title)

    false_modelspecs, true_modelspecs = list(conf.tranche.values())
    modelspecs = false_modelspecs + true_modelspecs
    assert set(modelspecs) == set(np.unique(metadata["modelspec"]))

    n_labels = 2
    n_modelspecs = len(modelspecs)

    pdf = PdfPages(pdf_file)
    fig_w, fig_h = plt.figaspect(aspect)
    fig, ax = plt.subplots(1, 1, figsize=(scale * fig_w, scale * fig_h))

    cm_modelspecs = np.empty(shape=(n_labels, n_modelspecs))

    # labels x modelspecs
    for i in range(n_labels):
        idx = np.where(np.abs(pred - i) < 0.5)[0]
        n_pred = len(idx)
        for j in range(n_modelspecs):
            n_true = len(np.where(metadata["modelspec"][idx] == modelspecs[j])[0])
            if n_pred == 0:
                cm_modelspecs[i, j] = float("nan")
            else:
                cm_modelspecs[i, j] = n_true / n_pred

    modelspec_pfx = longest_common_prefix(modelspecs)
    short_modelspecs = [mspec[len(modelspec_pfx) :] for mspec in modelspecs]
    tranch_keys = list(conf.tranche.keys())

    confusion1(ax, cm_modelspecs, tranch_keys, short_modelspecs, cbar=False)

    if title:
        ax.set_title(title)
    fig.tight_layout()
    pdf.savefig(figure=fig)
    plt.close(fig)
    pdf.close()


def reliability(conf, labels, preds, pdf_file, aspect=10 / 16, scale=1.5, bins=10):
    """
    Reliability plot, aka calibration curve.
    """

    # Nuisance parameters that set spacing in the histogram.
    hist_width = round(1.5 * len(preds))
    hist_delta = hist_width - len(preds)

    aspect = conf.get("eval.plot.aspect", aspect)
    scale = conf.get("eval.plot.scale", scale)

    aspect = conf.get("eval.reliability.plot.aspect", aspect)
    scale = conf.get("eval.reliability.plot.scale", scale)
    bins = conf.get("eval.reliability.plot.bins", bins)
    hist_width = conf.get("eval.reliability.plot.hist_width", hist_width)
    hist_delta = conf.get("eval.reliability.plot.hist_delta", hist_delta)

    pdf = PdfPages(pdf_file)
    fig_w, fig_h = plt.figaspect(aspect)
    fig, axs = plt.subplots(1, 2, figsize=(scale * fig_w, scale * fig_h))
    ax1, ax2 = axs

    itv = 1.0 / bins
    binv = np.linspace(0, 1, bins + 1)[:-1]

    colours = plt.get_cmap("tab20").colors
    fc_colours = [colours[2 * i + 1] for i in range(len(colours) // 2)]
    ec_colours = [colours[2 * i] for i in range(len(colours) // 2)]
    markers = "oxd+^*"

    # Perfect reliability.
    ax1.plot([0.0, 1.0], [0.0, 1.0], c="lightgray", linestyle="--")

    for j, ((cal_label, p), marker, bfc, bec) in enumerate(
        zip(preds, markers, fc_colours, ec_colours)
    ):
        pbinned = collections.defaultdict(list)
        tbinned = collections.defaultdict(list)
        for xi, yi in zip(p, labels):
            i = xi - (xi % itv)
            pbinned[i].append(xi)
            tbinned[i].append(yi)

        nperbin = np.empty(shape=len(binv))
        accuracy = np.empty(shape=len(binv))
        binmean = np.empty(shape=len(binv))
        for i, t in enumerate(binv):
            accuracy[i] = np.mean(tbinned.get(t, [0]))
            nperbin[i] = len(tbinned.get(t, []))
            binmean[i] = np.mean(pbinned.get(t, [np.nan]))

        Z = calibrate.Z_score(p, labels)

        Z_label = "{}, Z={:.3g}".format(cal_label, Z)
        ax1.plot(
            binmean,
            accuracy,
            c=bec,
            linestyle="-",
            linewidth=1.5,
            markerfacecolor="none",
            marker=marker,
            ms=10,
            mew=2,
            label=Z_label,
        )
        ax2.bar(
            binv + itv * (j + hist_delta) / hist_width,
            nperbin,
            width=itv / hist_width,
            align="edge",
            linewidth=1.5,
            facecolor=bfc,
            edgecolor=bec,
            label=cal_label,
        )

    _, condition_positive = list(conf.tranche.keys())

    ax1.legend()
    ax1.set_title("Reliability of model predictions")
    ax1.set_xlabel(f"Model prediction (Pr{{{condition_positive}}})")
    ax1.set_ylabel("Proportion of true positives")

    ax2.legend()
    ax2.set_xlabel(f"Model prediction (Pr{{{condition_positive}}})")
    ax2.set_ylabel("Counts")
    ax2.set_title("Histogram of model predictions")
    ax2.set_yscale("log", nonposy="clip")

    fig.tight_layout()
    pdf.savefig(figure=fig)
    plt.close(fig)
    pdf.close()


def hap_matrix1(
    A,
    *,
    ax,
    title,
    sample_counts,
    pop_indices,
    sequence_length,
    cmap,
    rasterized,
    phased,
    ploidy,
    vmin=None,
    vmax=None,
    plot_cbar=True,
    ax_cbar=None,
    cbar_label=None,
    plot_pop_labels=True,
    ax_pops=None,
):
    """
    Plot one haplotype (or genotype) matrix, with a bar at the bottom
    indicating the population of origin for a given haplotype/genotype column.
    """
    if vmax is None:
        # vmax heuristic to make the patterns clear
        x = 1 if phased else ploidy
        vmax = int(round(x * np.log2(sequence_length / 20 / A.shape[0])))
        int_ticks = True
    else:
        int_ticks = False

    im = ax.imshow(
        A,
        interpolation="none",
        origin="lower",
        rasterized=rasterized,
        # left, right, bottom, top
        extent=(0, A.shape[1], 0, A.shape[0]),
        aspect="auto",
        cmap=cmap,
        norm=matplotlib.colors.PowerNorm(0.5, vmin=vmin, vmax=vmax),
    )

    ax.set_title(title)
    ax.set_ylabel("Genomic position")
    ax.set_yticks([0, ax.get_ylim()[1]])
    ax.set_yticklabels(["$0\,$kb", f"${sequence_length//1000}\,$kb"])  # noqa: W605

    plt.setp(ax.get_xticklabels(), visible=False)
    ax.tick_params(axis="x", length=0)

    for sp in ("top", "right", "bottom", "left"):
        ax.spines[sp].set_visible(False)

    if plot_cbar:
        if cbar_label is None:
            cbar_label = "Density of minor alleles"
        cb = ax.figure.colorbar(
            im,
            ax=ax if ax_cbar is None else None,
            cax=ax_cbar,
            extend="max" if int_ticks else "neither",
            pad=0.05,
            fraction=0.04,
            label=cbar_label,
        )
        cb.ax.yaxis.set_ticks_position("left")
        if int_ticks:
            cticks = list(range(vmax + 1))
            cb.set_ticks(cticks)

    if plot_pop_labels:
        #
        # Add population-labels colour bar on a new axis.
        #
        if ax_pops is None:
            divider = axes_grid1.make_axes_locatable(ax)
            ax_pops = divider.append_axes("bottom", 0.2, pad=0.05, sharex=ax)
        ax_pops.set_ylim([0, 1])

        for sp in ("top", "right", "bottom", "left"):
            ax_pops.spines[sp].set_visible(False)

        pop_ticks = []
        boxes = []
        colours = plt.get_cmap("tab10").colors
        for index, count in zip(pop_indices.values(), sample_counts.values()):
            # (x, y), width, height
            r = Rectangle((index, 0), count, 1)
            boxes.append(r)
            pop_ticks.append(index + count / 2)

        pc = PatchCollection(boxes, fc=colours, rasterized=rasterized)
        ax_pops.add_collection(pc)

        ax_pops.set_yticks([])
        ax_pops.set_yticklabels([])
        ax_pops.set_xticks(pop_ticks)
        ax_pops.set_xticklabels(list(pop_indices.keys()))
        if phased:
            ax_pops.set_xlabel("Haplotypes")
        else:
            ax_pops.set_xlabel("Individuals")

    ax.figure.tight_layout()


def hap_matrix(
    conf,
    data_generator,
    pdf_file,
    aspect=5 / 16,
    scale=1.0,
    cmap="viridis",
    rasterized=False,
):
    """
    Plot haplotype matrices.
    """
    scale = conf.get("eval.plot.scale", scale)
    scale = conf.get("eval.genotype_matrices.plot.scale", scale)
    aspect = conf.get("eval.genotype_matrices.plot.aspect", aspect)

    sample_counts = conf.sample_counts(haploid=conf.phased)
    pop_indices = conf.pop_indices(haploid=conf.phased)

    pdf = PdfPages(pdf_file)

    for (A, title) in data_generator:
        fig_w, fig_h = plt.figaspect(aspect)
        fig, ax = plt.subplots(1, 1, figsize=(scale * fig_w, scale * fig_h))
        hap_matrix1(
            A,
            ax=ax,
            title=title,
            sample_counts=sample_counts,
            pop_indices=pop_indices,
            sequence_length=conf.sequence_length,
            cmap=cmap,
            rasterized=rasterized,
            phased=conf.phased,
            ploidy=conf.ploidy,
        )
        pdf.savefig(figure=fig)
        plt.close(fig)

    pdf.close()


def ts_hap_matrix(
    conf,
    data,
    pred,
    metadata,
    pdf_file,
    n_examples=10,
):
    """
    Plot haplotype matrices for each modelspec. Plot up to n_examples for each.
    """
    assert n_examples >= 1
    modelspecs = list(itertools.chain(*conf.tranche.values()))
    modelspec_indexes = collections.defaultdict(list)
    want_more = set(modelspecs)

    for i in range(len(data)):
        modelspec = metadata[i]["modelspec"]
        if modelspec in want_more:
            modelspec_indexes[modelspec].append(i)
            if len(modelspec_indexes[modelspec]) == n_examples:
                want_more.remove(modelspec)
                if len(want_more) == 0:
                    break

    _, condition_positive = list(conf.tranche.keys())

    def data_generator():
        for i in list(itertools.chain(*modelspec_indexes.values())):
            meta = metadata[i]
            p = pred[i]
            title = (
                f"{meta['modelspec']}\n"
                f"$T_{{mut}}$={int(round(meta['T_mut']))}, "
                f"$T_{{sel}}$={int(round(meta['T_sel']))}, "
                f"$s$={meta['s']:.4f}, "
                f"Pr{{{condition_positive}}}={p:.4g}"
            )
            yield data[i], title

    hap_matrix(conf, data_generator(), pdf_file)


def vcf_hap_matrix(
    conf,
    vcf_batch_gen,
    pdf_file,
):
    """
    Plot empirical haplotype/genotype matrices.
    """

    def flatten_batches():
        for coords_list, vcf_data in vcf_batch_gen:
            for coords, A in zip(coords_list, vcf_data):
                chrom, start, end = coords
                title = f"{chrom}:{start}$-${end}"
                yield A[..., 0], title

    hap_matrix(conf, flatten_batches(), pdf_file)


# helper functions for tf_keras_vis plots


def _model_modifier(model):
    import tensorflow as tf

    # Change last layer from sigmoid activation to linear.
    # The model will have been cloned before this call, so we modify directly.
    assert model.layers[-1].name == "output"
    model.layers[-1].activation = tf.keras.activations.linear
    return model


def _loss(output):
    return output[:, 0]


def saliency(
    *,
    conf,
    model,
    data,
    pred,
    metadata,
    pdf_file,
    n_examples=300,
    smooth_samples=0,
    aspect=12 / 16,
    scale=1.2,
    cmap="cool",
    rasterized=False,
):
    """
    Plot saliency map for each modelspec. Plot up to n_examples for each.
    """
    from tf_keras_vis.saliency import Saliency
    from tf_keras_vis.utils import normalize

    assert n_examples >= 1
    modelspecs = list(itertools.chain(*conf.tranche.values()))
    modelspec_indexes = collections.defaultdict(list)
    want_more = set(modelspecs)

    for i in range(len(data)):
        modelspec = metadata[i]["modelspec"]
        if modelspec in want_more:
            modelspec_indexes[modelspec].append(i)
            if len(modelspec_indexes[modelspec]) == n_examples:
                want_more.remove(modelspec)
                if len(want_more) == 0:
                    break

    indexes = list(itertools.chain(*modelspec_indexes.values()))

    saliency = Saliency(model, model_modifier=_model_modifier)
    X = np.expand_dims(data[indexes], axis=-1).astype(np.float32)
    saliency_maps = saliency(_loss, X, smooth_samples=smooth_samples)
    saliency_maps = normalize(saliency_maps)

    _, condition_positive = list(conf.tranche.keys())
    sample_counts = conf.sample_counts(haploid=conf.phased)
    pop_indices = conf.pop_indices(haploid=conf.phased)

    pdf = PdfPages(pdf_file)
    fig_w, fig_h = plt.figaspect(aspect)
    fig = plt.figure(figsize=(scale * fig_w, scale * fig_h))

    nrows = len(modelspecs) + 1
    gs = fig.add_gridspec(
        nrows=nrows,
        ncols=2,
        height_ratios=[8] * (nrows - 1) + [1],
        width_ratios=[40, 1],
    )
    axs = [fig.add_subplot(gs[i, 0]) for i in range(nrows)]
    ax_cbar = fig.add_subplot(gs[:-1, 1])
    axs[-1].get_shared_x_axes().join(axs[-1], axs[-2])
    ax_pops = axs[-1]

    i = 0
    for ax, modelspec in zip(axs, modelspecs):
        # Take average over maps with the same modelspec.
        saliency_map = np.mean(saliency_maps[i : i + n_examples], axis=0)
        i += n_examples

        last_subplot = ax == axs[-2]
        title = modelspec.split("/")[-2]
        hap_matrix1(
            saliency_map,
            ax=ax,
            title=title,
            sample_counts=sample_counts,
            pop_indices=pop_indices,
            sequence_length=conf.sequence_length,
            cmap=cmap,
            rasterized=rasterized,
            phased=conf.phased,
            ploidy=conf.ploidy,
            vmin=0,
            vmax=1,
            plot_cbar=last_subplot,
            ax_cbar=ax_cbar if last_subplot else None,
            cbar_label="abs(Gradient)",
            plot_pop_labels=last_subplot,
            ax_pops=ax_pops if last_subplot else None,
        )

    pdf.savefig(figure=fig)
    plt.close(fig)
    pdf.close()
