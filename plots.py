import itertools
import operator
import collections

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

import vcf  # noqa: E402
import calibrate  # noqa: E402


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

    ax.hlines(p_thres, -0, chrlen, linestyle="--", color="gray", lw=0.5)
    ax.grid(linestyle="-", color="gray", lw=0.1)

    start = header[1]
    end = header[2]
    label = header[3]
    midpos = (preds[start] + preds[end]) // 2
    p = preds[label]
    idx1 = np.where(p > p_thres)
    idx2 = np.where(np.logical_not(p > p_thres))
    i = 0
    ax.scatter(
        midpos[idx1],
        p[idx1],
        edgecolor=ec[i],
        facecolor=fc[i],
        marker=sym2[i],
        label=None,
        s=sz2[i],
        lw=0.5,
    )
    ax.scatter(
        midpos[idx2],
        p[idx2],
        edgecolor=ec2[i],
        facecolor=fc2[i],
        marker=sym[i],
        label=label,
        s=sz1[i],
        lw=0.5,
    )
    ax.set_ylabel(label)

    xt_interval = 1e7
    if chrlen > 1e8:
        xt_interval *= 2
    if chrlen > 2e8:
        xt_interval *= 2
    xticks = np.arange(xt_interval, chrlen, xt_interval)
    xlabels = [str(int(x / 1e6)) + " mbp" for x in xticks]
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
        dtype = [(header[0], chr_dtype), (header[1], int), (header[2], int)] + [
            (h, float) for h in header[3:]
        ]
        preds = np.loadtxt(f, dtype=dtype)
    preds_by_chr = dict()
    for k, g in itertools.groupby(preds, operator.itemgetter(0)):
        preds_by_chr[k] = np.fromiter(g, dtype=dtype)
    return header, preds_by_chr


def predictions(conf, pred_file, pdf_file, aspect=9 / 16, scale=1.0):
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


def roc(conf, labels, pred, metadata, pdf_file, aspect=10 / 16, scale=1.5, inset=False):

    (
        (condition_negative, false_modelspecs),
        (condition_positive, true_modelspecs),
    ) = list(conf.tranche.items())

    tp = pred[np.where(labels == 1)]
    fp_list = []
    for fp_modelspec in false_modelspecs:
        fp = pred[np.where(metadata["modelspec"] == fp_modelspec)]
        fp_list.append(fp)

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
        for i, ch in zip((50, 90), ("o", "x")):
            if ch == "x":
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
        [
            Line2D(
                [0],
                [0],
                marker="o",
                c="none",
                markerfacecolor="none",
                markeredgecolor="k",
                markersize=10,
                label=f"Pr{{{condition_positive}}} > 0.50",
            ),
            Line2D(
                [0],
                [0],
                marker="x",
                c="none",
                markerfacecolor="k",
                markeredgecolor="k",
                markersize=10,
                label=f"Pr{{{condition_positive}}} > 0.90",
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

    # axs[2].set_title("Specificity vs. Negative Predictive Value")
    axs[2].set_title("TNR-NPR")
    axs[2].set_xlabel("TNR: TN/(TN+FP)")
    axs[2].set_ylabel("NPV: TN/(TN+FN)")

    fig.tight_layout()
    pdf.savefig(figure=fig)
    plt.close(fig)
    pdf.close()


def partition2d(x, y, z, bins):
    # Add a small value to the max, so that the max data point
    # contributes to the last bin.
    xmax = np.max(x) + 1e-9
    ymax = np.max(y) + 1e-9
    xitv = xmax / bins
    yitv = ymax / bins

    binned = collections.defaultdict(list)
    for xi, yi, zi in zip(x, y, z):
        _x = xi - (xi % xitv)
        _y = yi - (yi % yitv)
        binned[(_x, _y)].append(zi)

    return binned, xitv, yitv


def accuracy(
    conf, labels, pred, metadata, pdf_file, aspect=10 / 16, scale=1.5, bins=20
):

    aspect = conf.get("eval.plot.aspect", aspect)
    scale = conf.get("eval.plot.scale", scale)
    aspect = conf.get("eval.accuracy.plot.aspect", aspect)
    scale = conf.get("eval.accuracy.plot.scale", scale)
    bins = conf.get("eval.accuracy.plot.bins", bins)

    pdf = PdfPages(pdf_file)
    fig_w, fig_h = plt.figaspect(aspect)
    fig, ax = plt.subplots(1, 1, figsize=(scale * fig_w, scale * fig_h))

    x = metadata["s"]
    y = metadata["T_sel"]
    z = 1 - np.abs(labels - pred)  # accuracy

    binned, xitv, yitv = partition2d(x, y, z, bins)

    boxes = []
    colours = []
    for (xi, yi), zi_array in binned.items():
        r = Rectangle((xi, yi), xitv, yitv)
        boxes.append(r)
        colours.append(np.mean(zi_array))

    pc = PatchCollection(boxes, rasterized=True)
    pc.set_array(np.array(colours))
    ax.add_collection(pc)

    ax.set_title("Accuracy")
    ax.set_xlabel("$s$")
    ax.set_ylabel("$T_{sel}$ (years ago)")

    ax.set_xlim([0, bins * xitv])
    ax.set_ylim([0, bins * yitv])

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


def confusion(conf, labels, pred, metadata, pdf_file, aspect=10 / 16, scale=1.5):
    """
    Confusion matrices.
    """

    aspect = conf.get("eval.plot.aspect", aspect)
    scale = conf.get("eval.plot.scale", scale)
    aspect = conf.get("eval.confusion.plot.aspect", aspect)
    scale = conf.get("eval.confusion.plot.scale", scale)

    pdf = PdfPages(pdf_file)
    fig_w, fig_h = plt.figaspect(aspect)
    fig, axs = plt.subplots(1, 2, figsize=(scale * fig_w, scale * fig_h))

    modelspecs = list(itertools.chain(*conf.tranche.values()))
    assert set(modelspecs) == set(np.unique(metadata["modelspec"]))

    n_labels = 2
    n_modelspecs = len(modelspecs)
    cm_labels = np.empty(shape=(n_labels, n_labels))
    cm_modelspecs = np.empty(shape=(n_labels, n_modelspecs))

    # labels x labels
    for i in range(n_labels):
        idx = np.where(np.abs(pred - i) < 0.5)[0]
        n_pred = len(idx)
        for j in range(n_labels):
            n_true = len(np.where(labels[idx] == j)[0])
            cm_labels[i, j] = n_true / n_pred

    # labels x modelspecs
    for i in range(n_labels):
        idx = np.where(np.abs(pred - i) < 0.5)[0]
        n_pred = len(idx)
        for j in range(n_modelspecs):
            n_true = len(np.where(metadata["modelspec"][idx] == modelspecs[j])[0])
            cm_modelspecs[i, j] = n_true / n_pred

    modelspec_pfx = longest_common_prefix(modelspecs)
    short_modelspecs = [mspec[len(modelspec_pfx) :] for mspec in modelspecs]
    tranch_keys = list(conf.tranche.keys())

    confusion1(axs[0], cm_labels, tranch_keys, tranch_keys, cbar=False)
    confusion1(axs[1], cm_modelspecs, tranch_keys, short_modelspecs)

    fig.suptitle("Confusion matrices")
    fig.tight_layout()
    pdf.savefig(figure=fig)
    plt.close(fig)
    pdf.close()


def reliability(conf, labels, preds, pdf_file, aspect=10 / 16, scale=1.5, bins=10):
    """
    Reliability plot., aka calibration curve.
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
    A, ax, title, sample_counts, pop_indices, sequence_length, aspect, cmap, rasterized
):
    im = ax.imshow(
        A,
        interpolation="none",
        origin="lower",
        rasterized=rasterized,
        # left, right, bottom, top
        extent=(0, A.shape[1], 0, A.shape[0] * aspect),
        cmap=cmap,
        norm=matplotlib.colors.PowerNorm(0.5, vmax=5),
    )
    shrink = 0.8
    cb = ax.figure.colorbar(
        im,
        ax=ax,
        extend="max",
        shrink=shrink,
        pad=shrink * 0.05,
        # pad=0.01,
        # fraction=0.025,
        fraction=shrink * 0.10,
        label="Density of minor alleles",
    )
    cb.set_ticks([])

    ax.set_title(title)
    ax.set_ylabel("Genomic position")  # , labelpad=-10)
    ax.set_yticks([0, ax.get_ylim()[1]])
    ax.set_yticklabels(["$0\,$kb", f"${sequence_length//1000}\,$kb"])  # noqa: W605

    plt.setp(ax.get_xticklabels(), visible=False)
    ax.tick_params(axis="x", length=0)

    #
    # Add population-labels colour bar on a new axis.
    #
    divider = axes_grid1.make_axes_locatable(ax)
    ax_pops = divider.append_axes("bottom", 0.2, pad=0, sharex=ax)
    ax_pops.set_ylim([0, 1])

    pop_ticks = []
    boxes = []
    colours = plt.get_cmap("tab10").colors
    for index, count in zip(pop_indices.values(), sample_counts.values()):
        r = Rectangle((index, 0), count, 1)
        boxes.append(r)
        pop_ticks.append(index + count / 2)

    pc = PatchCollection(boxes, fc=colours, rasterized=rasterized)
    ax_pops.add_collection(pc)

    ax_pops.set_yticks([])
    ax_pops.set_yticklabels([])
    ax_pops.set_xticks(pop_ticks)
    ax_pops.set_xticklabels(list(pop_indices.keys()))
    ax_pops.set_xlabel("Haplotypes")

    # ax.figure.subplots_adjust(top=0.8)
    ax.figure.tight_layout()


def hap_matrix(
    conf,
    data,
    pred,
    metadata,
    pdf_file,
    n_examples=10,
    aspect=5 / 16,
    scale=1.0,
    cmap="Greys",
    rasterized=False,
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

    scale = conf.get("eval.plot.scale", scale)
    scale = conf.get("eval.genotype_matrices.plot.scale", scale)
    aspect = conf.get("eval.genotype_matrices.plot.aspect", aspect)

    _, condition_positive = list(conf.tranche.keys())
    pdf = PdfPages(pdf_file)

    for i in itertools.chain(*modelspec_indexes.values()):
        A = data[i]
        params = metadata[i]
        p = pred[i]
        sample_counts = conf.sample_counts()
        pop_indices = conf.pop_indices()

        fig_w, fig_h = plt.figaspect(aspect)
        fig, ax = plt.subplots(1, 1, figsize=(scale * fig_w, scale * fig_h))

        title = (
            f"{params['modelspec']}\n"
            f"$T_{{mut}}$={int(round(params['T_mut']))}, "
            f"$T_{{sel}}$={int(round(params['T_sel']))}, "
            f"$s$={params['s']:.4f}, "
            f"Pr{{{condition_positive}}}={p:.4g}"
        )
        hap_matrix1(
            A,
            ax,
            title,
            sample_counts,
            pop_indices,
            conf.sequence_length,
            aspect,
            cmap,
            rasterized,
        )

        pdf.savefig(figure=fig)
        plt.close(fig)

    pdf.close()
