#!/usr/bin/env python3

import os
import argparse
import logging
import itertools
import functools
import concurrent.futures
import random
import tempfile
import pathlib

import numpy as np

from genomatnn import (
    config,
    convert,
    calibrate,
    sim,
    vcf,
    plots,
)


logger = logging.getLogger(__name__)


def _sim_wrapper(args, conf=None):
    modelspec, seed = args
    ts = sim.sim(
        modelspec,
        sequence_length=conf.sequence_length,
        min_allele_frequency=conf.min_allele_frequency,
        seed=seed,
        sample_counts=conf.sample_counts(haploid=True),
    )
    assert ts is not None
    odir = conf.dir / modelspec
    odir.mkdir(parents=True, exist_ok=True)
    ofile = odir / f"{seed}.trees"
    ts.dump(str(ofile))


def do_sim(conf):
    if conf.list:
        # Just print available modelspecs and exit
        for modelspec in sim._models().keys():
            print(modelspec)
        return
    parallelism = conf.parallelism if conf.parallelism > 0 else os.cpu_count()
    if conf.modelspec is not None:
        modelspecs = [conf.modelspec]
    else:
        modelspecs = list(itertools.chain(*conf.tranche.values()))
    sim_func = functools.partial(_sim_wrapper, conf=conf)
    rng = random.Random(conf.seed)

    def sim_func_arg_generator():
        for _ in range(conf.num_reps):
            for spec in modelspecs:
                yield spec, rng.randrange(1, 2 ** 32)

    with concurrent.futures.ProcessPoolExecutor(parallelism) as ex:
        for _ in ex.map(sim_func, sim_func_arg_generator()):
            pass


def do_train(conf):
    rng = random.Random(conf.seed)
    cache = conf.dir / f"zarrcache_{conf.num_rows}-rows"
    # Translate ref_pop and pop_indices to tree sequence population indices.
    ref_pop = conf.pop2tsidx[conf.ref_pop]
    pop_indices = {
        conf.pop2tsidx[pop]: idx
        for pop, idx in conf.pop_indices(haploid=conf.phased).items()
    }
    parallelism = conf.parallelism if conf.parallelism > 0 else os.cpu_count()
    af_filter = conf.get("train.af_filter")
    if af_filter is not None:
        filter_pop = conf.pop2tsidx[af_filter["pop"]]
        filter_modelspec = af_filter["modelspec"]
        filter_AF = af_filter["AF"]
    else:
        filter_pop = None
        filter_modelspec = None
        filter_AF = 0
    data = convert.prepare_training_data(
        path=conf.dir,
        tranche=conf.tranche,
        pop_indices=pop_indices,
        ref_pop=ref_pop,
        num_rows=conf.num_rows,
        num_cols=conf.num_cols,
        num_haplotypes=conf.num_haplotypes,
        rng=rng,
        parallelism=parallelism,
        maf_thres=conf.maf_threshold,
        cache=cache,
        train_frac=conf.get("train.train_frac", 0.9),
        phased=conf.phased,
        ploidy=conf.ploidy,
        filter_pop=filter_pop,
        filter_modelspec=filter_modelspec,
        filter_AF=filter_AF,
    )
    train_data, train_labels, _, val_data, val_labels, _ = data
    n_train = train_data.shape[0]
    n_val = val_data.shape[0]
    logger.debug(
        f"Loaded {n_train+n_val} instances with {n_train}/{n_val} "
        "training/validation split."
    )
    if conf.convert_only:
        return

    from genomatnn import tfstuff

    conf.nn_hdf5_file = str(conf.dir / f"{conf.nn_model}_{conf.seed}.hdf5")
    tfstuff.train(conf, train_data, train_labels, val_data, val_labels)
    do_eval(conf)


def do_eval(conf):
    cache = conf.dir / f"zarrcache_{conf.num_rows}-rows"
    data = convert.load_data_cache(cache)
    convert.check_data(data, conf.tranche, conf.num_rows, conf.num_cols)
    train_data, train_labels, train_metadata, val_data, val_labels, val_metadata = data

    extra_sims = conf.get("sim.extra")
    extra_labels = None
    extra_metadata = None
    if extra_sims is not None and len(extra_sims) == 0:
        extra_sims = None
    if extra_sims is not None:
        extra_cache = conf.dir / f"zarrcache_{conf.num_rows}-rows_extra"
        rng = random.Random(conf.seed)
        # Translate ref_pop and pop_indices to tree sequence population indices.
        ref_pop = conf.pop2tsidx[conf.ref_pop]
        pop_indices = {
            conf.pop2tsidx[pop]: idx
            for pop, idx in conf.pop_indices(haploid=conf.phased).items()
        }
        parallelism = conf.parallelism if conf.parallelism > 0 else os.cpu_count()
        data = convert.prepare_extra(
            path=conf.dir,
            tranche=extra_sims,
            pop_indices=pop_indices,
            ref_pop=ref_pop,
            num_rows=conf.num_rows,
            num_cols=conf.num_cols,
            num_haplotypes=conf.num_haplotypes,
            rng=rng,
            parallelism=parallelism,
            maf_thres=conf.maf_threshold,
            cache=extra_cache,
            phased=conf.phased,
            ploidy=conf.ploidy,
        )
        extra_data, _, extra_metadata = data
        n = len(extra_data)
        extra_labels = np.zeros(n)
        logger.debug(f"Loaded {n} extra validation simulations.")

    plot_dir = pathlib.Path(conf.nn_hdf5_file[: -len(".hdf5")])
    plot_dir.mkdir(parents=True, exist_ok=True)

    logger.debug("Applying tensorflow to validation data...")
    from genomatnn import tfstuff
    import tensorflow as tf
    from tensorflow.keras import models

    tfstuff.tf_config(conf.parallelism)
    model = models.load_model(conf.nn_hdf5_file)
    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        train_pred = model.predict(train_data)
        val_pred = model.predict(val_data)
        extra_pred = None
        if extra_sims is not None:
            extra_pred = model.predict(extra_data)

    if val_pred.shape[1] != 1:
        raise NotImplementedError("Only binary predictions are supported")
    val_pred = val_pred[:, 0]
    train_pred = train_pred[:, 0]
    if extra_pred is not None:
        extra_pred = extra_pred[:, 0]

    hap_pdf = str(plot_dir / "genotype_matrices.pdf")
    plots.ts_hap_matrix(conf, val_data, val_pred, val_metadata, hap_pdf)

    roc_pdf = str(plot_dir / "roc.pdf")
    plots.roc(
        conf=conf,
        labels=val_labels,
        pred=val_pred,
        metadata=val_metadata,
        extra_labels=extra_labels,
        extra_pred=extra_pred,
        extra_metadata=extra_metadata,
        pdf_file=roc_pdf,
    )

    accuracy_pdf = str(plot_dir / "accuracy.pdf")
    plots.accuracy(
        conf,
        val_labels,
        val_pred,
        val_metadata,
        accuracy_pdf,
    )

    confusion_pdf = str(plot_dir / "confusion.pdf")
    plots.confusion(
        conf,
        val_labels,
        val_pred,
        val_metadata,
        confusion_pdf,
    )

    weights = conf.get("calibrate.weights")
    val_upidx = calibrate.resample_indexes(val_metadata["modelspec"], weights)

    resampled_val_labels = val_labels[val_upidx]
    resampled_val_pred = val_pred[val_upidx]

    # Apply various calibrations to the prediction probabilties.
    upidx = calibrate.resample_indexes(train_metadata["modelspec"], weights)
    resampled_train_pred = train_pred[upidx]
    resampled_train_labels = train_labels[upidx]
    preds = [("Uncal.", resampled_val_pred)]
    for cc in calibrate.calibration_classes:
        label = cc.__name__
        cc_pred = (
            cc().fit(resampled_train_pred, resampled_train_labels).predict(val_pred)
        )
        preds.append((label, cc_pred[val_upidx]))

    reliability_pdf = str(plot_dir / "reliability.pdf")
    plots.reliability(conf, resampled_val_labels, preds, reliability_pdf)


def get_predictions(conf, pred_file, samples_file):
    rng = random.Random(conf.seed)
    parallelism = conf.parallelism if conf.parallelism > 0 else os.cpu_count()

    logger.debug("Applying tensorflow to vcf data...")
    from genomatnn import tfstuff
    import tensorflow as tf
    from tensorflow.keras import models

    tfstuff.tf_config(conf.parallelism)
    model = models.load_model(conf.nn_hdf5_file)
    strategy = tf.distribute.MirroredStrategy()

    if conf.calibration is not None:
        cache = conf.dir / f"zarrcache_{conf.num_rows}-rows"
        data = convert.load_data_cache(cache)
        convert.check_data(data, conf.tranche, conf.num_rows, conf.num_cols)
        train_data, train_labels, train_metadata, _, _, _ = data
        with strategy.scope():
            train_pred = model.predict(train_data)
        if train_pred.shape[1] != 1:
            raise NotImplementedError("Only binary predictions are supported")
        train_pred = train_pred[:, 0]
        cal = calibrate.calibrate(conf, train_labels, train_metadata, train_pred)

    logger.debug("Setting up data generator...")
    vcf_batch_gen = vcf.matrix_batches(
        conf.file,
        conf.chr,
        winsize=conf.sequence_length,
        winstep=conf.apply["step"],
        num_rows=conf.num_rows,
        counts=conf.sample_counts(haploid=conf.phased),
        indices=conf.pop_indices(haploid=conf.phased),
        ref_pop=conf.ref_pop,
        samples_file=samples_file,
        min_seg_sites=conf.apply["min_seg_sites"],
        max_missing_thres=conf.apply["max_missing_genotypes"],
        maf_thres=conf.maf_threshold,
        parallelism=parallelism,
        rng=rng,
        phased=conf.phased,
        ploidy=conf.ploidy,
    )

    label = list(conf.tranche.keys())[1]
    with open(pred_file, "w") as f:
        print("chrom", "start", "end", f"Pr{{{label}}}", sep="\t", file=f)
        with strategy.scope():
            for coords, vcf_data in vcf_batch_gen:
                predictions = model.predict_on_batch(vcf_data)
                if conf.calibration is not None:
                    # Calibrate the model predictions.
                    predictions = cal.predict(predictions)
                for (chrom, start, end), pred in zip(coords, predictions):
                    printable_pred = [format(p, ".8f") for p in pred]
                    print(chrom, start, end, *printable_pred, sep="\t", file=f)


def do_apply(conf):
    plot_dir = pathlib.Path(conf.nn_hdf5_file[: -len(".hdf5")])
    plot_dir.mkdir(parents=True, exist_ok=True)
    pred_file = str(plot_dir / "predictions.txt")
    pdf_file = str(plot_dir / "predictions.pdf")
    if not conf.plot_only:
        with tempfile.TemporaryDirectory() as tmpdir:
            samples_file = f"{tmpdir}/samples.txt"
            with open(samples_file, "w") as f:
                print(*conf.vcf_samples, file=f, sep="\n")
            get_predictions(conf, pred_file, samples_file)
    plots.predictions(conf, pred_file, pdf_file)


def parse_regions(filename):
    regions = []
    with open(filename) as f:
        for line in f:
            line = line.rstrip()
            fields = line.split()
            if line[0] == "#" or fields[0] == "chrom":
                # ignore header
                continue
            chrom = fields[0]
            start = int(fields[1])
            end = int(fields[2])
            regions.append((chrom, start, end))
    return regions


def do_vcfplot(conf):
    regions = list(conf.regions)
    if conf.regions_file is not None:
        regions.extend(parse_regions(conf.regions_file))
    if len(regions) == 0:
        raise ValueError("No genomic regions found")
    chr_file = {str(chrom): vcf for vcf, chrom in zip(conf.file, conf.chr)}
    coordinates = [(chr_file[r[0]], *r) for r in regions]

    rng = random.Random(conf.seed)
    parallelism = conf.parallelism if conf.parallelism > 0 else os.cpu_count()

    with tempfile.TemporaryDirectory() as tmpdir:
        samples_file = f"{tmpdir}/samples.txt"
        with open(samples_file, "w") as f:
            print(*conf.vcf_samples, file=f, sep="\n")

        logger.debug("Setting up data generator...")
        vcf_batch_gen = vcf.matrix_batches(
            conf.file,
            conf.chr,
            coordinates=coordinates,
            winsize=conf.sequence_length,
            winstep=conf.apply["step"],
            num_rows=conf.num_rows,
            counts=conf.sample_counts(haploid=conf.phased),
            indices=conf.pop_indices(haploid=conf.phased),
            ref_pop=conf.ref_pop,
            samples_file=samples_file,
            min_seg_sites=conf.apply["min_seg_sites"],
            max_missing_thres=conf.apply["max_missing_genotypes"],
            maf_thres=conf.maf_threshold,
            parallelism=parallelism,
            rng=rng,
            phased=conf.phased,
            ploidy=conf.ploidy,
        )
        plots.vcf_hap_matrix(conf, vcf_batch_gen, conf.pdf_file)


def parse_args(args_list):
    parser = argparse.ArgumentParser(
        description="Simulate, train, and apply a CNN to genotype matrices."
    )
    subparsers = parser.add_subparsers(dest="subcommand")
    subparsers.required = True

    sim_parser = subparsers.add_parser("sim", help="Simulate tree sequences.")
    sim_parser.set_defaults(func=do_sim)
    train_parser = subparsers.add_parser("train", help="Train a CNN.")
    train_parser.set_defaults(func=do_train)
    eval_parser = subparsers.add_parser("eval", help="Evaluate trained CNN.")
    eval_parser.set_defaults(func=do_eval)
    apply_parser = subparsers.add_parser("apply", help="Apply trained CNN.")
    apply_parser.set_defaults(func=do_apply)
    vcfplot_parser = subparsers.add_parser(
        "vcfplot", help="Plot haplotype/genotype matrices from a VCF/BCF."
    )
    vcfplot_parser.set_defaults(func=do_vcfplot)

    # Arguments common to all subcommands.
    for i, p in enumerate(
        (sim_parser, train_parser, eval_parser, apply_parser, vcfplot_parser)
    ):

        p.add_argument(
            "-j",
            "--parallelism",
            default=0,
            type=int,
            help="Number of processes or threads to use for parallel things. "
            "E.g. simultaneous simulations, or the number of threads "
            "used by tensorflow when running on CPU. "
            "If set to zero, os.cpu_count() is used. "
            "[default=%(default)s].",
        )
        p.add_argument(
            "-s",
            "--seed",
            default=random.randrange(1, 2 ** 32),
            type=int,
            help="Seed for the random number generator [default=%(default)s].",
        )
        p.add_argument(
            "-v",
            "--verbose",
            default=0,
            action="count",
            help="Increase verbosity. Specify twice for messages from "
            "third party libraries (e.g. tensorflow and matplotlib).",
        )
        p.add_argument(
            "conf", metavar="conf.toml", type=str, help="Configuration file."
        )

    sim_parser.add_argument(
        "-n",
        "--num-reps",
        default=1,
        type=int,
        help="Number of replicate simulations. For each replicate, one "
        "simulation is run for each modelspec. "
        "[default=%(default)s]",
    )
    sim_parser.add_argument(
        "-l",
        "--list",
        action="store_true",
        default=False,
        help="List available model specifications.",
    )
    sim_parser.add_argument(
        "modelspec",
        nargs="?",
        default=None,
        help="Model specification to simulated. "
        "If not provided, modelspecs from the config file will be simulated",
    )

    train_parser.add_argument(
        "-c",
        "--convert-only",
        default=False,
        action="store_true",
        help="Convert simulated tree sequences into genotype matrices "
        "ready for training, and then exit.",
    )

    eval_parser.add_argument(
        "nn_hdf5_file",
        metavar="nn.hdf5",
        type=str,
        help="The trained neural network model to evaulate.",
    )

    apply_parser.add_argument(
        "-p",
        "--plot-only",
        default=False,
        action="store_true",
        help="Just make the plots.",
    )
    apply_parser.add_argument(
        "nn_hdf5_file",
        metavar="nn.hdf5",
        type=str,
        help="The trained neural network model to apply.",
    )

    def region_type(x, parser=vcfplot_parser):
        err = (
            "Region must have the format chrom:a-b "
            "where a and b are genomic coordinates"
        )
        fields = x.split(":")
        if len(fields) != 2:
            parser.error(err)
        chrom, rest = fields
        fields = rest.split("-")
        if len(fields) != 2:
            parser.error(err)
        try:
            from_ = int(fields[0])
            to = int(fields[1])
        except TypeError:
            parser.error(err)
        return chrom, from_, to

    vcfplot_parser.add_argument(
        "-r",
        "--regions-file",
        type=str,
        help="bcftools-like regions to be plotted.",
    )
    vcfplot_parser.add_argument(
        "pdf_file",
        metavar="plot.pdf",
        type=str,
        help="Filename of the output file.",
    )
    vcfplot_parser.add_argument(
        "regions",
        nargs="*",
        type=region_type,
        help="bcftools-like region(s) to plot, of the form chrom:a-b",
    )

    args = parser.parse_args(args_list)
    args.conf = config.Config(args.conf)

    if args.parallelism > 0:
        # Set the number of threads used by openblas, MKL, etc.
        os.environ["OMP_NUM_THREADS"] = str(args.parallelism)

    # Pin threads to CPUs when using tensorflow MKL builds.
    if args.verbose >= 2:
        os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"
    else:
        os.environ["KMP_AFFINITY"] = "granularity=fine,noverbose,compact,1,0"

    # Unmute tensorflow.
    if args.verbose >= 2:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

    config.logger_setup(args.verbose)

    # Transfer args to the config
    if args.subcommand == "sim":
        args.conf.num_reps = args.num_reps
        args.conf.list = args.list
        args.conf.modelspec = args.modelspec
    elif args.subcommand == "train":
        args.conf.convert_only = args.convert_only
    elif args.subcommand == "eval":
        args.conf.nn_hdf5_file = args.nn_hdf5_file
    elif args.subcommand == "apply":
        args.conf.plot_only = args.plot_only
        args.conf.nn_hdf5_file = args.nn_hdf5_file
    elif args.subcommand == "vcfplot":
        if len(args.regions) == 0 and args.regions_file is None:
            vcfplot_parser.error("Must specify a region to plot.")
        args.conf.regions = args.regions
        args.conf.regions_file = args.regions_file
        args.conf.pdf_file = args.pdf_file
    args.conf.parallelism = args.parallelism
    args.conf.seed = args.seed
    args.conf.verbose = args.verbose

    return args


def main(args_list=None):
    random.seed()
    args = parse_args(args_list)
    args.func(args.conf)


if __name__ == "__main__":
    main()
