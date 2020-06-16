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

import config
import convert
import calibrate
import sim
import vcf
import plots


_module_name = "genomatnn"
__version__ = "0.1"

logger = logging.getLogger(__name__)


def _sim_wrapper(args, conf=None):
    modelspec, seed = args
    ts = sim.sim(
        modelspec,
        conf.sequence_length,
        conf.min_allele_frequency,
        seed=seed,
        sample_counts=conf.sample_counts(),
    )
    assert ts is not None
    odir = conf.dir / modelspec
    odir.mkdir(parents=True, exist_ok=True)
    ofile = odir / f"{seed}.trees"
    ts.dump(str(ofile))


def do_sim(conf):
    parallelism = conf.parallelism if conf.parallelism > 0 else os.cpu_count()
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
    pop_indices = {conf.pop2tsidx[pop]: idx for pop, idx in conf.pop_indices().items()}
    parallelism = conf.parallelism if conf.parallelism > 0 else os.cpu_count()
    data = convert.prepare_training_data(
        conf.dir,
        conf.tranche,
        pop_indices,
        ref_pop,
        conf.num_rows,
        conf.num_cols,
        rng,
        parallelism,
        conf.maf_threshold,
        cache,
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

    import tfstuff

    tfstuff.train(conf, train_data, train_labels, val_data, val_labels)


def do_eval(conf):
    cache = conf.dir / f"zarrcache_{conf.num_rows}-rows"
    data = convert.load_data_cache(cache)
    convert.check_data(data, conf.tranche, conf.num_rows, conf.num_cols)
    _, _, _, val_data, val_labels, val_metadata = data

    plot_dir = pathlib.Path(conf.nn_hdf5_file[: -len(".hdf5")])
    plot_dir.mkdir(parents=True, exist_ok=True)

    logger.debug("Applying tensorflow to validation data...")
    import tfstuff
    import tensorflow as tf
    from tensorflow.keras import models

    tfstuff.tf_config(conf.parallelism)
    model = models.load_model(conf.nn_hdf5_file)
    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        val_pred = model.predict(val_data)

    if val_pred.shape[1] != 1:
        raise NotImplementedError("Only binary predictions are supported")
    val_pred = val_pred[:, 0]

    if conf.calibration is not None:
        cal = conf.calibration()
        val_pred_cal = cal.fit(val_pred, val_labels).predict(val_pred)
    else:
        val_pred_cal = val_pred

    hap_pdf = str(plot_dir / "genotype_matrices.pdf")
    plots.hap_matrix(conf, val_data, val_pred, val_metadata, hap_pdf)

    roc_pdf = str(plot_dir / "roc.pdf")
    plots.roc(conf, val_labels, val_pred_cal, val_metadata, roc_pdf)

    accuracy_pdf = str(plot_dir / "accuracy.pdf")
    plots.accuracy(conf, val_labels, val_pred_cal, val_metadata, accuracy_pdf)

    confusion_pdf = str(plot_dir / "confusion.pdf")
    plots.confusion(conf, val_labels, val_pred_cal, val_metadata, confusion_pdf)

    # Apply various calibrations to the prediction probabilties.
    # We use only the first half of the validation set for calibrating,
    # and calculate reliability on the second half.
    n = len(val_pred)
    x1, y1 = val_pred[: n // 2], val_labels[: n // 2]
    x2, y2 = val_pred[n // 2 :], val_labels[n // 2 :]
    preds = [("Uncal.", x2)]
    for cc in calibrate.calibration_classes:
        label = cc.__name__
        cc_x_pred = cc().fit(x1, y1).predict(x2)
        preds.append((label, cc_x_pred))

    reliability_pdf = str(plot_dir / "reliability.pdf")
    plots.reliability(conf, y2, preds, reliability_pdf)


def vcf_get1(
    args,
    samples_file=None,
    sequence_length=None,
    num_rows=None,
    min_seg_sites=None,
    max_missing=None,
    maf_thres=None,
    counts=None,
    indices=None,
    ref_pop=None,
):
    (vcf_file, chrom, start, end), seed = args
    rng = random.Random(seed)
    pos, A = vcf.vcf2mat(
        vcf_file,
        samples_file,
        chrom,
        start,
        end,
        rng,
        max_missing_thres=max_missing,
        maf_thres=maf_thres,
    )
    if len(pos) < min_seg_sites:
        return None
    relative_pos = pos - start
    A = vcf.resize(relative_pos, A, sequence_length, num_rows)
    A_sorted = convert.reorder_and_sort(A, counts, indices, indices, ref_pop)
    return A_sorted[np.newaxis, :, :, np.newaxis]


def vcf_batch_generator(
    coordinates,
    sample_list,
    min_seg_sites,
    max_missing,
    maf_thres,
    sequence_length,
    num_rows,
    rng,
    parallelism,
    batch_size,
    counts,
    indices,
    ref_pop,
):
    icoordinates = iter(coordinates)
    with tempfile.TemporaryDirectory() as tmpdir:
        samples_file = f"{tmpdir}/samples.txt"
        with open(samples_file, "w") as f:
            print(*sample_list, file=f, sep="\n")

        vcf_get_func = functools.partial(
            vcf_get1,
            samples_file=samples_file,
            sequence_length=sequence_length,
            num_rows=num_rows,
            min_seg_sites=min_seg_sites,
            max_missing=max_missing,
            maf_thres=maf_thres,
            counts=counts,
            indices=indices,
            ref_pop=ref_pop,
        )

        with concurrent.futures.ProcessPoolExecutor(parallelism) as ex:
            while True:
                batch_coords = list(itertools.islice(icoordinates, batch_size))
                if len(batch_coords) == 0:
                    break
                seeds = [rng.randrange(1, 2 ** 32) for _ in batch_coords]
                pred_coords = []
                B = []
                map_res = ex.map(vcf_get_func, zip(batch_coords, seeds))
                for coords, A in zip(batch_coords, map_res):
                    if A is None:
                        continue
                    pred_coords.append(coords)
                    B.append(A)
                if len(pred_coords) > 0:
                    yield pred_coords, np.concatenate(B)


def get_predictions(conf, pred_file):
    # XXX: each part of the input vcf is loaded window/step times.
    rng = random.Random(conf.seed)
    logger.debug("Generating window coordinates for vcf data...")
    coordinates = vcf.coordinates(
        conf.file, conf.chr, conf.sequence_length, conf.apply["step"]
    )
    logger.debug("Setting up data generator...")
    parallelism = conf.parallelism if conf.parallelism > 0 else os.cpu_count()
    vcf_batch_gen = vcf_batch_generator(
        coordinates,
        conf.vcf_samples,
        conf.apply["min_seg_sites"],
        conf.apply["max_missing_genotypes"],
        conf.maf_threshold,
        conf.sequence_length,
        conf.num_rows,
        rng,
        parallelism,
        conf.apply["batch_size"],
        conf.sample_counts(),
        conf.pop_indices(),
        conf.ref_pop,
    )

    logger.debug("Applying tensorflow to vcf data...")
    import tfstuff
    import tensorflow as tf
    from tensorflow.keras import models

    tfstuff.tf_config(conf.parallelism)
    model = models.load_model(conf.nn_hdf5_file)
    strategy = tf.distribute.MirroredStrategy()

    if conf.calibration is not None:
        cache = conf.dir / f"zarrcache_{conf.num_rows}-rows"
        data = convert.load_data_cache(cache)
        convert.check_data(data, conf.tranche, conf.num_rows, conf.num_cols)
        _, _, _, val_data, val_labels, _ = data
        with strategy.scope():
            val_pred = model.predict(val_data)
        if val_pred.shape[1] != 1:
            raise NotImplementedError("Only binary predictions are supported")
        val_pred = val_pred[:, 0]
        cal = conf.calibration().fit(val_pred, val_labels)

    label = list(conf.tranche.keys())[1]
    with open(pred_file, "w") as f:
        print("chrom", "start", "end", f"Pr{{{label}}}", sep="\t", file=f)
        with strategy.scope():
            for coords, vcf_data in vcf_batch_gen:
                predictions = model.predict_on_batch(vcf_data)
                if conf.calibration is not None:
                    # Calibrate the model predictions.
                    predictions = cal.predict(predictions)
                for (_, chrom, start, end), pred in zip(coords, predictions):
                    printable_pred = [format(p, ".8f") for p in pred]
                    print(chrom, start, end, *printable_pred, sep="\t", file=f)


def do_apply(conf):
    pred_file = conf.nn_hdf5_file[: -len(".hdf5")] + "_predictions.txt"
    pdf_file = conf.nn_hdf5_file[: -len(".hdf5")] + "_predictions.pdf"
    if not conf.plot_only:
        get_predictions(conf, pred_file)
    plots.predictions(conf, pred_file, pdf_file)


def parse_args():
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

    # Arguments common to all subcommands.
    for i, p in enumerate((sim_parser, train_parser, eval_parser, apply_parser)):

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
            default=False,
            action="store_true",
            help="Increase verbosity to debug level.",
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

    train_parser.add_argument(
        "-c",
        "--convert-only",
        default=False,
        action="store_true",
        help="Convert simulated tree sequences into genotype matrices "
        "ready for training, and then exit.",
    )

    for p in (eval_parser, apply_parser):

        def cc_type(cc_str, parser=p):
            for cc in calibrate.calibration_classes:
                if cc.__name__ == cc_str:
                    break
            else:
                if cc == "None":
                    cc = None
                else:
                    raise parser.error(f"Invalid calibration method {cc_str}")

        choices = [cc.__name__ for cc in calibrate.calibration_classes]
        choices.append("None")
        p.add_argument(
            "-C",
            "--calibration",
            choices=choices,
            default=choices[0],
            type=cc_type,
            help="Calibrate model prediction probabilities using the "
            "specifed method [default=%(default)s].",
        )

    eval_parser.add_argument(
        "nn_hdf5_file",
        metavar="nn.hdf5",
        type=str,
        help="The trained nerual network model to evaulate.",
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
        help="The trained nerual network model to apply.",
    )

    args = parser.parse_args()
    args.conf = config.Config(args.conf)

    if args.parallelism > 0:
        # Set the number of threads used by openblas, MKL, etc.
        os.environ["OMP_NUM_THREADS"] = str(args.parallelism)

    # Pin threads to CPUs when using tensorflow MKL builds.
    if args.verbose:
        os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"
    else:
        os.environ["KMP_AFFINITY"] = "granularity=fine,noverbose,compact,1,0"

    if args.verbose:
        config.logger_setup("DEBUG")
        # Unmute tensorflow.
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
    else:
        config.logger_setup("INFO")

    # Transfer args to the config
    if args.subcommand == "sim":
        args.conf.num_reps = args.num_reps
    elif args.subcommand == "train":
        args.conf.convert_only = args.convert_only
    elif args.subcommand == "eval":
        args.conf.calibration = args.calibration
        args.conf.nn_hdf5_file = args.nn_hdf5_file
    elif args.subcommand == "apply":
        args.conf.calibration = args.calibration
        args.conf.plot_only = args.plot_only
        args.conf.nn_hdf5_file = args.nn_hdf5_file
    args.conf.parallelism = args.parallelism
    args.conf.seed = args.seed
    args.conf.verbose = args.verbose

    return args


if __name__ == "__main__":
    random.seed()
    args = parse_args()
    args.func(args.conf)
