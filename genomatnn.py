#!/usr/bin/env python3

import os
import argparse
import logging
import itertools
import functools
import concurrent.futures
import random

import config
import sim
import convert


_module_name = "genomatnn"
__version__ = "0.1"

logger = logging.getLogger(__name__)


def _sim_wrapper(args, conf=None):
    modelspec, seed = args
    ts = sim.sim(
            modelspec, conf.sequence_length, conf.min_allele_frequency,
            seed=seed, sample_counts=conf.sample_counts())
    assert ts is not None
    odir = conf.dir / modelspec
    odir.mkdir(parents=True, exist_ok=True)
    ofile = odir / f"{seed}.trees"
    ts.dump(str(ofile))


def do_sim(conf):
    modelspecs = list(itertools.chain(*conf.tranche.values()))
    sim_func = functools.partial(_sim_wrapper, conf=conf)
    rng = random.Random(conf.seed)

    def sim_func_arg_generator():
        for _ in range(conf.num_reps):
            for spec in modelspecs:
                yield spec, rng.randrange(1, 2**32)

    with concurrent.futures.ProcessPoolExecutor(conf.parallelism) as ex:
        for _ in ex.map(sim_func, sim_func_arg_generator()):
            pass


def do_train(conf):
    rng = random.Random(conf.seed)
    cache = conf.dir / "zarr.cache"
    # Translate ref_pop and pop_indices to tree sequence population indices.
    ref_pop = conf.pop2tsidx[conf.ref_pop]
    pop_indices = {conf.pop2tsidx[pop]: idx
                   for pop, idx in conf.pop_indices().items()}
    data = convert.prepare_training_data(
            conf.dir, conf.tranche, pop_indices, ref_pop, conf.num_rows,
            conf.num_cols, rng, conf.parallelism, cache)
    train_data, train_labels, val_data, val_labels = data
    n_train = train_data.shape[0]
    n_val = val_data.shape[0]
    logger.debug(
            f"Loaded {n_train+n_val} instances with {n_train}/{n_val} "
            "training/validation split.")
    if conf.convert_only:
        return

    import tfstuff
    tfstuff.train(conf, train_data, train_labels, val_data, val_labels)


def do_apply(conf):
    raise NotImplementedError("'apply' not yet implemented")


def parse_args():
    parser = argparse.ArgumentParser(
            description="Simulate, train, and apply a CNN to genotype matrices.")
    subparsers = parser.add_subparsers(dest="subcommand")
    subparsers.required = True

    sim_parser = subparsers.add_parser("sim", help="Simulate tree sequences.")
    sim_parser.set_defaults(func=do_sim)
    train_parser = subparsers.add_parser("train", help="Train a CNN.")
    train_parser.set_defaults(func=do_train)
    apply_parser = subparsers.add_parser("apply", help="Apply trained CNN.")
    apply_parser.set_defaults(func=do_apply)

    # Arguments common to all subcommands.
    for i, p in enumerate((sim_parser, train_parser, apply_parser)):

        parallelism_default = 0
        if i == 0:
            parallelism_default = os.cpu_count()
        p.add_argument(
                "-p", "--parallelism", default=parallelism_default, type=int,
                help="Number of processes or threads to use for parallel things. "
                     "E.g. simultaneous simulations, or the number of threads "
                     "used by tensorflow when running on CPU. "
                     "[default=%(default)s].")
        p.add_argument(
                "-s", "--seed", default=random.randrange(1, 2**32), type=int,
                help="Seed for the random number generator [default=%(default)s].")
        p.add_argument(
                "-v", "--verbose", default=False, action="store_true",
                help="Increase verbosity to debug level.")
        p.add_argument(
                "conf", metavar="conf.toml", type=str,
                help="Configuration file.")

    sim_parser.add_argument(
            "-n", "--num-reps", default=1, type=int,
            help="Number of replicate simulations. For each replicate, one "
                 "simulation is run for each modelspec. "
                 "[default=%(default)s]")

    train_parser.add_argument(
            "-c", "--convert-only", default=False, action="store_true",
            help="Convert simulated tree sequences into genotype matrices "
                 "ready for training, and then exit.")

    args = parser.parse_args()
    args.conf = config.Config(args.conf)

    if args.parallelism > 0:
        # Set the number of threads used by openblas, MKL, etc.
        os.environ['OMP_NUM_THREADS'] = str(args.parallelism)

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
    args.conf.parallelism = args.parallelism
    args.conf.seed = args.seed
    args.conf.verbose = args.verbose

    return args


if __name__ == "__main__":
    random.seed()
    args = parse_args()
    args.func(args.conf)
