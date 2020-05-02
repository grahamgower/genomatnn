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


_module_name = "genomatnn"
__version__ = "0.1"

logger = logging.getLogger(__name__)


def _sim_wrapper(args, config=None):
    modelspec, seed = args
    ts = sim.sim(
            modelspec, config.sequence_length, config.min_allele_frequency,
            seed=seed, sample_counts=config.sample_counts())
    assert ts is not None
    odir = config.dir / modelspec
    odir.mkdir(parents=True, exist_ok=True)
    ofile = odir / f"{seed}.trees"
    ts.dump(str(ofile))


def do_sim(config):
    modelspecs = list(itertools.chain(*config.tranche.values()))
    sim_func = functools.partial(_sim_wrapper, config=config)
    rng = random.Random(config.seed)

    def sim_func_arg_generator():
        for _ in range(config.num_reps):
            for spec in modelspecs:
                yield spec, rng.randrange(1, 2**32)

    with concurrent.futures.ProcessPoolExecutor(config.parallelism) as ex:
        for _ in ex.map(sim_func, sim_func_arg_generator()):
            pass


def do_train(config):
    import tfstuff
    tfstuff.config(config.parallelism)
    raise NotImplementedError("'train' not yet connected to CLI")


def do_apply(config):
    import tfstuff
    tfstuff.config(config.parallelism)
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

    sim_parser.add_argument(
            "-n", "--num-reps", default=1, type=int,
            help="Number of replicate simulations. For each replicate, one "
                 "simulation is run for each modelspec. "
                 "[default=%(default)s]")

    for p in (sim_parser, train_parser, apply_parser):
        p.add_argument(
                "-p", "--parallelism", default=os.cpu_count(), type=int,
                help="Number of processes or threads to use for parallel things. "
                     "E.g. simultaneous simulations, or the number of threads "
                     "used by tensorflow when running on CPU. "
                     "[default=%(default)s].")
        p.add_argument(
                "-v", "--verbose", default=False, action="store_true",
                help="Increase verbosity to debug level.")
        p.add_argument(
                "-s", "--seed", default=None, type=int,
                help="Seed for the random number generator.")
        p.add_argument(
                "config", metavar="config.toml", type=config.Config,
                help="Configuration file.")

    args = parser.parse_args()

    if args.seed is None:
        random.seed()
        args.seed = random.randrange(1, 2**32)

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
        args.config.num_reps = args.num_reps
    args.config.parallelism = args.parallelism
    args.config.seed = args.seed

    return args


if __name__ == "__main__":
    args = parse_args()
    args.func(args.config)
