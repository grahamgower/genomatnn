import re
import sys
import string
import logging
import pathlib
import itertools

import numpy as np
import toml

import genomatnn
from genomatnn import (
    sim,
    calibrate,
)


class _CLIFormatter(logging.Formatter):
    def format(self, record):
        if record.levelno == logging.INFO:
            self._style = logging.PercentStyle("%(message)s")
        else:
            self._style = logging.PercentStyle("%(levelname)s: %(message)s")
        if record.levelno == logging.WARNING and len(record.args) > 0:
            # trim the ugly warnings.warn message
            match = re.search(
                r"Warning:\s*(.*?)\s*warnings.warn\(", record.args[0], re.DOTALL
            )
            if match is not None:
                record.args = (match.group(1),)
        return super().format(record)


def logger_setup(verbosity):
    level = "INFO"
    if verbosity >= 1:
        level = "DEBUG"
    logging.captureWarnings(True)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_CLIFormatter())
    # messages from genomatnn
    logger = logging.getLogger(genomatnn.__name__)
    logger.setLevel(level)
    # messages from non-genomatnn imports
    root_level = "WARNING"
    if verbosity >= 2:
        root_level = "DEBUG"
    logging.basicConfig(handlers=[handler], level=root_level)


class ConfigError(RuntimeError):
    pass


class Config:
    def __init__(self, filename):

        # Attributes for external use.
        self.filename = filename
        self.dir = None
        self.ref_pop = None
        self.maf_threshold = None
        self.num_rows = None
        self.num_haplotypes = None
        self.pop = None
        self.sequence_length = None
        self.min_allele_frequency = None
        self.vcf_samples = None
        self.tranche = None
        self.chr = None
        self.pop2tsidx = None
        self.calibration = None
        self.ploidy = 2  # TODO: find and fix ploidy=2 assumptions.

        # Read in toml file and fill in the attributes.
        self.config = toml.load(self.filename)
        self._getcfg_toplevel()
        self._getcfg_pop()
        self._getcfg_sim()
        self._getcfg_vcf()
        self._getcfg_train()
        self._getcfg_eval()
        self._getcfg_apply()
        self._getcfg_calibration()

        self.phased = self.get("vcf.phased", False)
        self.num_cols = self.num_haplotypes
        if not self.phased:
            self.num_cols = self.num_haplotypes // self.ploidy

        # TODO: Add introspection to check required attributes are set.
        #       Check types? Use attrs somehow?

    def _verify_keys_exist(self, dict_, keys, pfx=""):
        for k in keys:
            if k not in dict_:
                raise ConfigError(f"{self.filename}: {pfx}{k} not defined.")

    def _getcfg_toplevel(self):
        toplevel_keys = [
            "dir",
            "ref_pop",
            "maf_threshold",
            "pop",
            "sim",
            "vcf",
            "train",
            "apply",
        ]
        self._verify_keys_exist(self.config, toplevel_keys)
        self.dir = pathlib.Path(self.config["dir"])
        self.ref_pop = self.config["ref_pop"]
        self.maf_threshold = self.config["maf_threshold"]

    def _getcfg_pop(self):
        pop = self.config.get("pop")
        if len(pop) == 0:
            raise ConfigError(f"{self.filename}: no populations defined.")
        for k, v in pop.items():
            if not isinstance(v, list):
                pop[k] = []
                with open(v) as f:
                    for line in f:
                        pop[k].append(line.strip())
        if self.ref_pop not in pop:
            raise ConfigError(
                f"{self.filename}: ref_pop {self.ref_pop} not among those "
                f"to be used for the genotype matrix: {pop}."
            )
        self.pop = pop
        self.vcf_samples = list(itertools.chain(*self.pop.values()))
        self.num_haplotypes = self.ploidy * len(self.vcf_samples)

    def _getcfg_sim(self):
        self.sim = self.config.get("sim")
        if len(self.sim) == 0:
            raise ConfigError(f"{self.filename}: no simulations defined.")
        sim_keys = ["sequence_length", "min_allele_frequency", "tranche"]
        self._verify_keys_exist(self.sim, sim_keys, "sim.")
        self.sequence_length = self.sim["sequence_length"]
        self.min_allele_frequency = self.sim["min_allele_frequency"]
        self._getcfg_tranche(self.sim["tranche"])

    def _getcfg_tranche(self, tranche):
        assert tranche is not None
        if len(tranche) == 0:
            raise ConfigError(f"{self.filename}: no tranches defined.")
        for label, speclist in tranche.items():
            for modelspec in speclist:
                model = sim.get_demog_model(modelspec)
                pop2tsidx = {p.id: i for i, p in enumerate(model.populations)}
                if self.pop2tsidx is None:
                    self.pop2tsidx = pop2tsidx
                elif pop2tsidx.items() != self.pop2tsidx.items():
                    raise ConfigError(
                        f"{self.filename} populations defined for {modelspec} "
                        "do not match earlier modelspecs."
                    )
                for pop in self.pop.keys():
                    if pop not in pop2tsidx:
                        raise ConfigError(
                            f"{self.filename}: {pop} not found in {modelspec}. "
                            f"Options are: {list(pop2tsidx.keys())}"
                        )
        self.tranche = tranche

    def _getcfg_vcf(self):
        self.vcf = self.config.get("vcf")
        if len(self.vcf) == 0:
            raise ConfigError(f"{self.filename}: no vcf defined.")
        vcf_keys = ["file", "chr"]
        self._verify_keys_exist(self.vcf, vcf_keys, "vcf.")
        file_ = self.vcf.get("file")
        self.chr = self.vcf.get("chr")
        self.file = []
        for c in self.chr:
            fn = string.Template(file_).substitute(chr=c)
            self.file.append(pathlib.Path(fn))
        for fn in self.file:
            if not fn.exists():
                raise ConfigError(f"{fn}: file not found.")

    def _getcfg_train(self):
        train = self.config.get("train")
        if len(train) == 0:
            raise ConfigError(f"{self.filename}: no training details defined.")
        train_keys = ["num_rows", "epochs", "batch_size", "model"]
        self._verify_keys_exist(train, train_keys, "train.")
        self.num_rows = train.get("num_rows")
        self.train_epochs = train.get("epochs")
        self.train_batch_size = train.get("batch_size")
        self.nn_model = train.get("model")

        nn_model_keys = {
            "cnn": [
                "n_conv",
                "n_conv_filt",
                "filt_size_x",
                "filt_size_y",
                "n_dense",
                "dense_size",
            ],
        }
        params = train.get(self.nn_model)
        if params is None or self.nn_model not in nn_model_keys:
            raise ConfigError(
                f"{self.filename}: train.model must be set to one of: "
                f"{list(nn_model_keys.keys())}"
            )
        self._verify_keys_exist(
            params, nn_model_keys[self.nn_model], "train.{self.nn_model}"
        )
        self.nn_model_params = params

    def _getcfg_eval(self):
        self.eval = self.config.get("eval")
        if self.eval is not None:
            pass

    def _getcfg_apply(self):
        self.apply = self.config.get("apply")
        apply_keys = ["step", "batch_size", "max_missing_genotypes", "min_seg_sites"]
        self._verify_keys_exist(self.apply, apply_keys, "apply.")

    def _getcfg_calibration(self):
        calib_str = self.config.get(
            "calibration", calibrate.calibration_classes[0].__name__
        )
        for cc in calibrate.calibration_classes:
            if cc.__name__ == calib_str:
                self.calibration = cc
                break
        else:
            if cc != "None":
                raise ConfigError(
                    f"{self.filename}: invalid calibration method {calib_str}"
                )

    def __getitem__(self, key):
        key_fields = key.split(".")
        attr = self.config
        for k in key_fields:
            attr = attr.get(k)
            if attr is None:
                raise KeyError(f"{self.filename}: {key}: not found.")
        return attr

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def sample_counts(self, haploid=True):
        """
        Sample numbers for each population.
        """
        if haploid:
            # Return the number of haploid chromosomes.
            x = self.ploidy
        if not haploid:
            # Return the number of individuals.
            x = 1
        return {p: x * len(v) for p, v in self.pop.items()}

    def pop_indices(self, haploid=True):
        """
        Indices partitioning the populations in a haplotype/genotype matrix.
        """
        indices = np.cumsum(list(self.sample_counts(haploid=haploid).values()))
        # Record the starting indices.
        indices = [0] + list(indices[:-1])
        return dict(zip(self.pop.keys(), indices))
