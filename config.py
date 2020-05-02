import re
import sys
import string
import logging
import pathlib
import itertools

import toml

import sim
import vcf


class _CLIFormatter(logging.Formatter):
    def format(self, record):
        if record.levelno == logging.INFO:
            self._style = logging.PercentStyle("%(message)s")
        else:
            self._style = logging.PercentStyle("%(levelname)s: %(message)s")
        if record.levelno == logging.WARNING and len(record.args) > 0:
            # trim the ugly warnings.warn message
            match = re.search(
                    r"Warning:\s*(.*?)\s*warnings.warn\(", record.args[0], re.DOTALL)
            if match is not None:
                record.args = (match.group(1),)
        return super().format(record)


def logger_setup(level):
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_CLIFormatter())
    logging.basicConfig(handlers=[handler], level=level)
    logging.captureWarnings(True)


class ConfigError(RuntimeError):
    pass


class Config():
    def __init__(self, filename):
        self.filename = filename
        self.config = toml.load(self.filename)
        self._parse_toplevel()
        self._parse_pop()
        self._parse_sim()
        self._parse_vcf()

    def check_keys(self, dict_, keys, pfx=""):
        for k in keys:
            if k not in dict_:
                raise ConfigError(f"{self.filename}: {pfx}{k} not defined.")

    def _parse_toplevel(self):
        toplevel_keys = ("dir", )
        self.check_keys(self.config, toplevel_keys)
        self.dir = pathlib.Path(self.config["dir"])

    def _parse_pop(self):
        pop = self.config.get("pop")
        if pop is None or len(pop) == 0:
            raise ConfigError(
                    f"{self.filename}: no populations defined.")
        for k, v in pop.items():
            if not isinstance(v, list):
                pop[k] = []
                with open(v) as f:
                    for line in f:
                        pop[k].append(line.strip())
        self.pop = pop
        self.vcf_samples = list(itertools.chain(*self.pop.values()))

    def _parse_sim(self):
        self.sim = self.config.get("sim")
        if self.sim is None or len(self.sim) == 0:
            raise ConfigError(f"{self.filename}: no simulations defined.")
        sim_keys = ("sequence_length", "min_allele_frequency", "tranche")
        self.check_keys(self.sim, sim_keys, "sim.")
        self.sequence_length = self.sim["sequence_length"]
        self.min_allele_frequency = self.sim["sequence_length"]
        self._parse_tranche(self.sim["tranche"])

    def _parse_tranche(self, tranche):
        assert tranche is not None
        if len(tranche) == 0:
            raise ConfigError(f"{self.filename}: no tranches defined.")
        for label, speclist in tranche.items():
            for modelspec in speclist:
                model = sim.get_demog_model(modelspec)
                pops = {p.id for p in model.populations}
                for pop in self.pop.keys():
                    if pop not in pops:
                        raise ConfigError(
                                f"{self.filename}: {pop} not found in {modelspec}. "
                                f"Options are: {pops}")
        self.tranche = tranche

    def _parse_vcf(self):
        self.vcf = self.config.get("vcf")
        if self.vcf is None or len(self.vcf) == 0:
            raise ConfigError(f"{self.filename}: no vcf defined.")
        vcf_keys = ("file", )
        self.check_keys(self.vcf, vcf_keys, "vcf.")
        file_ = self.vcf.get("file")
        chr_ = self.vcf.get("chr")
        if chr_ is None:
            self.file = [pathlib.Path(file_)]
        else:
            self.file = []
            for c in chr_:
                fn = string.Template(file_).substitute(chr=c)
                self.file.append(pathlib.Path(fn))
        for fn in self.file:
            if not fn.exists():
                raise ConfigError(f"{fn}: file not found.")
        self.phasing = vcf.sample_phasing(self.file[0], self.vcf_samples)

    def sample_counts(self):
        return {p: len(v) for p, v in self.pop.items()}


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} config.toml")
    cfg_file = sys.argv[1]
    cfg = Config(cfg_file)
    print(cfg.sample_counts())
    print(cfg.phasing.count(True), cfg.phasing.count(False))
    print(cfg.phasing)
