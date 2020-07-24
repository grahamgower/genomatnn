import unittest
from unittest import mock
import tempfile
import pathlib
import io
import sys
import contextlib
import shutil

import genomatnn
import config
import sim


@contextlib.contextmanager
def capture(func, *args, **kwargs):
    saved_stdout, saved_stderr = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
        func(*args, **kwargs)
        yield sys.stdout.getvalue(), sys.stderr.getvalue()
    finally:
        sys.stdout = saved_stdout
        sys.stderr = saved_stderr


def build_config(path):
    path = pathlib.Path(path)
    vcf_file = path / "test.vcf"
    config_file = path / "config.toml"
    with open(vcf_file, "w") as f:
        print(
            """\
##fileformat=VCFv4.1
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##contig=<ID=1,length=249250621,assembly=b37>
##reference=file:///path/to/human_g1k_v37.fasta
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	ind0	ind1	ind2	ind3
1	12345	.	A	T	.	.	.	GT	 0|0	0|0	0|0	0|0""",
            file=f,
        )
    with open(config_file, "w") as f:
        print(
            f"""\
dir = "{str(path)}"
ref_pop = "CEU"
maf_threshold = 0.05

[sim]
sequence_length = 100_000
min_allele_frequency = 0.01

[sim.tranche]
trancheA = [
    "HomSap/HomininComposite_4G20/Neutral/slim",
    "HomSap/HomininComposite_4G20/Sweep/CEU",
]
trancheB = [
    "HomSap/HomininComposite_4G20/AI/Nea_to_CEU",
]

[vcf]
chr = [ 1 ]
file = "{str(vcf_file)}"

[pop]
CEU = [ "ind0", "ind3" ]
YRI = [ "ind1", "ind2" ]

[train]
train_frac = 0.5
num_rows = 32
epochs = 1
batch_size = 128
model = "cnn"

[train.cnn]
n_conv = 7
n_conv_filt = 16
filt_size_x = 4
filt_size_y = 4
n_dense = 0
dense_size = 0

[apply]
batch_size = 256
step = 20_000
max_missing_genotypes = 0.1
min_seg_sites = 20
""",
            file=f,
        )
    return vcf_file, config_file


class ConfigMixin:
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.vcf_file, cls.config_file = build_config(cls.temp_dir.name)
        cls.conf = config.Config(cls.config_file)

    @classmethod
    def tearDownClass(cls):
        del cls.temp_dir
        cls.vcf_file = None
        cls.config_file = None


def sim__models(
    modelspec="HomSap/HomininComposite_4G20/Neutral/msprime", models=sim._models()
):
    """
    Return a function to replace sim._models(), which always returns a
    model_func and sim_func corresponding to the provided modelspec.
    """
    model_func, sim_func = models[modelspec]
    new_models = {spec: (model_func, sim_func) for spec in models}
    return lambda: new_models


class TestSim(ConfigMixin, unittest.TestCase):
    def test_missing_config_file(self):
        with self.assertRaises(FileNotFoundError):
            genomatnn.main(["sim", "nonexistent.toml"])

    def test_list_modelspecs(self):
        with capture(genomatnn.main, f"sim -l {self.config_file}".split()) as (
            out,
            err,
        ):
            self.assertEqual(len(err), 0)
            modelspecs = set(out.splitlines())
            for modelspec in [
                "HomSap/HomininComposite_4G20/Neutral/slim",
                "HomSap/HomininComposite_4G20/Sweep/CEU",
                "HomSap/HomininComposite_4G20/AI/Nea_to_CEU",
            ]:
                self.assertTrue(modelspec in modelspecs)

    @mock.patch("sim._models", new=sim__models())
    def test_sim(self):
        genomatnn.main(f"sim -n 2 {self.config_file}".split())
        path = pathlib.Path(self.temp_dir.name)
        ts_files = list(path.glob("**/*.trees"))
        self.assertEqual(len(ts_files), 2 * 3)


class TestTrain(ConfigMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        with mock.patch("sim._models", new=sim__models()):
            genomatnn.main(f"sim -n 10 --seed 1 {cls.config_file}".split())

    def tearDown(self):
        for path in self.conf.dir.glob("*.hdf5"):
            path.unlink()
        for path in self.conf.dir.glob("zarrcache_*"):
            shutil.rmtree(path)

    def test_missing_config_file(self):
        with self.assertRaises(FileNotFoundError):
            genomatnn.main(["train", "nonexistent.toml"])

    @mock.patch("genomatnn.do_eval", autospec=True)
    def test_train(self, mocked_do_eval):
        genomatnn.main(f"train {self.config_file}".split())
        cache_dirs = list(self.conf.dir.glob("zarrcache_*"))
        models = list(self.conf.dir.glob("*.hdf5"))
        self.assertEqual(len(cache_dirs), 1)
        self.assertEqual(len(models), 1)
        self.assertEqual(mocked_do_eval.call_count, 1)

    @mock.patch("genomatnn.do_eval", autospec=True)
    def test_train_from_cache(self, mocked_do_eval):
        genomatnn.main(f"train -c {self.config_file}".split())
        cache_dirs = list(self.conf.dir.glob("zarrcache_*"))
        self.assertEqual(len(cache_dirs), 1)

        genomatnn.main(f"train {self.config_file}".split())
        cache_dirs = list(self.conf.dir.glob("zarrcache_*"))
        models = list(self.conf.dir.glob("*.hdf5"))
        self.assertEqual(len(cache_dirs), 1)
        self.assertEqual(len(models), 1)
        self.assertEqual(mocked_do_eval.call_count, 1)


class TestEval(ConfigMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        with mock.patch("sim._models", new=sim__models()):
            genomatnn.main(f"sim -n 10 --seed 1 {cls.config_file}".split())
            genomatnn.main(f"train --seed 1 {cls.config_file}".split())
        models = list(cls.conf.dir.glob("*.hdf5"))
        assert len(models) == 1
        cls.nn_hdf5_file = str(models[0])

    def test_missing_config_file(self):
        with self.assertRaises(FileNotFoundError):
            genomatnn.main(f"eval nonexistent.toml {self.nn_hdf5_file}".split())

    def test_eval_creates_plots(self):
        genomatnn.main(f"eval {self.config_file} {self.nn_hdf5_file}".split())
        plot_dir = pathlib.Path(self.nn_hdf5_file[: -len(".hdf5")])
        for plot_file in [
            "genotype_matrices.pdf",
            "roc.pdf",
            "accuracy.pdf",
            "confusion.pdf",
            "reliability.pdf",
        ]:
            self.assertTrue((plot_dir / plot_file).exists())


class TestApply(unittest.TestCase):
    pass


class TestVcfplot(unittest.TestCase):
    pass
