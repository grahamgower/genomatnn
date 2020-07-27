import unittest
from unittest import mock
import tempfile
import pathlib
import io
import sys
import contextlib
import shutil

import toml

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


class ConfigMixin:
    need_sim_data = False
    need_trained_model = False

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.config_file = pathlib.Path(cls.temp_dir.name) / "config.toml"
        # load the example toml, patch in temp_dir, and write it back out
        d = toml.load("examples/test-example.toml")
        d["dir"] = cls.temp_dir.name
        with open(cls.config_file, "w") as f:
            toml.dump(d, f)
        cls.conf = config.Config(cls.config_file)

        if cls.need_sim_data:
            with mock.patch("sim._models", new=sim__models()):
                genomatnn.main(f"sim -n 10 --seed 1 {cls.config_file}".split())
                if cls.need_trained_model:
                    genomatnn.main(f"train --seed 1 {cls.config_file}".split())
                    models = list(cls.conf.dir.glob("*.hdf5"))
                    assert len(models) == 1
                    cls.nn_hdf5_file = str(models[0])

    @classmethod
    def tearDownClass(cls):
        del cls.temp_dir
        cls.config_file = None


def sim__models(
    modelspec="HomSap/HomininComposite_4G20/Neutral/msprime", models=sim._models()
):
    """
    Return a function to replace sim._models(). The replacement always returns a
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
    need_sim_data = True

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
    need_sim_data = True
    need_trained_model = True

    def test_missing_config_file(self):
        with self.assertRaises(FileNotFoundError):
            genomatnn.main(f"eval nonexistent.toml {self.nn_hdf5_file}".split())

    def test_missing_nn_hdf5_file(self):
        with self.assertRaises((FileNotFoundError, IOError)):
            genomatnn.main(f"eval {self.config_file} nonexistent.hdf5".split())

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


class TestApply(ConfigMixin, unittest.TestCase):
    need_sim_data = True
    need_trained_model = True

    def test_missing_config_file(self):
        with self.assertRaises(FileNotFoundError):
            genomatnn.main(f"apply nonexistent.toml {self.nn_hdf5_file}".split())

    def test_missing_nn_hdf5_file(self):
        with self.assertRaises((FileNotFoundError, IOError)):
            genomatnn.main(f"apply {self.config_file} nonexistent.hdf5".split())

    def test_apply_creates_predictions(self):
        genomatnn.main(f"apply {self.config_file} {self.nn_hdf5_file}".split())
        plot_dir = pathlib.Path(self.nn_hdf5_file[: -len(".hdf5")])
        for pred_file in [
            "predictions.txt",
            "predictions.pdf",
        ]:
            self.assertTrue((plot_dir / pred_file).exists())


class TestVcfplot(ConfigMixin, unittest.TestCase):
    need_sim_data = True
    need_trained_model = True
    region = "22:21000000-21100000"

    def test_missing_config_file(self):
        plot_file = pathlib.Path(self.temp_dir.name) / "plot.pdf"
        with self.assertRaises(FileNotFoundError):
            genomatnn.main(
                f"vcfplot nonexistent.toml {plot_file} {self.region}".split()
            )

    def test_vcfplot_creates_plot(self):
        plot_file = pathlib.Path(self.temp_dir.name) / "plot.pdf"
        genomatnn.main(f"vcfplot {self.config_file} {plot_file} {self.region}".split())
        self.assertTrue(plot_file.exists())
