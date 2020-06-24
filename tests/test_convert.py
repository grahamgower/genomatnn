import tempfile
import subprocess
import unittest
import unittest.mock as mock

import numpy as np

import vcf
import convert
import tests


class TestSorting(unittest.TestCase):
    sample_counts = tests.HashableDict(YRI=10, CHB=20, CEU=30, Papuan=40)

    def verify_sorted(self, A, c):
        assert c.shape[1] == 1
        last_dist = -1
        for k in range(A.shape[1]):
            dist = np.sum((A[:, k] - c[:, 0]) ** 2)
            self.assertLessEqual(
                last_dist, dist, msg=f"\nc={c}\nA={A}\nA[:, k={k}]={A[:, k]}"
            )
            last_dist = dist

    def test_sort_similarity(self):
        n = 10
        rng = np.random.default_rng(seed=31415)
        for coeff in (-100, 0, 100):
            A = rng.uniform(-100, 100, size=(n + 1, n))
            c = coeff * np.ones((A.shape[0], 1))
            B = convert.sort_similarity(A, c)
            self.verify_sorted(B, c)

    def test_ts_pop_counts_indices(self):
        ts, model = tests.basic_sim(self.sample_counts)
        counts, indices = convert.ts_pop_counts_indices(ts)
        self.assertEqual(len(indices), len(counts))
        self.assertEqual(indices.keys(), counts.keys())
        self.assertEqual(len(counts), len(self.sample_counts))
        pop_id = {pop.id: j for j, pop in enumerate(model.populations)}
        for pop, count in self.sample_counts.items():
            j = pop_id[pop]
            self.assertEqual(counts[j], count)
        offset = 0
        for j, index in indices.items():
            self.assertEqual(index, offset)
            offset += counts[j]

    def test_verify_partition(self):
        convert.verify_partition([0], [100], 100)
        convert.verify_partition([0, 10, 20, 30], [10, 10, 10, 10], 40)
        with self.assertRaises(ValueError):
            convert.verify_partition([0], [100], 200)
            convert.verify_partition([0], [200], 100)
            convert.verify_partition([0, 10], [20, 10], 30)
            convert.verify_partition([0, 100], [10, 10], 20)

        ts, model = tests.basic_sim(self.sample_counts)
        counts, indices = convert.ts_pop_counts_indices(ts)
        convert.verify_partition(
            indices.values(), counts.values(), sum(counts.values())
        )

    def test_reorder(self):
        rng = np.random.default_rng(seed=31415)
        maf_thres = 0.05
        num_rows = 32
        ts, model = tests.basic_sim(self.sample_counts)
        A = convert.ts2mat(ts, num_rows, maf_thres, rng)

        counts, indices = convert.ts_pop_counts_indices(ts)
        pop_id = {pop.id: j for j, pop in enumerate(model.populations)}
        ref_pop = pop_id["YRI"]
        j, k = indices[ref_pop], indices[ref_pop] + counts[ref_pop]
        c = np.empty((num_rows, 1))
        for i in range(num_rows):
            c[i, 0] = np.mean(A[i, j:k])

        new_indices = tests.reorder_indices(model, self.sample_counts)
        # Check that per-population submatrices are each sorted.
        # We check with the original population indices, then reorder
        # populations according to new_indices and check again.
        assert list(indices.keys()) != list(new_indices.keys())
        for dest_indices in (indices, new_indices):
            B = convert.reorder_and_sort(A, counts, indices, dest_indices, ref_pop)
            offset = 0
            for _id in dest_indices.keys():
                j, k = offset, offset + counts[_id]
                offset = k
                self.verify_sorted(B[:, j:k], c)


class TestGenotypeMatrixes(unittest.TestCase):
    sample_counts = tests.HashableDict(YRI=10, CHB=20, CEU=30, Papuan=40)

    def test_ts_genotype_matrix(self):
        num_haplotypes = sum(self.sample_counts.values())
        ts, _ = tests.basic_sim(self.sample_counts)
        maf_thres = 0.05
        rng = np.random.default_rng(seed=31415)
        for num_rows in (32, 64, 128):
            A = convert.ts2mat(ts, num_rows, maf_thres, rng)
            self.assertEqual(A.shape, (num_rows, num_haplotypes))
        # TODO add some actual tests!

    def test_compare_ts_vcf_genotype_matrixes(self):
        # Compare ts genotype matrix to vcf genotype matrix from ts.write_vcf().
        maf_thres = 0.05
        num_haplotypes = sum(self.sample_counts.values())
        ac_thres = maf_thres * num_haplotypes
        ts, _ = tests.basic_sim(self.sample_counts)
        positions = [
            # List of MAF filtered positions.
            v.site.position
            for v in ts.variants()
            if sum(v.genotypes) >= ac_thres
            and num_haplotypes - sum(v.genotypes) >= ac_thres
        ]
        individual_names = [f"ind{j}" for j in range(num_haplotypes // 2)]
        # Mock out random variables to ensure we get consistent behaviour
        # between ts and vcf versions.
        rng = mock.MagicMock()
        rng.random = mock.MagicMock(return_value=0.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            vcf_file = tmpdir + "/ts.vcf"
            samples_file = tmpdir + "/samples.txt"
            with open(vcf_file, "w") as f:
                ts.write_vcf(
                    f,
                    ploidy=2,
                    individual_names=individual_names,
                    position_transform=np.round,
                )
            with open(samples_file, "w") as f:
                print(*individual_names, file=f, sep="\n")
            subprocess.run(["bgzip", vcf_file])
            vcf_file += ".gz"
            subprocess.run(["bcftools", "index", vcf_file])
            chrom = "1"
            start = 1
            end = int(ts.sequence_length)
            vcf_pos, V = vcf.vcf2mat(
                vcf_file,
                samples_file,
                chrom,
                start,
                end,
                rng,
                maf_thres=maf_thres,
                unphase=False,
            )
        np.testing.assert_array_equal(vcf_pos, np.round(positions))
        for num_rows in (32, 64, 128):
            A = convert.ts2mat(ts, num_rows, maf_thres, rng)
            self.assertEqual(A.shape, (num_rows, num_haplotypes))
            # Use the float `positions` vector to resize here, not the integer
            # `vcf_pos` vector, to ensure resizing equivalence.
            B = vcf.resize(positions, V, end, num_rows)
            self.assertEqual(A.shape, B.shape)
            self.assertEqual(A.dtype, B.dtype)
            np.testing.assert_array_equal(A, B)
