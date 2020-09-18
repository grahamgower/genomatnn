import unittest

import numpy as np

from genomatnn import vcf


class TestVCF(unittest.TestCase):
    vcf_file = "examples/1000g.Nea.22.1mb.vcf.gz"
    vcf_chrom = "22"
    vcf_start = 21000001
    vcf_end = 22000000

    def test_bcftools_query(self):
        prev_pos = 0
        for line in vcf.bcftools_query("%CHROM\t%POS\n", self.vcf_file):
            fields = line.split()
            self.assertEqual(len(fields), 2)
            chrom = fields[0]
            pos = int(fields[1])
            self.assertEqual(chrom, self.vcf_chrom)
            self.assertGreaterEqual(pos, self.vcf_start)
            self.assertLessEqual(pos, self.vcf_end)
            self.assertGreater(pos, prev_pos)
            prev_pos = pos

    def test_genotypes(self):
        maf_thres = 0.1
        max_missing_thres = 0.1
        n_sites = 0
        prev_pos = 0
        for chrom, pos, gt in vcf.genotypes(
            self.vcf_file, maf_thres=maf_thres, max_missing_thres=max_missing_thres,
        ):
            n_sites += 1
            self.assertEqual(chrom, self.vcf_chrom)
            self.assertGreaterEqual(pos, self.vcf_start)
            self.assertLessEqual(pos, self.vcf_end)
            self.assertGreater(pos, prev_pos)
            prev_pos = pos

            ac_major = (gt == 0).sum()
            ac_minor = (gt == 1).sum()
            missing = (gt == 2).sum()
            # check major/minor polarisation
            self.assertLessEqual(ac_minor, ac_major)
            # check maf_thres
            self.assertGreater(len(gt) * ac_major, maf_thres)
            self.assertGreater(len(gt) * ac_minor, maf_thres)
            # check max_missing_thres
            self.assertLessEqual(missing, len(gt) * max_missing_thres)

        # make sure we actually tested something
        self.assertGreater(n_sites, 0)

    def test_accumulate_matrices(self):
        maf_thres = 0.01
        max_missing_thres = 0.1
        winsize = 100 * 1000
        winstep = 20 * 1000
        min_seg_sites = 10
        prev_start = 1
        prev_end = winsize
        acc_pos_lists = dict()
        for chrom, start, end, M, pos_list in vcf.accumulate_matrices(
            self.vcf_file,
            winsize=winsize,
            winstep=winstep,
            min_seg_sites=min_seg_sites,
            maf_thres=maf_thres,
            max_missing_thres=max_missing_thres,
        ):
            # check the intervals make sense
            self.assertEqual(chrom, self.vcf_chrom)
            self.assertEqual(start % winstep, 1)
            self.assertEqual(end % winstep, 0)
            self.assertGreater(start, prev_start)
            self.assertGreater(end, prev_end)
            self.assertEqual((start - prev_start) % winstep, 0)
            self.assertEqual(start - prev_start, end - prev_end)
            prev_start = start
            prev_end = end
            self.assertEqual(M.shape[0], len(pos_list))
            acc_pos_lists[(start, end)] = pos_list.copy()

        # get a list of all the (filtered) sites in the vcf
        sites_list = []
        for _, pos, _ in vcf.genotypes(
            self.vcf_file,
            regions=f"{self.vcf_chrom}:{self.vcf_start}-{self.vcf_end}",
            maf_thres=maf_thres,
            max_missing_thres=max_missing_thres,
        ):
            sites_list.append(pos)

        # check the position list for each interval
        starts = np.arange(self.vcf_start, self.vcf_end - winsize + winstep, winstep)
        ends = starts + winsize - 1
        retained_sites = 0
        for start, end in zip(starts, ends):
            sites = [x for x in sites_list if start <= x <= end]
            if len(sites) < min_seg_sites:
                self.assertNotIn(
                    (start, end), acc_pos_lists.keys(), msg=f"\nsites={sites}\n"
                )
            else:
                self.assertIn((start, end), acc_pos_lists.keys())
                acc_sites = acc_pos_lists[(start, end)]
                self.assertEqual(sites, acc_sites, msg=f"start={start}, end={end}")
                retained_sites += 1
        self.assertEqual(retained_sites, len(acc_pos_lists))
