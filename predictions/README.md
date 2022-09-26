This folder contains predictions from the pre-trained neural networks,
as presented in Gower et al. 2021, https://doi.org/10.7554/eLife.64669.
The files are in tab-separated-value format where coordinates are 1-based
and inclusive.

For each of the two scenarios (Denisovan introgression into Melanesians,
and Neanderthal introgression into Europeans), there are two trained networks.
One trained on AI simulations with selected mutations of allele frequency > 5%,
and one for a frequency threshold of 25%.
Furthermore, for each trained network, the predictions were each calibrated in
three different ways to correspond to Beta calibrators with Neutral:Sweep:AI
class ratios of 1:1:1, 1:0.1:0.1, or 1:0.1:0.02.

File | Scenario | AF threshold | class ratios
--- | --- | --- | ---
`Den_to_Papuan_af-0.05_w-0.02.tsv` | Denisovan introgression | 5% | 1:0.1:0.02
`Den_to_Papuan_af-0.05_w-0.1.tsv` | Denisovan introgression | 5% | 1:0.1:0.1
`Den_to_Papuan_af-0.05_w-1.tsv` | Denisovan introgression | 5% | 1:1:1
`Den_to_Papuan_af-0.25_w-0.02.tsv` | Denisovan introgression | 25% | 1:0.1:0.02
`Den_to_Papuan_af-0.25_w-0.1.tsv` | Denisovan introgression | 25% | 1:0.1:0.1
`Den_to_Papuan_af-0.25_w-1.tsv` | Denisovan introgression | 25% | 1:1:1
`Nea_to_CEU_af-0.05_w-0.02.tsv` | Neanderthal introgression | 5% | 1:0.1:0.02
`Nea_to_CEU_af-0.05_w-0.1.tsv` | Neanderthal introgression | 5% | 1:0.1:0.1
`Nea_to_CEU_af-0.05_w-1.tsv` | Neanderthal introgression | 5% | 1:1:1
`Nea_to_CEU_af-0.25_w-0.02.tsv` | Neanderthal introgression | 25% | 1:0.1:0.02
`Nea_to_CEU_af-0.25_w-0.1.tsv` | Neanderthal introgression | 25% | 1:0.1:0.1
`Nea_to_CEU_af-0.25_w-1.tsv` | Neanderthal introgression | 25% | 1:1:1
