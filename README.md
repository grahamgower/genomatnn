# About
Genomatnn is a program for detecting archaic adaptive introgression from
genotype matrices, using a convolutional neutral network (CNN). The CNN is
trained with the [tensorflow](http://www.tensorflow.org) deep learning
framework, on simulations produced by the [SLiM](https://messerlab.org/slim/)
engine in [stdpopsim](https://stdpopsim.readthedocs.io/en/latest/introduction.html).
The trained CNN can then be used to predict genomic windows of adaptive
introgression from empirical datasets in the `vcf` or `bcf` formats.

# Installation
The most trouble-free way to use genomatnn is from within a virtual environment,
and the instructions below use [conda](https://docs.conda.io/en/latest/miniconda.html)
to create such an environment. We assume you are using `Linux`. Other platforms
may work, but these have not been tested.

1. Install SLiM. Refer to the instructions on the
   [SLiM website](https://messerlab.org/slim/), or in the manual, to install
   the command line `slim` program. The SLiM GUI is not required.
   After successful installation, running `slim -v` should produce output
   similar to the following:
   ```
   SLiM version 3.4, built May 20 2020 16:00:54
   ```

2. Create a new conda environment. If you intend to do CPU-only training,
   you can remove `cudnn` from the list. We pin the numpy version here because
   recent versions are incompatible with tensorflow. We choose the `mkl` variant
   of blas, which uses the Intel math kernel and markedly improves the speed of
   matrix operations on Intel CPUs.
   ```
   conda create -n genomatnn gsl cudnn "numpy<1.19" "blas=*=mkl"
   conda activate genomatnn
   ```

3. Install our `selection` stdpopsim branch. This branch will only be installed
   in the active conda environment, and will not clash with any existing
   stdpopsim installation. We are working to get this merged into stdpopsim
   so that this step will no longer be necessary in the future.
   ```
   pip install git+https://github.com/grahamgower/stdpopsim.git@selection
   ```

4. With the conda environment still activated, clone the genomatnn git
   repository, then build and install.
   ```
   git clone https://github.com/grahamgower/genomatnn.git
   cd genomatnn
   python setup.py install
   ```

5. Run the tests to check that installation was successful.
   The tests can take a few minutes to run.
   ```
   pip install nose
   python setup.py build_ext -i
   nosetests -v tests
   ```

We recognise that this installation process is not trivial. Please open a
github issue if you have any trouble, or the tests fail. Be sure to include
as much detail as possible.


# Usage
The `genomatnn` command will provide a concise usage summary if it is invoked
without parameters. Alternately, the `-h` option will print more detailed
usage information. In the text below, the `$ ` indicates the command prompt.
```
$ genomatnn
usage: genomatnn [-h] {sim,train,eval,apply,vcfplot} ...
genomatnn: error: the following arguments are required: subcommand
```
```
$ genomatnn -h
usage: genomatnn [-h] {sim,train,eval,apply,vcfplot} ...

Simulate, train, and apply a CNN to genotype matrices.

positional arguments:
  {sim,train,eval,apply,vcfplot}
    sim                 Simulate tree sequences.
    train               Train a CNN.
    eval                Evaluate trained CNN.
    apply               Apply trained CNN.
    vcfplot             Plot haplotype/genotype matrices from a VCF/BCF.

optional arguments:
  -h, --help            show this help message and exit
```

Here we see that genomatnn has several subcommands. `sim` to run simulations,
`train` to train a CNN, `eval` to produce evaluation plots for a trained CNN,
and `apply` to predict AI on empirical data by applying a trained CNN.

Each of the subcommands may also be invoked without parameters to produce
a concise usage summary, or invoked with the `-h` option to get more detailed
usage information.
```
$ genomatnn sim
usage: genomatnn sim [-h] [-j PARALLELISM] [-s SEED] [-v] [-n NUM_REPS] [-l]
                     conf.toml [modelspec]
genomatnn sim: error: the following arguments are required: conf.toml
```
```
$ genomatnn train 
usage: genomatnn train [-h] [-j PARALLELISM] [-s SEED] [-v] [-c] conf.toml
genomatnn train: error: the following arguments are required: conf.toml
```
```
$ genomatnn apply
usage: genomatnn apply [-h] [-j PARALLELISM] [-s SEED] [-v] [-p] conf.toml nn.hdf5
genomatnn apply: error: the following arguments are required: conf.toml, nn.hdf5
```
```
$ genomatnn eval
usage: genomatnn eval [-h] [-j PARALLELISM] [-s SEED] [-v] [--no-reliability]
                      conf.toml nn.hdf5
genomatnn eval: error: the following arguments are required: conf.toml, nn.hdf5
```
```
$ genomatnn sim -h
usage: genomatnn sim [-h] [-j PARALLELISM] [-s SEED] [-v] [-n NUM_REPS] [-l]
                     conf.toml [modelspec]

positional arguments:
  conf.toml             Configuration file.
  modelspec             Model specification to simulate. If not provided, modelspecs from
                        the config file will be simulated

optional arguments:
  -h, --help            show this help message and exit
  -j PARALLELISM, --parallelism PARALLELISM
                        Number of processes or threads to use for parallel things. E.g.
                        simultaneous simulations, or the number of threads used by
                        tensorflow when running on CPU. If set to zero, os.cpu_count() is
                        used. [default=0].
  -s SEED, --seed SEED  Seed for the random number generator [default=1836563304].
  -v, --verbose         Increase verbosity. Specify twice for messages from third party
                        libraries (e.g. tensorflow and matplotlib).
  -n NUM_REPS, --num-reps NUM_REPS
                        Number of replicate simulations. For each replicate, one simulation
                        is run for each modelspec. [default=1]
  -l, --list            List available model specifications.
```

There are some options which are common to all subcommands, such as the
`-j`, `-s`, and `-v` options.


## Configuration
From the usage information above, we can see that `genomatnn` subcommands
require a configuration file. A genomatnn analysis procedes through
multiple steps, and each step needs to be configured consistently with the
previous step(s). The most extreme example of this, is that the number of
individuals that are simulated in the first `sim` step must match the
empirical data used in the final `apply` step. While it may at first appear
counterintuitive, this means that to do any simulations, we must first
describe which individuals will be used from the vcf(s), and how these
relate to the populations that will be simulated.
The configuration file uses the [toml](https://github.com/toml-lang/toml)
format, and we provide an extensively commented [example configuration](examples/Nea_to_CEU.toml).


## Worked example
Here we provide a worked example for detecting adaptive introgression in humans,
where Neanderthals are the donor population and Europeans are the recipient.
Create a new working directory. We will use this to store the configuration
file, the example vcf, and all output files.
```
$ mkdir ai_example
$ cd ai_example
```

Copy the `Nea_to_CEU.toml`, `1000g.Nea.22.1mb.vcf.gz`,
and `1000g.Nea.22.1mb.vcf.gz.csi` files from the `examples` folder of
your cloned genomatnn repository. We will also need the `YRI.indlist` and
`CEU.indlist` files, which list the vcf's individual IDs for the YRI and
CEU populations.
```
$ export GENOMATNN=/path/to/the/cloned/genomatnn
$ cp $GENOMATNN/examples/Nea_to_CEU.toml .
$ cp $GENOMATNN/examples/1000g.Nea.22.1mb.vcf.gz{,.csi} .
$ cp $GENOMATNN/examples/{YRI,CEU}.indlist .
```

### VCF and populations
We will lightly modify the example configuration file. First, change the
`dir = ...` line at the top to be the current directory. I.e.
```
dir = "."
```

Next, find the `[vcf]` section, and modify the `chr = ...` and `file = ...`
parameters to be:
```
chr = [22]
file = "1000g.Nea.${chr}.1mb.vcf.gz"
```
This vcf file contains genotype calls for a 1 Mbp region on chr22, for the
YRI and CEU populations from the 1000 genomes project, plus the Altai and
Vindija Neanderthals.

Scrolling down the config file, we see the `[pop]` section, which describes
which individuals in the vcf file correspond to which population labels.
The possible population labels here are defined by the demographic model
that will be simulated.
```
[pop]
# For each population, specify the individual IDs in the VCF. This can either
# be a list of IDs, or the name of a file containing the IDs (one per line).
# The population names must match those used for the simulations.
# The order of populations given here will be used for the ordering in the
# genotype matrices. It's recommended for the donor and recipient populations
# to be adjacent in the genotype matrices!
Nea = ["AltaiNeandertal", "Vindija33.19"]
CEU = "CEU.indlist"
YRI = "YRI.indlist"
```

### Simulating
We are now ready to start simulating. We shall first list the model
specifications that are currently defined by genomatnn.
```
$ genomatnn sim -l Nea_to_CEU.toml 
HomSap/PapuansOutOfAfrica_10J19/Neutral/slim
HomSap/PapuansOutOfAfrica_10J19/Neutral/msprime
HomSap/PapuansOutOfAfrica_10J19/DFE
HomSap/PapuansOutOfAfrica_10J19/AI/Den1_to_Papuan
HomSap/PapuansOutOfAfrica_10J19/AI/Den2_to_Papuan
HomSap/PapuansOutOfAfrica_10J19/Sweep/Papuan
HomSap/HomininComposite_4G20/Neutral/slim
HomSap/HomininComposite_4G20/Neutral/msprime
HomSap/HomininComposite_4G20/DFE
HomSap/HomininComposite_4G20/AI/Nea_to_CEU
HomSap/HomininComposite_4G20/Sweep/CEU
```
This indicates that there are two demographic models available for the
`HomSap` species: `PapuansOutOfAfrica_10J19` and `HomininComposite_4G20`.
For each of these demographic models, there are multiple lines, which
genomatnn refers to as `modelspec`s. These describe the different ways we can
simulate a given demographic model. We will use the following modelspecs:
 * `HomSap/HomininComposite_4G20/Neutral/slim`: only neutral mutations
    are simulated, and SLiM will be used for the simulation (rather
    than msprime).
 * `HomSap/HomininComposite_4G20/AI/Nea_to_CEU`: adaptive introgression
    where a mutation is drawn in Neanderthals, passed to Europeans via
    admixture, and is then positively selected in Europeans.
 * `HomSap/HomininComposite_4G20/Sweep/CEU`: a selective sweep in Europeans.
 
In our configuration file, we see that these modelspecs are used in the
`[sim.tranche]` section.
```
[sim.tranche]
# The labels and modelspec(s) for each tranche. The network will be trained to
# classify data as coming from one of these tranches. Each tranche consists of
# a list of simulation modelspecs.
# Only two tranches are supported.
"not AI" = [
        "HomSap/HomininComposite_4G20/Neutral/slim",
        "HomSap/HomininComposite_4G20/Sweep/CEU",
]

AI = [
        "HomSap/HomininComposite_4G20/AI/Nea_to_CEU",
]
```
Above, there are two "tranches" (or groups) described. We only support 
binary classification CNNs, i.e. prediction between two possible classes,
and here we have described what the two classes are: condition negative
is "not AI"; while condition positive is "AI". This means that when the
CNN outputs a value close to 0, its prediction is "not AI", and when it
outputs a value near 1, its prediction is AI. Each tranche is just a list
of modelspecs for simulation, that will later be used to train the CNN.

So without further ado, lets do 1000 simulations for each modelspec
defined in our config file.
```
$ genomatnn sim -n 1000 Nea_to_CEU.toml
```

While 1000 simulations are not enough for accurate predictions (above 10,000
for each modelspec might be reasonable, although more is better), we can
expect that a CNN will learn from this quantity of data.
By default, the above command will use all available CPU cores on the
system. This can be changed with the `-j` parameter. It can take a while
to do a lot of simulations though (you might want to leave this running
overnight if you're running it on a workstation or laptop), so you definitely
want as many cores as you can get. If you have multiple compute servers
available, then you can run `genomatnn sim` independently on each system.

The output of each simulation is tree sequence file, located under a folder
hierarchy which is named according to the modelspec being simulated. E.g.
as we run the above `genomatnn sim` command, the
`HomSap/HomininComposite_4G20/Neutral/slim` folder will be created, and
will be populated with `*.trees` files, each numbered according to the random
seed that was used to simulate it.


### Creating genotype matrices and training a CNN
We now want to convert the tree sequences into genotype matrices, ready
for CNN training. Let's modify the configuration file once again, to
set the size of the genotype matrices. In the `[train]` section, change
`num_rows = ...` to read
```
num_rows = 32
```
This will resize the genotype matrices so that each haplotype is 
represented by 32 bins. The smaller the matrices, the less memory we'll
use, and the faster it will be to train. 32 may not seem like much, but
a CNN is suprisingly effective with even very low resolution input.

Given that we only have 1000 simulations per modelspec, we'll also increase
the number of training epochs.
```
epochs = 15
```

And in the `[train.cnn]` section we'll decrease the depth of the network
to 3 convolutional layers. Additional layers are really only useful for
larger genotype matrices.
```
n_conv = 3
```

To convert the tree sequences into genotype matrices, run
```
$ genomatnn train -c Nea_to_CEU.toml
```
This command will search for `*.trees` files in the folder hierarchy
corresponding to the modelspecs listed in the config file. It will
split your data into training/validation sets (with approximate 90/10 split),
convert each file into a genotype matrix, resize, sort the resized "haplotypes",
and store the result into a Zarr cache. This process is done in parallel
by default, which can be changed with the `-j` command. If you have a slow
disk, then it might be quicker to run this with only 1 process. For fast disks,
parallelism really helps. Your milage may vary. Once complete, you should
see a `zarrcache_32-rows` folder in the working directory. If you decide to
add more simulations at a later date, just delete this folder and regenerate
the cache. You can also change the `num_rows` parameter and generate an
additional cache folder with differently-sized genotype matrices.

Now let's train a CNN!
```
$ genomatnn train Nea_to_CEU.toml
```

This will produce a file named `cnn_*.hdf5`, with the random seed that was
used in the name, e.g. `cnn_3029321311.hdf5`. By default, the `train`
subcommand will let tensorflow decide whether to do training on the GPU(s)
(if available), or CPUs. Tensorflow is greedy though, and will use all
available GPUs. Tensorflow can be coaxed into using a specific GPU by setting
the `CUDA_VISIBLE_DEVICES` enviroment variable to that GPU. If no GPUs are
found, tensorflow will use all available CPUs. If you'd like to use fewer CPUs
for training, use the `-j` parameter when calling `genomatnn train`.


### Evaluating the CNN.
In the previous step, we trained a CNN on just a handful of simulations.
Probably when you trained this, the training accuracy quickly saturated at 1.0,
indicating severe overfit. This is best overcome by feeding the CNN with more
simulations. Once you're happy with the loss/accuracy metrics reported
by tensorflow, its a good idea to check the confusion matrix, ROC curve, etc.
on your validation dataset. Genomatnn automatically plots these after the model
has been trained, and they can be found in a folder name for the trained model
(e.g. if the model is `cnn_3029321311.hdf5`, the folder will be `cnn_3029321311`).


### Predicting adaptive introgression on empirical data
Apply the trained CNN to the vcf(s) listed in the config file with the `apply`
subcommand.
```
$ genomatnn apply Nea_to_CEU.toml cnn_3029321311.hdf5
```
By default, this will use all CPU cores on your system for conversion of vcf
data into genotype matrices. If your disk is fast, this is very quick. If
your disk is slow though, you probably want to use fewer CPU cores with the
`-j` parameter.

This will output two files: `cnn_3029321311/predictions.txt` and
`cnn_3029321311/predictions.pdf`. The former contains tab-separated regions
and a probability for each region. The latter is a Manhattan plot of the
results (this won't make sense for the example vcf file, sorry).


# Troubleshooting
 * Try adding the `-v` parameter to any genomatnn subcommand you're having
   trouble with.
 * Open a github issue, and please include as much detail as possible.
