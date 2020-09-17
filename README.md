Genomatnn is a program for detecting archaic adaptive introgression from
genotype matrices, using a convolutional neutral network (CNN). The CNN is
trained with the [tensorflow](http://www.tensorflow.org) deep learning
framework, on simulations produced by the [SLiM](https://messerlab.org/slim/)
engine in [stdpopsim](https://stdpopsim.readthedocs.io/en/latest/introduction.html).
The trained CNN can then be used to predict genomic windows of adaptive
introgression from empirical datasets in the `vcf` or `bcf` formats.

# Installation
The most trouble-free way to use genomatnn is from within a virtual environment,
and the instructions below use `conda` to create such an environment. We assume
you are using `Linux`. Other platforms may work, but these have not been tested.

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
   python setup.py build_ext -i
   python setup.py install
   ```

5. Run the tests to check that installation was successful.
   The tests can take a few minutes to run.
   ```
   pip install nose
   nosetests -v tests
   ```

We recognise that this installation process is not trivial. Please open a
github issue if you have any trouble, or the tests fail. Be sure to include
as much detail as possible.
