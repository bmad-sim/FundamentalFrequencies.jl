# FundamentalFrequencies

[![Build Status](https://github.com/bmad-sim/FundamentalFrequencies.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/bmad-sim/FundamentalFrequencies.jl/actions/workflows/CI.yml?query=branch%3Amain)

This package provides a Julia implementation of the [Numerical Analysis of Fundamental Frequencies (NAFF) method by J. Laskar](https://www.sciencedirect.com/science/article/pii/016727899290028L) to compute fundamental frequencies of quasi-periodic systems to high precision. The evolution of the fundamental frequencies with time can be used to measure chaos. Some papers of interest:

- [The chaotic motion of the solar system: A numerical estimate of the size of the chaotic zones](https://www.sciencedirect.com/science/article/pii/001910359090084M)
- [Application of Frequency Map Analysis to the ALS](https://cds.cern.ch/record/301630/files/p183.pdf)

`FundamentalFrequencies.jl` in particular is implemented in a branchless and vectorized way for GPU-accelerated, batched NAFF. With this package, one can do NAFF on e.g. a million particles in parallel on a GPU. While this was initially developed for the [SciBmad accelerator physics code](https://github.com/bmad-sim/SciBmad.jl), the tools developed here may be useful to a much broader audience.

## Usage

`FundamentalFrequencies.jl` exports a single function `naff`:
```
    naff(data::AbstractMatrix, n_frequencies=1; window_order=1) -> (frequencies, amplitudes)

Performs a batched Numerical Analysis of Fundamental Frequencies (NAFF) method for 
the (complex) signals in each row of `data`, computing the first `n_frequencies`
frequencies from the signal. A Hanning window with order `window_order` is applied 
to the signal in order to gain a more accurate computation of the frequencies.

# Arguments
- `data::AbstractMatrix`: a matrix of size `num_signals x n_samples` to do NAFF on
- `n_frequencies`: number of frequencies to compute with NAFF
- `window_order`: order of the Hanning window applied to the signal throughout NAFF

# Output
- `frequencies`: a matrix of size `num_signals x n_frequencies` containing the frequencies
- `amplitudes`: a matrix of size `num_signals x n_frequencies` containing the complex 
                amplitudes associated with each frequency
```

For GPU usage, simply let `data` be a GPU array (e.g. `CuArray`). Note that the GPU backend must have implemented `AbstractFFTs.fft` in order to work on the GPU. At the time of this commit, `FundamentalFrequencies.jl` is confirmed to work on NVIDIA GPUs with CUDA. See the `cuda-test.jl` file in the `test` directory.

## Other NAFF Codes/Acknowledgements

[`FortNAFF`](https://github.com/MichaelEhrlichman/FortNAFF): a Fortran 90 implementation of NAFF

[`NAFF_UV`](https://github.com/kskoufar/NAFF_UV): a Fortran 2008 implementation of NAFF with additional features and window choices

[`PyNAFF`](https://github.com/nkarast/PyNAFF): a Python implementation of NAFF

In particular, I acknowledge the authors of `FortNAFF` and `NAFF_UV`, Michael Ehrlichman and Kyriacos Skoufaris respectively, for providing easiest-to-understand and modern NAFF implementations.

## AI Usage Disclosure

Two parts of this repository were created with the assistance of AI: 

1) Batched Brent optimizer
2) Test suite