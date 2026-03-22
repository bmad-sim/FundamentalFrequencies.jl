# NAFF

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://mattsignorelli.github.io/NAFF.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://mattsignorelli.github.io/NAFF.jl/dev/)
[![Build Status](https://github.com/mattsignorelli/NAFF.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/mattsignorelli/NAFF.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/mattsignorelli/NAFF.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/mattsignorelli/NAFF.jl)

This package provides a Julia implementation of the [Numerical Analysis of Fundamental Frequencies (NAFF) method by J. Laskar](https://www.sciencedirect.com/science/article/pii/016727899290028L) that is branchless and vectorized for GPU-accelerated, batched NAFF. With `NAFF.jl`, one can do NAFF on e.g. a million particles in parallel on a GPU.

## Usage

`NAFF.jl` exports a single function `naff`:
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

For GPU usage, simply let `data` be a GPU array.

## Other NAFF Codes/Acknowledgements

[`FortNAFF`](https://github.com/MichaelEhrlichman/FortNAFF): a Fortran 90 implementation of NAFF

[`NAFF_UV`](https://github.com/kskoufar/NAFF_UV): a Fortran 2008 implementation of NAFF with additional features and window choices

[`PyNAFF`](https://github.com/nkarast/PyNAFF): a Python implementation of NAFF

In particular, I acknowledge the authors of `FortNAFF` and `NAFF_UV`, Michael Ehrlichman and Kyriacos Skoufaris respectively, for providing easiest-to-understand and modern NAFF implementations.

## AI Usage Disclosure

Two parts of this repository were created with the assistance of AI: 

1) Batched Brent optimizer
2) Test suite