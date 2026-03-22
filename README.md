# NAFF

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://mattsignorelli.github.io/NAFF.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://mattsignorelli.github.io/NAFF.jl/dev/)
[![Build Status](https://github.com/mattsignorelli/NAFF.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/mattsignorelli/NAFF.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/mattsignorelli/NAFF.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/mattsignorelli/NAFF.jl)

This package provides a Julia implementation of the [Numerical Analysis of Fundamental Frequencies (NAFF) method by J. Laskar](https://www.sciencedirect.com/science/article/pii/016727899290028L) that is branchless and vectorized for GPU-accelerated, batched NAFF. With `NAFF.jl`, one can do NAFF on e.g. a million particles in parallel.

## Other NAFF Codes/Acknowledgements

[FortNAFF](https://github.com/MichaelEhrlichman/FortNAFF): a Fortran 90 implementation of NAFF

[NAFF\_UV](https://github.com/kskoufar/NAFF_UV): a Fortran 2008 implementation of NAFF with additional features and window choices

[PyNAFF](https://github.com/nkarast/PyNAFF): a Python implementation of NAFF

In particular, I acknowledge the authors of FortNAFF and NAFF\_UV, Michael Ehrlichman and Kyriacos Skoufaris respectively, for providing easiest-to-understand and modern NAFF implementations.

## AI Usage Disclosure

Two parts of this repository were created with the assistance of AI: 1) the batched Brent optimizer, and 2) the test suite. 