# NAFF

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://mattsignorelli.github.io/NAFF.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://mattsignorelli.github.io/NAFF.jl/dev/)
[![Build Status](https://github.com/mattsignorelli/NAFF.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/mattsignorelli/NAFF.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/mattsignorelli/NAFF.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/mattsignorelli/NAFF.jl)

This package is currently in development. It will provide a Julia implementation of [the Numerical Analysis of Fundamental Frequencies (NAFF) method by J. Laskar](https://www.sciencedirect.com/science/article/pii/001910359090084M). While the implementation will follow closely to that in [PyNAFF](https://github.com/nkarast/PyNAFF), it will be extended to provide an option of CUDA GPU-accelerated batched NAFF via the CUFFT library.