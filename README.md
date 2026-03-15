# NAFF

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://mattsignorelli.github.io/NAFF.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://mattsignorelli.github.io/NAFF.jl/dev/)
[![Build Status](https://github.com/mattsignorelli/NAFF.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/mattsignorelli/NAFF.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/mattsignorelli/NAFF.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/mattsignorelli/NAFF.jl)

This package is currently in development. It will provide a Julia implementation of [the Numerical Analysis of Fundamental Frequencies (NAFF) method by J. Laskar](https://www.sciencedirect.com/science/article/pii/016727899290028L), and most significantly an option for CUDA GPU-accelerated batched NAFF via the CUFFT library.