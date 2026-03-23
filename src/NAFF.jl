module QuasiperiodicFrequencies
using FFTW, LinearAlgebra
export naff

"""
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
"""
function naff(data::AbstractMatrix, n_frequencies=1; window_order=1)
  n_particles = size(data, 1)
  turns = size(data, 2) - 1 
  numtype = real(eltype(data))

  # Construct Hanning window chi
  chi = similar(data, numtype, turns+1)
  t = @. numtype(2^window_order * factorial(window_order)^2 / factorial(2*window_order) * (1 + cos( pi * (-turns:2:turns) / turns))^window_order)
  copyto!(chi, t)

  # Construct array to store current signal residual
  signal_res = complex(copy(data))

  # FFT frequency resolution
  f_resolution = 1/turns 
  
  # U array stores orthogonal basis vector amplitudes,
  # frequencies and amplitudes stores the NAFF founded frequencies and amplitudes
  U = similar(data, complex(eltype(data)), (n_particles, turns+1, n_frequencies))
  frequencies = similar(data, real(eltype(data)), (n_particles, n_frequencies))
  amplitudes = similar(data, complex(eltype(data)), (n_particles, n_frequencies))

  fill!(U, 0)
  fill!(frequencies, 0)
  fill!(amplitudes, 0)

  # Utility function to compute inner product
  inner_prod(f,g) = sum(@.(f * chi' * conj(g)), dims=2) ./ turns

  # Now we can start the NAFF loop
  for i in 1:n_frequencies
    # 1) Apply window to signal residual, which are ROWS
    # the last point is dropped
    windowed_signal_res = view(signal_res, :, 1:turns) .* view(chi, 1:turns)'

    # 2) Use the FFT as a really good initial guess
    y = fft(windowed_signal_res, 2) # 2 = along rows
    y_norm = @. norm(y) / turns

    # 3) locate FFT bin with largest amplitude, this is 
    # coarse frequency estimate
    __, idx_max = findmax(y_norm, dims=2)
    idx_max = map(x->x[2], idx_max)

    # check if peak is at DC (k=1)
    if any(idx_max .== 1)
      dc_signals = findall(idx_max .== 1)
      @warn "Peak at DC detected for signals $(dc_signals): did you remove the mean?"
    end

    # 4) convert FFT bin index into signed frequency
    # Bins 0 … turns/2 correspond to non-negative frequencies.
    # Bins turns/2+1 … turns-1 correspond to negative frequencies
    # (aliased into the upper half of the FFT output).
    idx_shifted = @. ifelse(idx_max - 1 <= div(turns, 2), idx_max - 1, idx_max - 1 - turns)
    guess = @. idx_shifted*f_resolution

    # 5) Now we need to vary our guess until the inner product 
    # is maximized. This is done using Brent's method
    cur_frequency, __, __ = brentb(guess .- f_resolution/2 , guess .+ f_resolution/2) do x
      return abs.(inner_prod(signal_res, @.(exp(2 * pi * im * (0:turns)' * x))))
    end

    @. frequencies[:,i] = cur_frequency

    # Do MGS:
    @. U[:,:,i] = exp(2 * pi * im * (0:turns)' * cur_frequency)
    for j in 1:i-1
      U[:,:,i] .-= inner_prod(view(U, :, :, j), view(U, :, :, i)) .* view(U, :, :, j) 
    end
    amplitudes[:,i] .= inner_prod(signal_res, view(U, :, :, i))
    signal_res .-= view(amplitudes, :, i) .* view(U, :, :, i)
  end
  return frequencies, amplitudes
end

include("brentb.jl")

end
