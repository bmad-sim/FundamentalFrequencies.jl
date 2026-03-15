module NAFF
using FFTW, LinearAlgebra
export naff

# NAFF basically works by using the fact that 
# for a function f(t) = A exp(iνt), the Fourier inner 
# product ⟨f(t),exp(iωt)⟩ will be maximized when ω = ν .

# For CUDA, we have each ROW being a different 
# particle, column is turn number. This is because
# we can then easily each turn vectorized write to 
# the columns

function naff(
  data::AbstractMatrix; # NAFF on each ROW
  window=1, # Hanning window order
  tol=1e-4, # Acceptance tolerance for harmonics     
  nterms=1, # Number of harmonics to compute
  warnings=true,
)
  turns = size(data, 2) - 1 
  if turns < 6
    error("Minimum number of turns is 6")
  end

  # use a multiple of 6 for turns so that the Hardy rule 
  # integral can be easily taken across chunks of size 6
  turns = 6 * div(turns, 6)

  # Construct the Hanning window to smooth the signal
  # First we will make the time-like (turns) variable 
  # symmetric via Eq. 2 in the paper, and then use Eq. 9
  # Note this is done once for all signals
  T = @. 2*range(0, turns; length=turns+1) - turns
  chi = @. 2^window * factorial(window)^2 / factorial(2*window) * (1 + cos( pi*T / turns))^window

  signal_res = view(data, :, 1:(turns+1))  # current signal residual
  f_resolution = 1/turns # FFT frequency resolution
  threshold = f_resolution/10^8 # convergence threshold

  # Now we can start the NAFF loop
  for term in 1:nterms
    # 1) Apply window to signal residual, which are ROWS
    # the last point is dropped
    data_for_fft = view(signal_res, 1:turns) .* chi'

    # 2) Use the FFT as a really good initial guess
    y = fft(data_for_fft, 2) # 2 = along rows
    y_norm = @. norm(y) / turns

    # 3) locate FFT bin with largest amplitude, this is 
    # coarse frequency estimate
    idx_max, y_max = findmax(y_norm, dims=2)

    # check if peak is at DC (k=1)
    if any(idx_max .== 1) && warnings
      dc_signals = findall(idx_max .== 1)
      @warn "Peak at DC detected for signals $(dc_signals): did you remove the mean?"
    end

    # 4) convert FFT bin index into signed frequency
    # Bins 0 … turns/2 correspond to non-negative frequencies.
    # Bins turns/2+1 … turns-1 correspond to negative frequencies
    # (aliased into the upper half of the FFT output).
    idx_shifted = @. ifelse(idx_max <= turns/2, idx_max - 1, idx_max - 1 - turns)
    guess = (idx_shifted + 1)*f_resolution

    # 5) Now we need to vary our guess until the inner product 
    # is maximized. This is done using quadratic interpolation
    

  end
end

# Construction of this should be done once and passed along
function newton_cotes_weights(integrand)
  N = size(integrand, 2)
  @assert rem(N, 6) == 0 "newton_cotes_weights only works for arrays with length = multiple of six"
  
  # Initialize first, always on CPU
  weights = Vector{eltype(integrand)}(undef, N) 
  K = div(N, 6)
  
  # interior shared-endpoint indices, require double counting
  interior = 6 .* (1:K-1)

  weights[1]   = 41.
  weights[2]   = 216.
  weights[3]   = 27.
  weights[4]   = 272.
  weights[5]   = 27.
  weights[6]   = 216.
  weights[N]  += 41.

  weights[interior]     .+= 82. 
  weights[interior .+ 1] .+= 216.
  weights[interior .+ 2] .+= 27. 
  weights[interior .+ 3] .+= 272.
  weights[interior .+ 4] .+= 27.
  weights[interior .+ 5] .+= 216.

  # Now copy result to device
  weights_device = similar(integrand, N)
  copyto!(weights_device, weights)

  return weights_device
end

function integrate(integrand::AbstractMatrix, weights=newton_cotes_weights(size(integrand, 2)))
  @assert size(integrand, 2) == length(weights) "Sizes of weights and integrand arrays do not match"
  return (integrand * weights) ./ 140
end

function f_refine()


end


end
