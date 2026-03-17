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

#=
    tol : float, default 1e-4
        Relative tolerance used in `fretes` (the duplicate-frequency guard).
        A newly found frequency is considered identical to a previously found
        one if |ν_new − ν_old| / ν₀ < tol, where ν₀ = 1/turns.  The default
        of 1e-4 matches Laskar's original Fortran code.
        =#

# Return value will be N_particle x nterms arrays of frequencies and complex amplitudes
function naff(
  data::AbstractMatrix; # NAFF on each ROW
  window=1, # Hanning window order
  tol=1e-4, # Acceptance tolerance for harmonics     
  nterms=1, # Number of harmonics to compute
  warnings=true,
)
  nparticles = size(data, 1)
  turns = size(data, 2) - 1 
  if turns < 6
    error("Minimum number of turns is 6")
  end

  # use a multiple of 6 for turns so that the Newton-Cotes
  # integral can be easily taken across chunks of size 6
  turns = 6 * div(turns, 6)

  # Construct the Hanning window to smooth the signal
  # First we will make the time-like (turns) variable 
  # symmetric via Eq. 2 in the paper, and then use Eq. 9
  # Note this is done once for all signals
  T = @. 2*range(0, turns; length=turns+1) - turns
  chi = @. 2^window * factorial(window)^2 / factorial(2*window) * (1 + cos( pi*T / turns))^window

  data = view(data, :, 1:(turns+1))  # current signal residual
  signal_res = data 
  f_resolution = 1/turns # FFT frequency resolution

  # Preallocate Newton-Cotes weights array for inner-product integral
  weights = newton_cotes_weights(data)

  # Preallocate Q and R matrices for Gram-Schmidt
  # If a duplicate frequency is found, the column Q will be left zero, with 
  # corresponding row also zero. This is done for vectorization.

  # Q will contain the frequencies as complex exponentials\
  # R will contain their amplitudes
  Q = similar(data, complex(eltype(data)), (nparticles, nterms, nterms))
  R = similar(Q)

  Q .= 0
  R .= 0

  # Now we can start the NAFF loop
  for i in 1:nterms
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
    guess = @. (idx_shifted + 1)*f_resolution

    # 5) Now we need to vary our guess until the inner product 
    # is maximized. This is done using Brent's method
    inner_prod = x -> @. abs(integrate(1/turns * data * chi * exp(-2 * pi * im * (0:turns) * x), weights))
    cur_frequency, cur_phi_max, __ = brentb(inner_prod, guess .- f_resolution/3 , guess .+ f_resolution/3)

    # 6) If not first iteration, check if this frequency is a duplicate
    if i != 1
      # Find column in frequencies array where there is the closest match
      dmin, idx_dmin = findmin(x->abs.(x .- cur_frequency), view(frequencies, :, 1:(i-1)), dims=2)
      iflag = @. ifelse(dmin < f_resolution, ifelse(dmin < tol*f_resolution, -1, 0), 1)
    end


    # 7) In order to subtract the frequency from the signal, we 
    # have to find the amplitude A of A exp(2*pi*im*nu). But 
    # for finite interval the basis function exp(2*pi*im*nu_1) is not  
    # orthogonal to all other exp(2*pi*im*nu_2), so we have to project 
    # the current signal residual onto the vector space generated 
    # by all found frequencies before subtracting it out. This requires 
    # Gram-Schmidt for new frequencies. 
    # For near-duplicates, we need to add amplitude correction but 
    # not do Gram-Schmidt
    # for very close duplicates but not exactly, there is a problem 
    # basically and original NAFF code exits (tuning the window order 
    # or FFT samples could help), because it would just 
    # keep finding this same peak. Here we just ignore and keep going 
    # in order to maintain branchlessness

    # Let's write out explicitly.

    # Each step, the new found frequency in our previously-found 
    # orthogonal basis may have some projection on the other frequency
    # amplitudes, as well as its own new orthogonalized direction. Therefore
    # we have to add that vector's contribution to the amplitudes previously found
    # the directions previously found of course don't change.

    # So really its a step-by-step QR factorization.
    # V = Q*R where Q has columns as orthogonal bases and R is 
    # upper triangular specifying the amplitude of each Q making up 
    # each column of V

    # The memory layout should be SIMD-able. Each particle has its own unit vectors
    # We should store each component of each unit vector together, e.g. 
    # nparticles x nterms x nterms


    # In the original NAFF code, an upper triangular matrix is constructed 
    # that transforms the input vector harmonics V into its orthogonal basis.

    # V = Q*R
    if term == 1
      # Something definitely was found for first iteration
      @. Q[:,1,1] = exp(im * 2 * pi * cur_frequency) # Already unit norm
      @. R[:,1,1] = abs(cur_phi_max)
    else
      @. Q[:,term,term] = ifelse(iflag == 1, exp(im * 2 * pi * cur_frequency), 0)
      for j in 1:term-1
        @. R[:,j,term] = inner_prod()
        @. Q[:,:,term] -= R[:,j,term]*Q[:,:,j]
      end
      @. R[:,term,term] = norm(Q[:,:,term])
      @. Q[:,:,term] /= R[:,term,term] 
    end

    


  end
end

# Construction of this should be done once and passed along
function newton_cotes_weights(integrand)
  N = size(integrand, 2)
  @assert rem(N, 6) == 0 "newton_cotes_weights only works for arrays with length = multiple of six"
  
  # Initialize first on CPU
  weights = zeros(eltype(integrand), N)
  K = div(N, 6)

  # interior shared-endpoint indices, require double counting
  interior = 6 .* (1:K-1)

  weights[1] = 41
  weights[2] = 216
  weights[3] = 27
  weights[4] = 272
  weights[5] = 27
  weights[6] = 216
  weights[N] = 41

  weights[interior]     .+= 82
  weights[interior .+ 1] .+= 216
  weights[interior .+ 2] .+= 27 
  weights[interior .+ 3] .+= 272
  weights[interior .+ 4] .+= 27
  weights[interior .+ 5] .+= 216

  # Now copy result to device
  weights_device = similar(integrand, N)
  copyto!(weights_device, weights)

  return weights_device
end

function integrate(integrand::AbstractMatrix, weights=newton_cotes_weights(size(integrand, 2)))
  @assert size(integrand, 2) == length(weights) "Sizes of weights and integrand arrays do not match"
  return (integrand * weights) ./ 140
end

end
