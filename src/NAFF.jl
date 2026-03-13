module NAFF
using FFTW

function naff(
  data::AbstractMatrix; # NAFF on each ROW
  window=1, # Hanning window order
  tol=1e-4, # Acceptance tolerance for harmonics     
  nterms=1, # Number of harmonics to compute
)
  turns = size(data, 2) - 1 
  if turns < 6
    error("Minimum number of turns is 6")
  end

  # turns needs to be multiple of 6 for frequency fitting algorithm, round down
  turns = 6 * div(turns, 6)



end

end
