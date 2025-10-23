using ITensors, ITensorMPS
let
    # Create spin-1/2 indices
    N = 50
    sites = siteinds("S=1/2", N)
    # Angular Frequency
    w = pi
    # Time
    t = 0
    # Hopping Amplitude
    t0 = 1
    # Interaction
    g = 1
      
    # Hamiltonian at the initial state from 10/20/25
    os = OpSum()
    for j in 1:(N - 2)
        os += 4 * t0 * cos(w * t), "Sz", j, "Sz", j + 1 # 4 is to cancel out the multiples of 1/2 from Sxz
        os += 2 * t0 * cos(w * t), "Sx", j
        os += -8 * g, "Sx", j, "Sz", j + 1, "Sz", j + 2
        os += -8 * g, "Sz", j, "Sz", j + 1, "Sx", j + 2
    end
    H = MPO(os, sites)

    psi0 = random_mps(sites)
    nsweeps = 8
    maxdim = [1, 2, 3, 10, 25, 50, 100, 500]
    cutoff = 1.0e-10

    energy, psi = dmrg(H, psi0; nsweeps, maxdim, cutoff)
    println("Final energy = $energy")

    nothing
end
