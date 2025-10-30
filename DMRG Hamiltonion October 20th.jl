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
      
    # Hamiltonian at the initial state from 10/20/25, where "X" and "Z" are Pauli Matrices
    os = OpSum()
    for j in 1:(N - 2)
        os += t0 * cos(w * t), "Z", j, "Z", j + 1
        os += t0 * cos(w * t), "X", j
        os -= g, "X", j, "Z", j + 1, "Z", j + 2
        os -= g, "Z", j, "Z", j + 1, "X", j + 2
    end
    
    # Boundary conditions
    os += t0 * cos(w * t), "Z", N - 1, "Z", N # j = N - 1
    os += t0 * cos(w * t), "X", N - 1
    os -= g, "X", N - 1, "Z", N, "Z", 1
    os -= g, "Z", N - 1, "Z", N, "X", 1
    os += t0 * cos(w * t), "Z", N, "Z", 1 # j = N
    os += t0 * cos(w * t), "X", N
    os -= g, "X", N, "Z", 1, "Z", 2
    os -= g, "Z", N, "Z", 1, "X", 2
        
    H = MPO(os, sites)
    
    psi0 = random_mps(sites;linkdims=10)
    nsweeps = 8
    maxdim = [5, 10, 50, 100, 500, 1000]
    cutoff = 1.0e-10

    energy1, psi1 = dmrg(H, psi0; nsweeps, maxdim, cutoff)
    energy2, psi2 = dmrg(H, [psi1], psi0; nsweeps, maxdim, cutoff, weight = 100)
    energy3, psi3 = dmrg(H, [psi1, psi2], psi0; nsweeps, maxdim, cutoff, weight = 100)
    energy4, psi4 = dmrg(H, [psi1, psi2, psi3], psi0; nsweeps, maxdim, cutoff, weight = 100)
    
    println("Final energy 1 = $energy1")
    println("Final energy 2 = $energy2")
    println("Final energy 3 = $energy3")
    println("Final energy 4 = $energy4")
    println("<psi1|psi1> = $(inner(psi1, psi1))")
    println("<psi2|psi2> = $(inner(psi2, psi2))")
    println("<psi3|psi3> = $(inner(psi3, psi3))")
    println("<psi4|psi4> = $(inner(psi4, psi4))")
    println("<psi1|psi2> = $(inner(psi1, psi2))")
    println("<psi1|psi3> = $(inner(psi1, psi3))")
    println("<psi1|psi4> = $(inner(psi1, psi4))")
    println("<psi2|psi3> = $(inner(psi2, psi3))")
    println("<psi2|psi4> = $(inner(psi2, psi4))")
    println("<psi3|psi4> = $(inner(psi3, psi4))")

    nothing
end

# Getting four degenerate states with an energy of -104, as we don't have the offset.
# Computation time is very long at over ten minutes.
# The overlap for different wavefunctions is not as precise as I would like; ~10^-3 accuracy.
# https://docs.itensor.org/ITensorMPS/stable/Observer.html may limit the number of sweeps and dimensions