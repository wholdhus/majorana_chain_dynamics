using ITensors, ITensorMPS
let
    # Create spin-1/2 indices
    N = 50
    sites = siteinds("S=1/2", N)

    # Create the Hamiltonian of the Ising chain at its self-dual critical point
    # from the O'Brien and Fendley paper.
    osI = OpSum()
    for j in 1:(N - 1)
        osI += -2, "Sx", j # Not sure if this should be -1 or -2 as S=1/2 might make them 1/4 https://docs.itensor.org/ITensorMPS/stable/IncludedSiteTypes.html#%22Qubit%22-and-%22S1/2%22-Operators
        osI += -4, "Sz", j, "Sz", j + 1 # Same here 
    end
    H_I = MPO(osI, sites)

    # Create the Hamiltonian of the three-spin interaction from the O'Brien and Fendley paper.
    os3 = OpSum()
    for j in 1:(N - 2)
        os3 += 8, "Sx", j, "Sz", j + 1, "Sz", j + 2
        os3 += 8, "Sz", j, "Sz", j + 1, "Sx", j + 2
    end
    H_3 = MPO(os3, sites)

    # Coupling coefficients
    lambdaI = 0.431
    lambda3 = 0.369
    # Such that lambda3 / lambdaI = 0.856

    # Initial energy offset
    L = 2
    E0 = L * (lambdaI^2 + lambda3^2) / lambda3

    # Making the initial energy offset a matrix product operator to match the Hamiltonians
    osE0 = OpSum()
    for j in 1:N
        osE0 += E0, "Id", j   # "Id" makes this a MPO term apparently
    end
    E0MPO = MPO(osE0, sites)

    # Full Hamiltonian from paper
    H = 2 * lambdaI * H_I + lambda3 * H_3 + E0MPO

    # Create an initial random matrix product state
    psi0 = random_mps(sites)
    nsweeps = 8
    maxdim = [1, 2, 3, 10, 25, 50, 100, 500]
    cutoff = 1.0e-10

    # Run the DMRG algorithm, returning energy
    # (dominant eigenvalue) and optimized MPS
    energy, psi = dmrg(H, psi0; nsweeps, maxdim, cutoff)
    println("Final energy = $energy")

    nothing
end