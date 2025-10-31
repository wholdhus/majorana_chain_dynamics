using ITensors, ITensorMPS, PythonPlot

let
    # Create L Spin-1/2 Indices
    L = 20
    sites = siteinds("S=1/2", L)

    # Create the Hamiltonian of the Ising chain at its self-dual critical point from the O'Brien and Fendley paper.
    osI = OpSum()
    for j in 1:(L - 1)
        osI -= "X", j 
        osI -= "Z", j, "Z", j + 1
    end
    osI -= "X", L
    osI -= "Z", L, "Z", 1
    HI = MPO(osI, sites)

    # Create the Hamiltonian of the three-spin interaction from the O'Brien and Fendley paper.
    os3 = OpSum()
    for j in 1:(L - 2)
        os3 += "X", j, "Z", j + 1, "Z", j + 2
        os3 += "Z", j, "Z", j + 1, "X", j + 2
    end
    os3 += "X", L - 1, "Z", L, "Z", 1
    os3 += "X", L, "Z", 1, "Z", 2
    os3 += "Z", L - 1, "Z", L, "X", 1
    os3 += "Z", L, "Z", 1, "X", 2
    H3 = MPO(os3, sites)

    # Coupling coefficients for Exact G.S.
        lambdaI = 1
        lambda3 = 1
    
    # Coupling coefficients for TCI such that lambda3 / lambdaI = 0.856
        # lambdaI = 0.759685
        # lambda3 = 0.650291
    
    # Energy offset
    EO = L * (lambdaI^2 + lambda3^2) / lambda3

    # Adding the energy offset to just one site
    osEO = OpSum()
        osEO += EO, "Id", 1 # "Id" is the identity operator
    E0 = MPO(osEO, sites)

    # Full Hamiltonian from paper
    H = 2 * lambdaI * HI + lambda3 * H3 + E0

    # Create an initial random matrix product state
    psi0 = random_mps(sites)
    nsweeps = 20

    maxdim = [10, 33, 66, 133, 200]
    mindim = [10, 10, 33, 66, 100]
    cutoff = 1.0e-10

    # Run the DMRG algorithm, returning energy
    # (dominant eigenvalue) and optimized MPS
    println("Maxdim per sweep: ", maxdim)
    println("Mindim per sweep: ", mindim)
    
    energy1, psi1 = dmrg(H, psi0; nsweeps, mindim, maxdim, cutoff)
    energy2, psi2 = dmrg(H, [psi1], psi0; nsweeps, mindim, maxdim, cutoff, weight = 100)
    energy3, psi3 = dmrg(H, [psi1, psi2], psi0; nsweeps, mindim, maxdim, cutoff, weight = 100)
    energy4, psi4 = dmrg(H, [psi1, psi2, psi3], psi0; nsweeps, mindim, maxdim, cutoff, weight = 100)
    
    println("Final Energy 1 = $energy1")
    println("Final Energy 2 = $energy2")
    println("Final Energy 3 = $energy3")
    println("Final Energy 4 = $energy4")
    
    println("Overlap 1,1 = $(inner(psi1, psi1))")
    println("Overlap 1,2 = $(inner(psi1, psi2))")
    println("Overlap 1,3 = $(inner(psi1, psi3))")
    println("Overlap 1,4 = $(inner(psi1, psi4))")
    println("Overlap 2,3 = $(inner(psi2, psi3))")
    println("Overlap 2,4 = $(inner(psi2, psi4))")
    println("Overlap 3,4 = $(inner(psi3, psi4))")

    # Compute Z-Z correlation matrix for the first G.S.
    C1zz = correlation_matrix(psi1, "Z", "Z")

    nothing
end