using ITensors, ITensorMPS
let
    # Create spin-1/2 indices
    N = 40
    sites = siteinds("S=1/2", N)

    # Create the Hamiltonian of the Ising chain at its self-dual critical point
    # from the O'Brien and Fendley paper.
    os_I = OpSum()
    for j in 1:(N - 1)
        os_I -= "X", j 
        os_I -= "Z", j, "Z", j + 1
    end
    # Boundary conditions
    os_I -= "X", N # j = N
    os_I -= "Z", N, "Z", 1

    H_I = MPO(os_I, sites)

    # Create the Hamiltonian of the three-spin interaction from the O'Brien and Fendley paper.
    os_3 = OpSum()
    for j in 1:(N - 2)
        os_3 += "X", j, "Z", j + 1, "Z", j + 2
        os_3 += "Z", j, "Z", j + 1, "X", j + 2
    end
    os_3 += "X", N - 1, "Z", N, "Z", 1
    os_3 += "X", N, "Z", 1, "Z", 2
    os_3 += "Z", N - 1, "Z", N, "X", 1
    os_3 += "Z", N, "Z", 1, "X", 2
    
    H_3 = MPO(os_3, sites)

    # Coupling coefficients
    lambda_I = 1
    lambda_3 = 1
    # Both equal one is when we are at the ground state

    # Initial energy offset
    E0 = N * (lambda_I^2 + lambda_3^2) / lambda_3

    # Making the initial energy offset a matrix product operator to match the Hamiltonians
    osE0 = OpSum()
        osE0 += E0, "Id", 1
    E0MPO = MPO(osE0, sites)

    # Full Hamiltonian from the paper
    H = 2 * lambda_I * H_I + lambda_3 * H_3 + E0MPO

    # Create an initial random matrix product state
    psi0 = random_mps(sites)

    # Sweeps and maximum dimensions
    nsweeps = 10
    maxdim = [5, 10, 25, 50, 100, 500, 1000]
    cutoff = 1.0e-10

    # Outputs
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