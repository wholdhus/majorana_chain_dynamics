using ITensors, ITensorMPS
let
    # Create spin-1/2 indices
    N = 25
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
    for j in 1:N
        osE0 += (E0 / N), "Id", j
    end
    E0MPO = MPO(osE0, sites)

    # Full Hamiltonian from the paper
    H = 2 * lambda_I * H_I + lambda_3 * H_3 + E0MPO

    # Create an initial random matrix product state
    psi0 = random_mps(sites)

    # Sweeps and maximum dimensions
    nsweeps = 12
    maxdim = [10, 25, 50, 100, 250, 500, 1000, 2500, 5000]
    cutoff = 1.0e-12

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
# With the cutoff at 1.0e-14 and 25 sites, it takes over two hours to run the code, assuming no problems.
# Energy 4 uses about 3700 diensions, taking significantly n=more time than the others, so we could probably
# cut down on the accuracy, as long as we keep the three-fold degenerate zero energy levels for energies 1, 2, and 3.

# Outputs for cutoff being 1.0e-14:
#After sweep 1 energy=0.05227581899130129  maxlinkdim=4 maxerr=0.00E+00 time=0.025
#After sweep 2 energy=2.881330414439276e-5  maxlinkdim=16 maxerr=6.65E-15 time=0.031
#After sweep 3 energy=1.429794113874527e-7  maxlinkdim=39 maxerr=9.64E-15 time=0.190
#After sweep 4 energy=6.963318810448982e-10  maxlinkdim=23 maxerr=9.99E-15 time=0.173
#After sweep 5 energy=1.312727704316785e-12  maxlinkdim=3 maxerr=9.74E-15 time=0.026
#After sweep 6 energy=1.5987211554602254e-13  maxlinkdim=1 maxerr=1.16E-15 time=0.019
#After sweep 7 energy=1.652011860642233e-13  maxlinkdim=1 maxerr=1.27E-27 time=0.028
#After sweep 8 energy=1.652011860642233e-13  maxlinkdim=1 maxerr=1.30E-27 time=0.016
#After sweep 9 energy=1.438849039914203e-13  maxlinkdim=1 maxerr=1.27E-27 time=0.017
#After sweep 10 energy=1.4566126083082054e-13  maxlinkdim=1 maxerr=1.26E-27 time=0.016
#After sweep 11 energy=1.5631940186722204e-13  maxlinkdim=1 maxerr=1.25E-27 time=0.023
#After sweep 12 energy=1.545430450278218e-13  maxlinkdim=1 maxerr=1.27E-27 time=0.027
#After sweep 1 energy=2.0663378045323135  maxlinkdim=4 maxerr=0.00E+00 time=0.023
#After sweep 2 energy=1.5362930130475214  maxlinkdim=16 maxerr=0.00E+00 time=0.040
#After sweep 3 energy=0.9934587763837612  maxlinkdim=50 maxerr=1.12E-08 time=0.225
#After sweep 4 energy=0.0004450753450415633  maxlinkdim=100 maxerr=2.64E-10 time=1.432
#After sweep 5 energy=5.495724408888236e-7  maxlinkdim=210 maxerr=9.93E-15 time=4.394
#After sweep 6 energy=1.2708945007489092e-9  maxlinkdim=66 maxerr=9.94E-15 time=2.534
#After sweep 7 energy=1.8404194870796477e-13  maxlinkdim=6 maxerr=9.63E-15 time=0.106
#After sweep 8 energy=-8.758496669663209e-13  maxlinkdim=2 maxerr=3.87E-15 time=0.028
#After sweep 9 energy=-9.022365106210075e-13  maxlinkdim=2 maxerr=5.82E-21 time=0.029
#After sweep 10 energy=-8.532643297539597e-13  maxlinkdim=2 maxerr=3.54E-24 time=0.023
#After sweep 11 energy=-8.874618091312085e-13  maxlinkdim=2 maxerr=1.24E-26 time=0.032
#After sweep 12 energy=-8.818323620714849e-13  maxlinkdim=2 maxerr=3.98E-28 time=0.025
#After sweep 1 energy=2.0678550491534153  maxlinkdim=4 maxerr=0.00E+00 time=0.031
#After sweep 2 energy=1.5385650457976818  maxlinkdim=16 maxerr=0.00E+00 time=0.051
#After sweep 3 energy=1.3368944091597188  maxlinkdim=50 maxerr=5.88E-09 time=0.245
#After sweep 4 energy=1.2651769783567444  maxlinkdim=100 maxerr=1.02E-08 time=1.898
#After sweep 5 energy=0.0022739679050988444  maxlinkdim=250 maxerr=4.95E-10 time=6.975
#After sweep 6 energy=8.049860866323376e-7  maxlinkdim=411 maxerr=9.98E-15 time=23.968
#After sweep 7 energy=4.904896590308351e-10  maxlinkdim=40 maxerr=9.95E-15 time=4.874
#After sweep 8 energy=-5.318678860928837e-13  maxlinkdim=3 maxerr=9.92E-15 time=0.059
#After sweep 9 energy=-6.486002858394507e-13  maxlinkdim=2 maxerr=1.25E-15 time=0.028
#After sweep 10 energy=-6.036324461988256e-13  maxlinkdim=2 maxerr=3.25E-20 time=0.036
#After sweep 11 energy=-6.120532468827766e-13  maxlinkdim=2 maxerr=2.54E-20 time=0.028
#After sweep 12 energy=-6.318888366511181e-13  maxlinkdim=2 maxerr=2.58E-20 time=0.034
#After sweep 1 energy=2.0678709493691265  maxlinkdim=4 maxerr=0.00E+00 time=0.036
#After sweep 2 energy=1.5391357855688812  maxlinkdim=16 maxerr=0.00E+00 time=0.049
#After sweep 3 energy=1.337802585113676  maxlinkdim=50 maxerr=1.17E-08 time=0.278
#After sweep 4 energy=1.2688362638707193  maxlinkdim=100 maxerr=7.99E-09 time=1.853
#After sweep 5 energy=1.2447471517466582  maxlinkdim=250 maxerr=2.17E-10 time=7.251
#After sweep 6 energy=1.2320455825802186  maxlinkdim=500 maxerr=6.38E-11 time=37.537
#After sweep 7 energy=1.2260917498347257  maxlinkdim=1000 maxerr=5.32E-12 time=157.022
#After sweep 8 energy=1.2233746693613696  maxlinkdim=2500 maxerr=1.23E-13 time=613.929
#After sweep 9 energy=1.221799098920942  maxlinkdim=3616 maxerr=9.99E-15 time=1461.548
#After sweep 10 energy=1.2208953385741979  maxlinkdim=3673 maxerr=9.98E-15 time=1651.254
#After sweep 11 energy=1.2203106727415978  maxlinkdim=3669 maxerr=9.97E-15 time=1793.449
#After sweep 12 energy=1.2199339510475338  maxlinkdim=3651 maxerr=9.99E-15 time=1563.831
#Final energy 1 = 1.545430450278218e-13
#Final energy 2 = -8.818323620714849e-13
#Final energy 3 = -6.318888366511181e-13
#Final energy 4 = 1.2199339510475338
#<psi1|psi1> = 0.9999999999999989
#<psi2|psi2> = 1.0000000000000002
#<psi3|psi3> = 0.9999999999999992
#<psi4|psi4> = 1.0000000000000064
#<psi1|psi2> = -3.5914196963582334e-18
#<psi1|psi3> = -2.0124027699621205e-15
#<psi1|psi4> = -0.00015503565215449933
#<psi2|psi3> = -2.9801730346390968e-8
#<psi2|psi4> = -1.5908277685378344e-6
#<psi3|psi4> = -0.00010761270736690961