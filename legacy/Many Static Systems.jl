# Functions for Creating Hamiltonian and Running Simulation

using ITensors, ITensorMPS
using LinearAlgebra
using DataFrames, CSV

function make_H0(L, sites, bc)
    osI = OpSum()
    for j in 1:L
        osI -= "Z", j
    end   
    for j in 1:L-1
        osI -= "X", j, "X", j+1
    end
    
    if bc == "PBC"
        osI -= "X", L, "X", 1
    elseif bc == "APBC"
        osI += "X", L, "X", 1
    end

    return MPO(osI, sites)
end

function make_H3(L, sites, bc)
    os3 = OpSum()
    for j in 1:L-2
        os3 += "Z", j, "X", j+1, "X", j+2
        os3 += "X", j, "X", j+1, "Z", j+2
    end

    if bc == "PBC"
        os3 += "Z", L-1, "X", L, "X", 1
        os3 += "X", L-1, "X", L, "Z", 1
        os3 += "Z", L, "X", 1, "X", 2
        os3 += "X", L, "X", 1, "Z", 2
    elseif bc == "APBC"
        os3 -= "Z", L-1, "X", L, "X", 1
        os3 += "X", L-1, "X", L, "Z", 1
        os3 += "Z", L, "X", 1, "X", 2
        os3 -= "X", L, "X", 1, "Z", 2
    end

    return MPO(os3, sites)
end

function local_H(t0, g, s1, s2, s3; signs=[1.0,1.0,1.0,1.0])

    H = 2*t0*(-signs[1]*op("Z", s1)*op("Id", s2)*op("Id", s3))

    if signs[2] != 0
        H += 2*t0*(-signs[2]*op("X", s1)*op("X", s2)*op("Id", s3))
    end

    if signs[3] != 0
        H += g*signs[3]*op("Z", s1)*op("X", s2)*op("X", s3)
    end
    if signs[4] != 0
        H += g*signs[4]*op("X", s1)*op("X", s2)*op("Z", s3)
    end

    return H
end

function build_trotter_gates(sites, tau, t0, g, bc)

    L = length(sites)
    gates = ITensor[]
    
    for j in 1:L-2
        H = local_H(t0, g, sites[j], sites[j+1], sites[j+2])
        push!(gates, exp(-im*tau/2 * H))
    end
    
    append!(gates, reverse(gates))
    return gates
end

# Running Time-Independent DMRG for Every System Size, Boundary Condition, and g-Value

t0 = 1
nsweeps = 50
cutoff = 1e-8
dmrg_tol = 1e-8
observer = DMRGObserver(energy_tol=dmrg_tol)
maxdim = [10, 50, 100, 500]
parity = 1

# g-Values
g_vals = 0.0 : 0.1 : 4.0

# System Sizes
L_vals = vcat([4, 6, 8, 12], collect(24 : 12 : 204))

# Boundary conditions
bcs = ["OBC", "PBC", "APBC"]

# Storage
results = DataFrame(
    L = Int[],
    bc = String[],
    g = Float64[],
    energy = Float64[],
    maxdim = Int[]
)

# :(

for L in L_vals
    sites = siteinds("S=1/2", L, conserve_szparity=true)
    for bc in bcs
        for g in g_vals

            # Build Hamiltonian
            H0 = make_H0(L, sites, bc)
            H3 = make_H3(L, sites, bc)
            H = 2*t0*H0 + g*H3

            state = fill("Up", L)
            if parity == -1
                state[1] = "Dwn"
            end

            psi = random_mps(sites, state, linkdims=4)

            E0, psi0 = dmrg(H, psi; nsweeps, maxdim, cutoff, observer, outputlevel=0, noise=[0.0], eigsolve_krylovdim=10)

            push!(results, (L, bc, g, E0, maxlinkdim(psi0)))
        end
    end
    println("Finished for L = $(L)")
end

fname = "DMRG_Static_Sweeps.csv"
CSV.write(fname, results)
println("Saved results to $fname")