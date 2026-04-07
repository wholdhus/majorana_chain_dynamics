using ITensors, ITensorMPS, DataFrames, CSV
using LinearAlgebra
include("tebd_sim.jl")

function gs_dmrg(L, bc, t0, g;
                 parity=1,
                 cutoff=1e-9,
                 nsweeps=20,
                 dmrg_tol=1e-8,
                 m=100,
                 eigsolve_krylovdim=10,
                 noise=[0.0])
    if Threads.nthreads() > 1
        BLAS.set_num_threads(1)
        ITensors.Strided.set_num_threads(1)
        println("Using threaded blocksparse with ", Threads.nthreads(), " threads!")
        ITensors.enable_threaded_blocksparse(true)
    end

    sites = siteinds("S=1/2", L, conserve_szparity=true)

    H0 = make_H0(L, sites, bc)
    H3 = make_H3(L, sites, bc)
    H = t0*H0 + g*H3
    
    state = ["Up" for n=1:L]
    if parity == -1
        println("Odd parity")
        state[1] = "Dn"
    else
        println("Even parity")
    end
    psi0 = random_mps(sites, state, linkdims=4)
    observer = DMRGObserver(energy_tol=dmrg_tol)
    md = Int(round(m/6))
    maxdim = [min(md, 10), min(md, 20), md, 2*md, 3*md, 4*md, 5*md, m]
    println(maxdim)
    E0, psi0 = dmrg(H, psi0; nsweeps, maxdim=maxdim, cutoff, observer, outputlevel=1, noise, eigsolve_krylovdim)
    
   return E0, psi0
end

function es_dmrg(L, bc, t0, g, psis;
                 weight=100,
                 parity=1,
                 cutoff=1e-9,
                 nsweeps=20,
                 dmrg_tol=1e-8,
                 m=100,
                 eigsolve_krylovdim=10,
                 noise=[0.0])
    if Threads.nthreads() > 1
        BLAS.set_num_threads(1)
        ITensors.Strided.set_num_threads(1)
        println("Using threaded blocksparse with ", Threads.nthreads(), " threads!")
        ITensors.enable_threaded_blocksparse(true)
    end

    sites = siteinds(psis[1])

    H0 = make_H0(L, sites, bc)
    H3 = make_H3(L, sites, bc)
    H = t0*H0 + g*H3

    state = ["Up" for n=1:L]
    if parity == -1
        println("Odd parity")
        state[1] = "Dn"
    else
        println("Even parity")
    end
    psi0 = random_mps(sites, state, linkdims=4)
    observer = DMRGObserver(energy_tol=dmrg_tol)
    md = Int(round(m/6))
    maxdim = [min(md, 10), min(md, 20), md, 2*md, 3*md, 4*md, 5*md, m]
    E, psi = dmrg(H, psis, psi0; weight=weight, nsweeps, maxdim=maxdim, cutoff, observer, outputlevel=1, noise, eigsolve_krylovdim)
    E = inner(psi', H, psi)
   return E, psi
end

function get_energies(L, bc, t0, gs, nstates;
                      weight=100,
                      parity=1,
                      cutoff=1e-9,
                      nsweeps=20,
                      dmrg_tol=1e-8,
                      m=100,
                      eigsolve_krylovdim=10,
                      noise=[0.0])
    energies = zeros((length(gs), nstates))
    for (i, g) in enumerate(gs)
        println("g = $g")
        e0, v0 = gs_dmrg(L, bc, t0, g;
                         parity,
                         cutoff,
                         nsweeps,
                         dmrg_tol,
                         m,
                         eigsolve_krylovdim,
                         noise)
        vs = [v0]
        es = [e0]
        println("e0: $(e0)")
        println()
        for j in 1:(nstates-1)
            ej, vj = es_dmrg(L, bc, t0, g, vs;
                             weight,
                             parity,
                             cutoff,
                             nsweeps,
                             dmrg_tol,
                             m,
                             eigsolve_krylovdim,
                             noise)
            println("e$(j): $(ej)")
            println()
            push!(vs, vj)
            push!(es, ej)
        end
        energies[i,:] = es
        println("Energies for g = $(g): $(energies[i,:])")
        println()
    end
    return gs, energies
end

function get_energies_theta(L, bc, steps, nstates;
                            weight=100,
                            parity=1,
                            cutoff=1e-9,
                            nsweeps=20,
                            dmrg_tol=1e-8,
                            m=100,
                            eigsolve_krylovdim=10,
                            noise=[0.0])
    ets = zeros(steps, nstates)
    thetas = LinRange(0, pi, steps)
    ts = sin.(thetas)
    gs = cos.(thetas)
    println("thetas:")
    println(thetas)
    println("gs:")
    println(gs)
    println("ts:")
    println(ts)
    for i in 1:steps
        println("g = $(gs[i]), t = $(ts[i])")
        e0, v0 = gs_dmrg(L, bc, ts[i], gs[i];
                         parity,
                         cutoff,
                         nsweeps,
                         dmrg_tol,
                         m,
                         eigsolve_krylovdim,
                         noise)
        vs = [v0]
        es = [e0]
        println("e0: $(e0)")
        println()
        for j in 1:(nstates-1)
            ej, vj = es_dmrg(L, bc, ts[i], gs[i], vs;
                             weight,
                             parity,
                             cutoff,
                             nsweeps,
                             dmrg_tol,
                             m,
                             eigsolve_krylovdim,
                             noise)
            println("e$(j): $(ej)")
            println()
            push!(vs, vj)
            push!(es, ej)
        end
        ets[i,:] = es
        println("Energies for g = $(gs[i]), t = $(ts[i]): $(ets[i,:])")
        println()
    end
    return thetas, ets
end