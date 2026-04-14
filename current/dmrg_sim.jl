using ITensors, ITensorMPS, HDF5
using LinearAlgebra
using YAML
include("operators.jl")

function entropy(state, bond; spectrum=false)
	s_orth = orthogonalize(state, bond)
	U,S,V = svd(s_orth[bond], 
				(linkinds(s_orth, bond-1)..., 
				 siteinds(s_orth, bond)...))
	SvN = 0.0
	ps = zeros(dim(S, 1))
	for n=1:dim(S, 1)
		p = S[n,n]^2
		SvN -= p * log(p)
		ps[n] = p
	end
	if spectrum
		return SvN, ps
	else
		return SvN
	end
end

function gs_dmrg(L, bc, parity,
                 t0, g;
                 cutoff=1e-9,
                 nsweeps=20,
                 dmrg_tol=1e-8,
                 m=100,
                 eigsolve_krylovdim=10,
                 noise=[0.0])
    sites = siteinds("S=1/2", L, conserve_szparity=true)
    H0 = make_H0(L, sites, bc)
    H3 = make_H3(L, sites, bc)
    H = t0*H0 + g*H3
    state = ["Up" for n=1:L]
    if parity == -1
        state[1] = "Dn"
    end
    psi0 = random_mps(sites, state, linkdims=4)
    observer = DMRGObserver(energy_tol=dmrg_tol)
    md = Int(round(m/6))
    maxdim = [min(md, 10), min(md, 20), md, 2*md, 3*md, 4*md, 5*md, m]
    E0, psi0 = dmrg(H, psi0; nsweeps, maxdim=maxdim, cutoff, observer, outputlevel=1, noise, eigsolve_krylovdim)
    return E0, psi0
end

function es_dmrg(L, bc, parity,
                 t0, g, 
                 psis;
                 weight=100,
                 cutoff=1e-9,
                 nsweeps=20,
                 dmrg_tol=1e-8,
                 m=100,
                 eigsolve_krylovdim=10,
                 noise=[0.0])
    sites = siteinds(psis[1])
    H0 = make_H0(L, sites, bc)
    H3 = make_H3(L, sites, bc)
    H = t0*H0 + g*H3
    state = ["Up" for n=1:L]
    if parity == -1
        state[1] = "Dn"
    end
    psi0 = random_mps(sites, state, linkdims=4)
    observer = DMRGObserver(energy_tol=dmrg_tol)
    md = Int(round(m/6))
    maxdim = [min(md, 10), min(md, 20), md, 2*md, 3*md, 4*md, 5*md, m]
    E, psi = dmrg(H, psis, psi0; weight=weight, nsweeps, maxdim=maxdim, cutoff, observer, outputlevel=1, noise, eigsolve_krylovdim)
    E = inner(psi', H, psi)
    return E, psi
end

function get_energies_measure_save(L, bc, parity, t, g, nstates, fname;
                                   weight=100,
                                   cutoff=1e-9,
                                   nsweeps=20,
                                   dmrg_tol=1e-8,
                                   m=100,
                                   eigsolve_krylovdim=10,
                                   noise=[0.0])
    # Compute ground state
    e0, v0 = gs_dmrg(L, bc, parity, t, g;
                     cutoff, nsweeps, dmrg_tol,
                     m, eigsolve_krylovdim, noise)
    vs = [v0]
    es = [e0]

    # Compute excited states
    for j in 1:(nstates-1)
        ej, vj = es_dmrg(L, bc, parity, t, g, vs;
                         weight, cutoff, nsweeps, dmrg_tol,
                         m, eigsolve_krylovdim, noise)
        push!(vs, vj)
        push!(es, ej)
    end

    # Sort states by energy
    sorted_indices = sortperm(es)
    vs = vs[sorted_indices]
    es = es[sorted_indices]
    println("Energies: $(sort(es))")

    # Allocate measurement arrays
    X  = Vector{Any}(undef, nstates)
    Y  = Vector{Any}(undef, nstates)
    Z  = Vector{Any}(undef, nstates)
    XX = Vector{Any}(undef, nstates)
    YY = Vector{Any}(undef, nstates)
    ZZ = Vector{Any}(undef, nstates)
    entropies = Vector{Any}(undef, nstates)
    
    # Measure all states
    for k in 1:nstates
        X[k] = real.(expect(vs[k], "X"))
        Y[k] = real.(expect(vs[k], "Y"))
        Z[k] = real.(expect(vs[k], "Z"))
        XX[k] = real.(correlation_matrix(vs[k], "X", "X"))
        YY[k] = real.(correlation_matrix(vs[k], "Y", "Y"))
        ZZ[k] = real.(correlation_matrix(vs[k], "Z", "Z"))
        entropies[k]= [entropy(vs[k], i) for i = 2:L-1]
    end
    
    # Save to HDF5
    h5open(fname, "w") do fid
        fid["energies"]  = es
        fid["t"]         = t
        fid["g"]         = g
        fid["bc"]        = bc
        fid["parity"]    = parity
        fid["X"]         = reduce(vcat, [x' for x in X])
        fid["Y"]         = reduce(vcat, [x' for x in Y])
        fid["Z"]         = reduce(vcat, [x' for x in Z])
        fid["XX"]        = reduce(vcat, XX)
        fid["YY"]        = reduce(vcat, YY)
        fid["ZZ"]        = reduce(vcat, ZZ)
        fid["entropy"]   = reduce(vcat, [e' for e in entropies])
    end
end

function main()
    if Threads.nthreads() > 1
        BLAS.set_num_threads(1)
        ITensors.Strided.set_num_threads(1)
        println("Using threaded blocksparse with ", Threads.nthreads(), " threads!")
        ITensors.enable_threaded_blocksparse(true)
    end
    params = YAML.load_file(ARGS[1])
    L       = params["L"]
    bc      = params["bc"]
    parity  = params["parity"]
    t       = params["t"]
    g       = params["g"]
    nstates = params["nstates"]
    fname   = params["fname"]
    println("Running L = $(L), bc = $(bc), parity = $(parity), t = $(t), g = $(g), nstates = $(nstates)")
    dmrg_params = params["dmrg_params"]
    dmrg_params = Dict(Symbol(k) => v for (k,v) in dmrg_params)
    get_energies_measure_save(L, bc, parity, t, g, nstates, fname; dmrg_params...)
    println("Done! Saved to $(fname)")
end

if ARGS != []
    main()
end