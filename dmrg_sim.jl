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

function get_energies_measure_save(L, bc, parity, t, g, nstates, fname;
                                   weight=100,
                                   cutoff=1e-9,
                                   nsweeps=20,
                                   dmrg_tol=1e-8,
                                   m=100,
                                   eigsolve_krylovdim=10,
                                   noise=[0.0])
    e0, v0 = gs_dmrg(L, bc, parity,
                     t, g;
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
        ej, vj = es_dmrg(L, bc, parity,
                         t, g, vs;
                            weight,
                            cutoff,
                            nsweeps,
                            dmrg_tol,
                            m,
                            eigsolve_krylovdim,
                            noise)
        println("e$(j): $(ej)")
        push!(vs, vj)
        push!(es, ej)
        for k = 1:j
            println("Overlap |<$(k)|$(j+1)>|")
            println(abs(inner(vs[k], vj)))
        end
        println()
    end
    println("Energies: $(es)")
    i0 = argmin(es)
    psi0 = vs[i0]
    println("i0: $(i0)")
    z = expect(psi0, "Sz")
    println(z)
    xx = correlation_matrix(psi0, "Sx","Sx")    
    yy = correlation_matrix(psi0, "Sx","Sx")
    zz = correlation_matrix(psi0, "Sz","Sz")
    entropies = [entropy(psi0, i) for i = 1:L]
    h5open(fname, "w") do fid
        fid["energies"] = sort(es)
        fid["t"] = t
        fid["g"] = g
        fid["bc"] = bc
        fid["parity"] = parity
        fid["z"] = z
        fid["xx"] = xx
        fid["yy"] = yy
        fid["zz"] = zz
        fid["entropies"] = entropies
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
    println("Params: ")
    display(params)
    L = params["L"]
    bc = params["bc"]
    parity = params["parity"]
    t = params["t"]
    g = params["g"]
    nstates = params["nstates"]
    fname = params["fname"]
    dmrg_params = params["dmrg_params"]
    dmrg_params = Dict(Symbol(k) => v for (k,v) in dmrg_params)
    get_energies_measure_save(L, bc, parity,
                              t, g, nstates, fname; dmrg_params...)
end

if ARGS != []
    main()
end