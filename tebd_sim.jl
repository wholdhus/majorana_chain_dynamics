using ITensors, ITensorMPS, DataFrames, CSV

function make_H0(L, sites)
    osI = OpSum()
    for j in 1:L
        osI -= "Z", j
        osI -= "X", j, "X", mod1(j+1, L)
    end
    return MPO(osI, sites)
end

function make_H3(L, sites)
    os3 = OpSum()
    for j in 1:L
        os3 += "Z", j, "X", mod1(j+1, L), "X", mod1(j+2, L)
        os3 += "X", j, "X", mod1(j+1, L), "Z", mod1(j+2, L)
    end
    return MPO(os3, sites)
end

function build_trotter_gates(sites, tau, t1t, g)
    L = length(sites)
    gates = ITensor[]
    
    # Forward sweep
    for j in 1:L
        s1, s2, s3 = sites[j], sites[mod1(j+1, L)], sites[mod1(j+2, L)]
        
        # Build term for Ising part HI
        X_part = -(1/3) * (op("Z", s1) * op("Id", s2) * op("Id", s3) +
                           op("Id", s1) * op("Z", s2) * op("Id", s3) +
                           op("Id", s1) * op("Id", s2) * op("Z", s3))
        ZZ_part = -op("X", s1) * op("X", s2) * op("Id", s3)
        H0 = X_part + ZZ_part
        
        # Build term for Three-spin part H3
        H3 = op("Z", s1) * op("X", s2) * op("X", s3) +
             op("X", s1) * op("X", s2) * op("Z", s3)

        # Define full Hamiltonian and create gate
        H = 2*t1t*H0 + g*H3
        push!(gates, exp(-im * tau/2 * H))
    end
    
    # Backward sweep
    append!(gates, reverse(gates))
    return gates
end

function tebd_sim(L, t1, g, omega, periods, op_str, ind;
                  parity=1,
                  steps_per_period=10,
                  cutoff=1e-9,
                  nsweeps=20,
                  dmrg_tol=1e-8,
                  maxdim=100,
                  eigsolve_krylovdim=10,
                  noise=[0.0])
    sites = siteinds("S=1/2", L, conserve_szparity=true)

    H0 = make_H0(L, sites)
    H3 = make_H3(L, sites)
    H = 2*t1*H0 + g*H3
    
    state = ["Up" for n=1:L]
    if parity == -1
        println("Changing to odd parity")
        state[1] = "Dwn"
    end
    psi0 = random_mps(sites, state)
    
    observer = DMRGObserver(energy_tol=dmrg_tol)
    md = Int(round(maxdim/6))
    maxdim = [md, 2*md, 3*md, 4*md, 5*md, maxdim]
    E0, psi0 = dmrg(H, psi0; nsweeps, maxdim, cutoff, observer, outputlevel=1, noise, eigsolve_krylovdim)
    
    println("DMRG Finished! E0 = $(E0), psi0 maxdim = $(maxlinkdim(psi0))")
    println("")
    drive_t1(t) = t1*cos(omega*t)

    T = 2*pi/omega
    tau = T/steps_per_period
    ts = [tau*(n-0.5) for n=1:steps_per_period] 

    O = op(op_str, sites[ind])

    gate_sets = [build_trotter_gates(sites, tau, drive_t1(t), g) for t in ts]

    psi_t = copy(psi0)
    phi_t = apply(O, psi0)

    steps = steps_per_period*periods + 1
    overlaps = zeros(Float64, steps)
    autocorrs = zeros(ComplexF64, steps)

    overlaps[1] = inner(psi_t, psi_t)
    autocorrs[1] = inner(phi_t, apply(O, psi_t))

    times = [tau*i for i=0:(steps-1)]
    for i in 2:steps
        t = times[i]
        println("t = $(t), t/T = $(t/T)")
        gates = gate_sets[mod1(i-1, steps_per_period)]
        psi_t = apply(gates, psi_t; cutoff=cutoff, maxdim=maximum(maxdim))
        normalize!(psi_t)

        phi_t = apply(gates, phi_t; cutoff=cutoff, maxdim=maximum(maxdim))
        normalize!(phi_t)

        overlaps[i] = abs2(inner(psi_t, psi0))
        autocorrs[i] = inner(phi_t, apply(O, psi_t))
        println("Current max bond dimension: $(maxlinkdim(psi_t))")
        println("Max bond dimension of phi: $(maxlinkdim(phi_t))")
        println("|<psi(t)|psi(0)>|^2: $(overlaps[i])")
        println("<O(t)|O(o)>^2: $(autocorrs[i])")
        println()
    end
    println("")
    println("Done!")

    output = DataFrame("time" => times, 
                       "overlap" => overlaps, 
                       "re_autoc" => real(autocorrs), 
                       "im_autoc" => imag(autocorrs))
    fname = "L$(L)_g$(g)_omega$(omega)_N$(periods).csv"
    CSV.write(fname, output)
    println("Wrote data to $fname")
    return times, overlaps, autocorrs
end