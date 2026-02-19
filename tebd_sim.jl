using ITensors, ITensorMPS, DataFrames, CSV
using LinearAlgebra

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

function local_H(t1, g, s1, s2, s3; signs=[1.0, 1.0, 1.0, 1.0])
    X_part = -signs[1]*op("Z", s1) * op("Id", s2) * op("Id", s3)
    H = 2*t1*X_part
    if signs[2] != 0.0
        ZZ_part = -signs[2]*op("X", s1) * op("X", s2) * op("Id", s3)
        H += 2*t1*ZZ_part
    end
    # Build term for Three-spin part H3
    if signs[3] != 0
        H += g*signs[3]*op("Z", s1) * op("X", s2) * op("X", s3) 
    end
    if signs[4] != 0
        H += g*signs[4]*op("X", s1) * op("X", s2) * op("Z", s3)
    end
    return H
end

function build_trotter_gates(sites, tau, t1, g, bc)
    L = length(sites)
    gates = ITensor[]
    
    # Forward sweep
    for j in 1:L-2
        H = local_H(t1, g,
                    sites[j], sites[j+1], sites[j+2])
        push!(gates, exp(-im * tau/2 * H))
    end
    if bc == "PBC"
        H = local_H(t1, g,
                    sites[L-1], sites[L], sites[1])
        push!(gates, exp(-im * tau/2 * H))
        H = local_H(t1, g,
                    sites[L], sites[1], sites[2])
        push!(gates, exp(-im * tau/2 * H))
    elseif bc == "APBC"
        H = local_H(t1, g,
                    sites[L-1], sites[L], sites[1], signs=[1.0, 1.0, -1.0, 1.0])
        push!(gates, exp(-im * tau/2 * H))
        H = local_H(t1, g,
                    sites[L], sites[1], sites[2], signs=[1.0, -1.0, 1.0, -1.0])
        push!(gates, exp(-im * tau/2 * H))
    else
        H = local_H(t1, g,
                    sites[L-1], sites[L], sites[1], signs=[1.0, 1.0, 0.0, 0.0])
        push!(gates, exp(-im * tau/2 * H))
        H = local_H(t1, g,
                    sites[L], sites[1], sites[2], signs=[1.0, 0.0, 0.0, 0.0])
        push!(gates, exp(-im * tau/2 * H))
    end
    
    # Backward sweep
    append!(gates, reverse(gates))
    return gates
end

function tebd_sim(L, bc, t1, g, omega, periods, op_str, ind;
                  parity=1,
                  steps_per_period=10,
                  cutoff=1e-9,
                  nsweeps=20,
                  dmrg_tol=1e-8,
                  maxdim=100,
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
    H = 2*t1*H0 + g*H3
    
    state = ["Up" for n=1:L]
    if parity == -1
        println("Changing to odd parity")
        state[1] = "Dwn"
    end
    psi0 = random_mps(sites, state, linkdims=4)
    
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

    gate_sets = [build_trotter_gates(sites, tau, drive_t1(t), g, bc) for t in ts]

    psi_t = copy(psi0)
    phi_t = apply(O, psi0)

    steps = steps_per_period*periods + 1
    overlaps = zeros(Float64, steps)
    autocorrs = zeros(ComplexF64, steps)
    maxdims = zeros(Int, steps)

    overlaps[1] = inner(psi_t, psi_t)
    autocorrs[1] = inner(phi_t, apply(O, psi_t))
    maxdims[1] = maxlinkdim(psi_t)

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
        maxdims[i] = maxlinkdim(psi_t)
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
                       "maxdims" => maxdims, 
                       "re_autoc" => real(autocorrs), 
                       "im_autoc" => imag(autocorrs))
    fname = "L$(L)_$(bc)_g$(g)_omega$(omega)_N$(periods)_steps$(steps_per_period)_$(op_str)$(ind).csv"
    CSV.write(fname, output)
    println("Wrote data to $fname")
    return times, overlaps, autocorrs
end

# Example usage: tebd_sim(L, bc, t1, g, omega, periods, op_str, ind)