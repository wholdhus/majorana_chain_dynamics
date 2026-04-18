using ITensors, ITensorMPS, HDF5
using LinearAlgebra
using YAML
include("operators.jl")

function build_trotter_gates(sites, tau, t1, g, bc)
    L = length(sites)
    gates = ITensor[]

    # Forward sweep
    for j in 1:L-2
        H = local_H(t1, g, sites[j], sites[j+1], sites[j+2])
        push!(gates, exp(-im * tau/2 * H))
    end

    # Boundary terms depend on bc
    if bc == "PBC"
        H = local_H(t1, g, sites[L-1], sites[L], sites[1])
        push!(gates, exp(-im * tau/2 * H))
        H = local_H(t1, g, sites[L], sites[1], sites[2])
        push!(gates, exp(-im * tau/2 * H))
    elseif bc == "APBC"
        H = local_H(t1, g, sites[L-1], sites[L], sites[1], signs=[1.0, 1.0, -1.0, 1.0])
        push!(gates, exp(-im * tau/2 * H))
        H = local_H(t1, g, sites[L], sites[1], sites[2], signs=[1.0, -1.0, 1.0, -1.0])
        push!(gates, exp(-im * tau/2 * H))
    else  # OBC: zero out boundary couplings
        H = local_H(t1, g, sites[L-1], sites[L], sites[1], signs=[1.0, 1.0, 0.0, 0.0])
        push!(gates, exp(-im * tau/2 * H))
        H = local_H(t1, g, sites[L], sites[1], sites[2], signs=[1.0, 0.0, 0.0, 0.0])
        push!(gates, exp(-im * tau/2 * H))
    end

    # Backward sweep for symmetric (2nd-order) Trotter
    append!(gates, reverse(gates))
    return gates
end

function tebd_evolve(L, bc, parity, t1, g, omega, periods, ac_site, fname;
                     steps_per_period=10,
                     corr_every=1,
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

    # DMRG ground state at full static H
    H0_mpo = make_H0(L, sites, bc)
    H3_mpo = make_H3(L, sites, bc)
    H_static = t1*H0_mpo + g*H3_mpo

    state = ["Up" for n in 1:L]
    if parity == -1
        state[1] = "Dn"
    end
    psi0 = random_mps(sites, state, linkdims=4)

    observer = DMRGObserver(energy_tol=dmrg_tol)
    md = Int(round(maxdim/6))
    maxdim_schedule = [min(md, 10), min(md, 20), md, 2*md, 3*md, 4*md, 5*md, maxdim]
    println("DMRG maxdim schedule: $maxdim_schedule")

    E0, psi0 = dmrg(H_static, psi0; nsweeps, maxdim=maxdim_schedule, cutoff,
                    observer, outputlevel=1, noise, eigsolve_krylovdim)
    println("DMRG done. E0 = $E0, maxlinkdim = $(maxlinkdim(psi0))")

    truncate!(psi0; maxdim=maxdim, cutoff=cutoff)
    println("After truncation: maxlinkdim = $(maxlinkdim(psi0))")

    # Time evolution setup
    T = 2*pi / omega
    tau = T / steps_per_period
    nsteps = steps_per_period * periods

    # Midpoint times within one period (for gate construction)
    midpoints = [tau*(n - 0.5) for n in 1:steps_per_period]
    drive_t1(t) = t1 * cos(omega * t)

    # Pre-compute gate sets for each sub-step in one period
    gate_sets = [build_trotter_gates(sites, tau, drive_t1(t), g, bc) for t in midpoints]

    # Autocorrelation
    Ox = op("X", sites[ac_site])
    Oy = op("Y", sites[ac_site])
    Oz = op("Z", sites[ac_site])
    phi_x = apply(Ox, psi0)
    phi_y = apply(Oy, psi0)
    phi_z = apply(Oz, psi0)

    # Storage for nsteps+1 time points (includes t=0)
    times      = [tau*i for i in 0:nsteps]
    loschmidt  = zeros(Float64, nsteps+1)
    autoc_x    = zeros(ComplexF64, nsteps+1)
    autoc_y    = zeros(ComplexF64, nsteps+1)
    autoc_z    = zeros(ComplexF64, nsteps+1)
    energy_t   = zeros(Float64, nsteps+1)
    energy_0   = zeros(Float64, nsteps+1)
    maxdims    = zeros(Int, nsteps+1)

    # Correlation (only at measured steps)
    meas_steps = [i for i in 0:nsteps if i % corr_every == 0]
    nmeas      = length(meas_steps)
    meas_times = times[meas_steps .+ 1]
    X_t  = zeros(Float64, nmeas, L)
    Y_t  = zeros(Float64, nmeas, L)
    Z_t  = zeros(Float64, nmeas, L)
    XX_t = zeros(Float64, nmeas, L, L)
    YY_t = zeros(Float64, nmeas, L, L)
    ZZ_t = zeros(Float64, nmeas, L, L)

    # Build instantaneous H(t)
    function H_of_t(t)
        return drive_t1(t)*H0_mpo + g*H3_mpo
    end

    # Measure correlations into row k of the arrays
    function measure_corrs!(k, psi)
        X_t[k, :]     = real.(expect(psi, "X"))
        Y_t[k, :]     = real.(expect(psi, "Y"))
        Z_t[k, :]     = real.(expect(psi, "Z"))
        XX_t[k, :, :] = real.(correlation_matrix(psi, "X", "X"))
        YY_t[k, :, :] = real.(correlation_matrix(psi, "Y", "Y"))
        ZZ_t[k, :, :] = real.(correlation_matrix(psi, "Z", "Z"))
    end

    # Initial state measurements
    psi_t = copy(psi0)
    H0_inst = H_of_t(0.0)
    loschmidt[1]  = 1.0
    autoc_x[1]    = inner(phi_x, apply(Ox, psi_t))
    autoc_y[1]    = inner(phi_y, apply(Oy, psi_t))
    autoc_z[1]    = inner(phi_z, apply(Oz, psi_t))
    energy_t[1]   = real(inner(psi_t', H0_inst, psi_t))
    energy_0[1]   = real(inner(psi_t', H_static, psi_t))
    maxdims[1]    = maxlinkdim(psi_t)
    meas_idx = 1
    if 0 in meas_steps
        measure_corrs!(meas_idx, psi_t)
        meas_idx += 1
    end

    # Time evolution loop
    for i in 1:nsteps
        t = times[i+1]
        println("Step $i / $nsteps  (t = $(round(t, digits=4)), t/T = $(round(t/T, digits=3)))")

        gates = gate_sets[mod1(i, steps_per_period)]

        psi_t = apply(gates, psi_t; cutoff=cutoff, maxdim=maxdim)
        normalize!(psi_t)
        phi_x = apply(gates, phi_x; cutoff=cutoff, maxdim=maxdim)
        normalize!(phi_x)
        phi_y = apply(gates, phi_y; cutoff=cutoff, maxdim=maxdim)
        normalize!(phi_y)
        phi_z = apply(gates, phi_z; cutoff=cutoff, maxdim=maxdim)
        normalize!(phi_z)

        H_inst = H_of_t(t)
        loschmidt[i+1]  = abs2(inner(psi_t, psi0))
        autoc_x[i+1]    = inner(phi_x, apply(Ox, psi_t))
        autoc_y[i+1]    = inner(phi_y, apply(Oy, psi_t))
        autoc_z[i+1]    = inner(phi_z, apply(Oz, psi_t))
        energy_t[i+1]   = real(inner(psi_t', H_inst, psi_t))
        energy_0[i+1]   = real(inner(psi_t', H_static, psi_t))
        maxdims[i+1]    = maxlinkdim(psi_t)

        println("  |<psi(t)|psi(0)>|^2 = $(loschmidt[i+1])")
        println("  <H(t)> = $(energy_t[i+1])  <H(0)> = $(energy_0[i+1])")
        println("  maxlinkdim(psi_t) = $(maxdims[i+1])")

        if i in meas_steps
            measure_corrs!(meas_idx, psi_t)
            meas_idx += 1
        end
    end

    # Save to HDF5
    h5open(fname, "w") do fid
        fid["L"]              = L
        fid["bc"]             = bc
        fid["parity"]         = parity
        fid["t1"]             = t1
        fid["g"]              = g
        fid["omega"]          = omega
        fid["periods"]        = periods
        fid["steps_per_period"] = steps_per_period
        fid["ac_site"]        = ac_site
        fid["corr_every"]     = corr_every
        fid["E0_dmrg"]        = E0

        # Time axes
        fid["times"]          = times           # shape (nsteps+1,)
        fid["meas_times"]     = meas_times       # shape (nmeas,)

        # Scalar time series
        fid["loschmidt"]      = loschmidt        # (nsteps+1,)
        fid["re_autoc_x"]     = real(autoc_x)
        fid["im_autoc_x"]     = imag(autoc_x)
        fid["re_autoc_y"]     = real(autoc_y)
        fid["im_autoc_y"]     = imag(autoc_y)
        fid["re_autoc_z"]     = real(autoc_z)
        fid["im_autoc_z"]     = imag(autoc_z)
        fid["energy_t"]       = energy_t         # <H(t)>
        fid["energy_0"]       = energy_0         # <H(t=0)>
        fid["maxdims"]        = maxdims

        # Correlation arrays have shape (nmeas, L) and (nmeas, L, L)
        # Python loader uses X_t[step, site], XX_t[step, i, j]
        fid["X_t"]            = X_t
        fid["Y_t"]            = Y_t
        fid["Z_t"]            = Z_t
        fid["XX_t"]           = XX_t
        fid["YY_t"]           = YY_t
        fid["ZZ_t"]           = ZZ_t
    end
    println("Saved to $fname")
end
