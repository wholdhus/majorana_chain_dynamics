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

function local_H(t, g, s1, s2, s3; signs=[1.0, 1.0, 1.0, 1.0])
    X_part = -signs[1]*op("Z", s1) * op("Id", s2) * op("Id", s3)
    H = 2*t*X_part
    if signs[2] != 0.0
        ZZ_part = -signs[2]*op("X", s1) * op("X", s2) * op("Id", s3)
        H += 2*t*ZZ_part
    end
    if signs[3] != 0
        H += g*signs[3]*op("Z", s1) * op("X", s2) * op("X", s3) 
    end
    if signs[4] != 0
        H += g*signs[4]*op("X", s1) * op("X", s2) * op("Z", s3)
    end
    return H
end

function dmrg_static(L, bc, t, g, parity, maxdim;
                  cutoff=1e-9,
                  nsweeps=20,
                  dmrg_tol=1e-8,
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
    H = t*H0 + g*H3
    
    state = ["Up" for n=1:L]
    if parity == -1
        println("Changing to odd parity")
        state[1] = "Down"
    end
    
    psi0 = random_mps(sites, state, linkdims=4)
    observer = DMRGObserver(energy_tol=dmrg_tol)
    md = Int(round(maxdim/6))
    maxdim = [md, 2*md, 3*md, 4*md, 5*md, maxdim]
    E1, psi1 = dmrg(H, psi0; nsweeps, maxdim, cutoff, observer, outputlevel=1, noise, eigsolve_krylovdim)
    E2, psi2 = dmrg(H, [psi1], psi0; nsweeps, maxdim, cutoff, observer, outputlevel=1, noise, eigsolve_krylovdim)
    gap = abs.(E1 - E2)
    
    # Energy
    println("\nDMRG Finished!")
    println("E1 = $(E1), psi1 maxdim = $(maxlinkdim(psi1))")
    println("E2 = $(E2), psi2 maxdim = $(maxlinkdim(psi2))")
    println("Energy Gap = $(gap)")

    # X-X Correlations
    XXC = Vector{Matrix{ComplexF64}}(undef, 2)
    XXCC = Vector{Matrix{Float64}}(undef, 2)
    XXC[1] = correlation_matrix(psi1, "X", "X")
    XXC[2] = correlation_matrix(psi2, "X", "X")
    X_exp1 = [real(expect(psi1, "X"; sites=j)) for j in 1:L]
    X_exp2 = [real(expect(psi2, "X"; sites=j)) for j in 1:L]
    XXCC[1] = real.(XXC[1] .- X_exp1 * X_exp1')
    XXCC[2] = real.(XXC[2] .- X_exp2 * X_exp2')

    # Y-Y Correlations
    YYC = Vector{Matrix{ComplexF64}}(undef, 2)
    YYCC = Vector{Matrix{Float64}}(undef, 2)
    YYC[1] = correlation_matrix(psi1, "Y", "Y")
    YYC[2] = correlation_matrix(psi2, "Y", "Y")
    Y_exp1 = [real(expect(psi1, "Y"; sites=j)) for j in 1:L]
    Y_exp2 = [real(expect(psi2, "Y"; sites=j)) for j in 1:L]
    YYCC[1] = real.(YYC[1] .- Y_exp1 * Y_exp1')
    YYCC[2] = real.(YYC[2] .- Y_exp2 * Y_exp2')

    # Z-Z Correlations
    ZZC = Vector{Matrix{ComplexF64}}(undef, 2)
    ZZCC = Vector{Matrix{Float64}}(undef, 2)
    ZZC[1] = correlation_matrix(psi1, "Z", "Z")
    ZZC[2] = correlation_matrix(psi2, "Z", "Z")
    Z_exp1 = [real(expect(psi1, "Z"; sites=j)) for j in 1:L]
    Z_exp2 = [real(expect(psi2, "Z"; sites=j)) for j in 1:L]
    ZZCC[1] = real.(ZZC[1] .- Z_exp1 * Z_exp1')
    ZZCC[2] = real.(ZZC[2] .- Z_exp2 * Z_exp2')

    output = DataFrame(
        "state" => [1, 2],
        "energy" => [E1, E2],
        "gap" => [gap, gap],
        "XXC" => [vec(XXC[1]), vec(XXC[2])],
        "XXCC" => [vec(XXCC[1]), vec(XXCC[2])],
        "YYC" => [vec(YYC[1]), vec(YYC[2])],
        "YYCC" => [vec(YYCC[1]), vec(YYCC[2])],
        "ZZC" => [vec(ZZC[1]), vec(ZZC[2])],
        "ZZCC" => [vec(ZZCC[1]), vec(ZZCC[2])])
    fname = "L$(L)_$(bc)_t$(t)_g$(g)_parity$(parity)_maxdim$(maxdim[end]).csv"
    CSV.write(fname, output)
    println("\nWrote data to $fname")

    return E1, E2, gap, XXC, XXCC, YYC, YYCC, ZZC, ZZCC
end