include("tebd_sim.jl")

using YAML

function main()
    params_file = ARGS[1]
    println("Reading parameters from $params_file")
    params = YAML.load_file(params_file)

    L        = params["L"]
    bc       = params["bc"]
    parity   = params["parity"]
    t1       = params["t1"]
    g        = params["g"]
    omega    = params["omega"]
    periods  = params["periods"]
    ac_site  = params["ac_site"]
    fname    = params["fname"]

    tebd_params = get(params, "tebd_params", Dict())

    steps_per_period   = get(tebd_params, "steps_per_period", 10)
    corr_every         = get(tebd_params, "corr_every", 1)
    maxdim             = get(tebd_params, "maxdim", 100)
    cutoff             = get(tebd_params, "cutoff", 1e-9)
    nsweeps            = get(tebd_params, "nsweeps", 20)
    dmrg_tol           = get(tebd_params, "dmrg_tol", 1e-8)
    eigsolve_krylovdim = get(tebd_params, "eigsolve_krylovdim", 10)
    noise              = get(tebd_params, "noise", [0.0])

    println("Running TEBD with:")
    println("  L=$L, bc=$bc, parity=$parity, t1=$t1, g=$g, omega=$omega")
    println("  periods=$periods, steps_per_period=$steps_per_period, ac_site=$ac_site")
    println("  maxdim=$maxdim, corr_every=$corr_every")
    println("  fname=$fname")

    tebd_evolve(L, bc, parity, t1, g, omega, periods, ac_site, fname;
                steps_per_period=steps_per_period,
                corr_every=corr_every,
                cutoff=cutoff,
                nsweeps=nsweeps,
                dmrg_tol=dmrg_tol,
                maxdim=maxdim,
                eigsolve_krylovdim=eigsolve_krylovdim,
                noise=noise)
end

main()
