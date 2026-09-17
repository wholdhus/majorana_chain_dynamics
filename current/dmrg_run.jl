include("dmrg_sim.jl")

using YAML

function main()
    params_file = ARGS[1]
    println("Reading parameters from $params_file")
    params = YAML.load_file(params_file)

    L       = params["L"]
    bc      = params["bc"]
    t       = params["t"]
    g       = params["g"]
    parity  = params["parity"]
    nstates = params["nstates"]
    fname   = params["fname"]

    dmrg_params = get(params, "dmrg_params", Dict())

    m                  = get(dmrg_params, "m", 100)
    cutoff             = get(dmrg_params, "cutoff", 1e-9)
    nsweeps            = get(dmrg_params, "nsweeps", 20)
    dmrg_tol           = get(dmrg_params, "dmrg_tol", 1e-8)
    eigsolve_krylovdim = get(dmrg_params, "eigsolve_krylovdim", 10)
    noise              = get(dmrg_params, "noise", [0.0])
    weight             = get(dmrg_params, "weight", 100)

    println("Running DMRG with:")
    println("  L = $L, bc = $bc, t = $t, g = $g, parity = $parity, m = $m, nstates = $nstates, weight = $weight")
    println("  fname = $fname")

    get_energies_measure_save(L, bc, parity, t, g, nstates, fname;
                              m=m,
                              cutoff=cutoff,
                              nsweeps=nsweeps,
                              dmrg_tol=dmrg_tol,
                              eigsolve_krylovdim=eigsolve_krylovdim,
                              noise=noise,
                              weight=weight)
end

main()