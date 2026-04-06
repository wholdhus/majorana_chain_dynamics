include("static.jl")

using YAML

function main()
    params_file = ARGS[1]
    println("Reading parameters from $params_file")
    params = YAML.load_file(params_file)

    # Parameters
    L       = params["L"]
    bc      = params["bc"]
    t       = params["t"]
    g       = params["g"]
    parity  = params["parity"]
    maxdim  = params["maxdim"]

    # Parameters with default values
    cutoff  = get(params, "cutoff", 1e-9)
    nsweeps = get(params, "nsweeps", 20)
    dmrg_tol = get(params, "dmrg_tol", 1e-8)
    eigsolve_krylovdim = get(params, "eigsolve_krylovdim", 10)
    noise   = get(params, "noise", [0.0])

    println("Running DMRG with:")
    println("  L=$L bc=$bc t=$t g=$g parity=$parity maxdim=$maxdim")

    dmrg_static(L, bc, t, g, parity, maxdim;
                cutoff=cutoff,
                nsweeps=nsweeps,
                dmrg_tol=dmrg_tol,
                eigsolve_krylovdim=eigsolve_krylovdim,
                noise=noise)
end

main()
