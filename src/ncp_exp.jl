using Distributed
include("args.jl")
include("read_data.jl")
@everywhere include("diffusions.jl")

args = parse_ncp_cmd()

filename, epsvals, setsizes =
    args["filename"], args["epsvals"], args["setsizes"]

if args["resfolder"] === nothing
    resfolder = pwd()*"/../data/output/"
else
    resfolder = args["resfolder"]
end
savepath = resfolder * filename * "/" * filename * "_ncp.csv"

@show savepath

A = load_graph(filename)

mkpath(resfolder*filename*"/")
ncp = DiffusionAlgorithms.bulk_local_ACL(
    A;
    alpha = 0.99,
    epsvals = epsvals,
    setsizes = setsizes,
    filename = savepath,
)

#Example for visualizing results

#include("ncpplots.jl")
#using Plots
#diffusion_ncpplot(ncp)
#savefig("$filename_ncp.pdf")
