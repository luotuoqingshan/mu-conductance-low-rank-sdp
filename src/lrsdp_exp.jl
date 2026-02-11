using Distributed
using TOML
include("args.jl")
include("read_data.jl")
@everywhere include("lrsdp.jl")

args = parse_lrsdp_cmd()

filename = args["filename"]

A = load_graph(filename)

saved_config = TOML.parsefile(pwd()*"/configs/lrsdp_config.toml")

Ktol, Ptol, maxfun, init_type, kvals, muvals, = args["Ktol"],
args["Ptol"],
args["maxfun"],
args["init_type"],
args["kvals"],
args["muvals"]

if length(kvals) == 0
    @assert filename in keys(saved_config) "Please provide a list of kvals"
    kvals = saved_config[filename]["kvals"]
end

if length(muvals) == 0
    @assert filename in keys(saved_config) "Please provide a list of muvals"
    muvals = saved_config[filename]["muvals"]
end

@show Ktol, Ptol, maxfun, init_type, kvals, muvals

if args["resfolder"] === nothing
    resfolder = pwd()*"/../data/output/"
else
    resfolder = args["resfolder"]
end

mus = Float64[]
ks = Int[]

for k in kvals
    for mu in muvals
        push!(mus, mu)
        push!(ks, k)
    end
end

mkpath(resfolder*filename*"/")
muconductance.bulk_eval_network_profile(
    A,
    mus,
    ks,
    init_type,
    Ktol,
    Ptol,
    maxfun,
    filename,
    resfolder*filename*"/",
)
