using ArgParse


function parse_lrsdp_cmd()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--Ktol"
            help = "Tolerance for the KKT conditions."
            arg_type = Float64
            default = 1e-5
        "--Ptol"
            help = "Tolerance for the Primal Feasibility."
            arg_type = Float64
            default = 1e-5
        "--init_type", "-i"
            help = "How we initialize the solution."
            arg_type = Int64
            default = 2 
        "--filename"
            help = "Which graph to run experiments on."
            arg_type = String
            default = "deezer"
        "--resfolder"
            help = "Location where we save resutls"
            arg_type = String
        "--kvals"
            help = "ranks of low-rank SDP matrices."
            arg_type = Int
            nargs = '+'
        "--muvals"
            help = "Which mus we run our low-rank SDP."
            arg_type = Float64 
            nargs = '+'
        "--maxfun"
            help = "Maximum number of function evaluations."
            arg_type = Int64
            default = 150000
    end
    return parse_args(s)
end


function parse_ncp_cmd()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--filename"
            help = "Which graph to run experiments on."
            arg_type = String
            default = "deezer"
        "--resfolder"
            help = "Location where we save resutls"
            arg_type = String
        "--epsvals"
            help = "Which set of eps we run seeded PageRank(ACL)."
            arg_type = Float64 
            nargs = '+'
        "--setsizes"
            help = "For each eps, how many seeds we run on each core."
            arg_type = Int
            nargs = '+'
    end
    return parse_args(s)
end



