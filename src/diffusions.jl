module DiffusionAlgorithms
using SparseArrays
using ProgressMeter
using Printf
using Statistics
using Random
using DataFrames
using DataStructures
using SharedArrays
using Distributed
using CSV

import LinearAlgebra.checksquare


function normout!(A::SparseMatrixCSC{Float64,Int64})
    d = sum(A, dims = 2) # sum over rows
    # use some internal julia magic
    for i = 1:length(A.nzval)
        A.nzval[i] = A.nzval[i] / d[A.rowval[i]]
    end
    return A
end


"""
- `maxresidvol::Int` - the maximum residual volume considered, if this is negative,
then we treat it as infinite.

Returns
-------
(x::Dict{Int,Float64},r::Dict{Int,Float64},flag::Int)
"""
function weighted_ppr_push(
    A::SparseMatrixCSC{T,Int},
    seed::Int,
    alpha::Float64,
    eps::Float64,
    maxpush::Int,
    dvec::Vector{Int},
    maxresidvol::Int,
) where {T}

    colptr = A.colptr
    rowval = A.rowval
    nzval = A.nzval

    n = size(A, 1)

    x = Dict{Int,Float64}()     # Store x, r as dictionaries
    r = Dict{Int,Float64}()     # initialize residual
    Q = Queue{Int}()            # initialize queue
    npush = 0.0

    if maxresidvol <= 0
        maxresidvol = typemax(Int)
    end

    rvol = 0

    # TODO handle a generic seed
    r[seed] = 1.0
    enqueue!(Q, seed)

    pushcount = 0
    pushvol = 0

    @inbounds while length(Q) > 0 && pushcount <= maxpush
        pushcount += 1
        u = dequeue!(Q)

        du = dvec[u] # get the degree

        pushval = r[u] - 0.5*eps*du
        x[u] = get(x, u, 0.0) + (1-alpha)*pushval
        r[u] = 0.5*eps*du

        pushval = pushval*alpha

        for nzi = colptr[u]:(colptr[u+1]-1)
            pushvol += 1
            v = rowval[nzi]
            dv = dvec[v] # degree of v

            rvold = get(r, v, 0.0)
            if rvold == 0.0
                rvol += dv
            end
            rvnew = rvold + pushval*nzval[nzi]/du

            r[v] = rvnew
            if rvnew > eps*dv && rvold <= eps*dv
                #push!(Q,v)
                enqueue!(Q, v)
            end
        end

        if rvol >= maxresidvol
            return x, r, -2
        end
    end

    if pushcount > maxpush
        return x, r, -1, pushcount
    else
        return x, r, 0, pushcount
    end
end


function weighted_ppr_push_solution(
    A::SparseMatrixCSC{T,Int},
    alpha::Float64,
    seed::Int,
    eps::Float64,
) where {T}
    maxpush = round(Int, max(1.0/(eps*(1.0-alpha)), 2.0*10^9))
    dvec = sum(A, dims = 2)
    return weighted_ppr_push(A, seed, alpha, eps, maxpush, vec(dvec), 0)[1]
end


"""
Compute the set with smallest conductance 
induced by sweepcut over x.
"""
function weighted_local_sweep_cut(
    A::SparseMatrixCSC{T,Int},
    x::Dict{Int,V},
    dvec::Vector{Int},
    Gvol::Int,
) where {T,V}
    colptr = A.colptr
    rowval = A.rowval
    nzval = A.nzval

    n = size(A, 1)

    sx = sort(collect(x), by = x->x[2], rev = true)
    S = Set{Int64}()
    volS = 0.0
    cutS = 0.0
    bestcond = 1.0
    beststats = (1, 1, 1, Gvol-1)
    bestpre = 0
    for (i, p) in enumerate(sx)
        if i == n
            break
        end
        u = p[1] # get the vertex
        #volS += colptr[u+1] - colptr[u]
        volS += dvec[u]

        push!(S, u)
        for nzi = colptr[u]:(colptr[u+1]-1)
            v = rowval[nzi]
            ew = nzval[nzi]

            if v == u
                continue
            end
            if v in S
                cutS -= ew
            else
                cutS += ew
            end
        end
        if cutS/min(volS, Gvol-volS) <= bestcond
            bestcond = cutS/min(volS, Gvol-volS)
            size = length(S)
            bestpre = i
            if Gvol - volS < volS
                size = n - size
            end
            beststats = (cutS, size, volS, Gvol-volS)
        end
    end
    bestset = Set{Int64}()
    for (i, p) in enumerate(sx)
        u = p[1]
        push!(bestset, u)
        if i == bestpre
            break
        end
    end
    return bestset, bestcond, beststats
end


"""
Compute the piecewise lower bound for conductance of 
sets induced by sweepcut over x
"""
function weighted_local_sweep_cut_curve(
    A::SparseMatrixCSC{T,Int},
    x::Dict{Int,V},
    dvec::Vector{Int},
    Gvol::Int,
) where {T,V}
    colptr = A.colptr
    rowval = A.rowval
    nzval = A.nzval

    n = size(A, 1)

    sx = sort(collect(x), by = x->x[2], rev = true)
    S = Set{Int64}()
    volS = 0.0
    cutS = 0.0
    bestcond = 1.0
    beststats = (1, 1, 1, Gvol-1)
    volS_list = Vector{Int64}()
    condS_list = Vector{Float64}()
    stats_list = Vector{Tuple}()
    for p in sx
        if length(S) == n-1
            break
        end
        u = p[1] # get the vertex
        #volS += colptr[u+1] - colptr[u]
        volS += dvec[u]

        for nzi = colptr[u]:(colptr[u+1]-1)
            v = rowval[nzi]
            ew = nzval[nzi]

            if v in S
                cutS -= ew
            else
                cutS += ew
            end
        end
        push!(S, u)
        minside_vol = min(volS, Gvol - volS)
        push!(volS_list, minside_vol)
        push!(condS_list, cutS / minside_vol)
        size = length(S)
        if volS > Gvol - volS
            size = n - size
        end
        push!(stats_list, (cutS, size, volS, Gvol - volS))
    end
    ord = sortperm(volS_list)
    volS_curve = Vector{Int64}()
    condS_curve = Vector{Float64}()
    stats_curve = Vector{Tuple}()
    for i = length(ord):-1:1
        j = ord[i]
        if length(volS_curve) == 0 || (condS_list[j] < condS_curve[end])
            push!(volS_curve, volS_list[j])
            push!(condS_curve, condS_list[j])
            push!(stats_curve, stats_list[j])
        end
    end
    return volS_curve, condS_curve, stats_curve
end


function weighted_degree_normalized_sweep_cut!(
    A::SparseMatrixCSC{T,Int},
    x::Dict{Int,V},
    dvec::Array{Int},
    Gvol::Int,
) where {T,V}
    for u in keys(x)
        x[u] = x[u]/dvec[u]
    end

    return weighted_local_sweep_cut(A, x, dvec, Gvol)
end


function weighted_degree_normalized_sweep_cut_curve!(
    A::SparseMatrixCSC{T,Int},
    x::Dict{Int,V},
    dvec::Array{Int},
    Gvol::Int,
) where {T,V}
    for u in keys(x)
        x[u] = x[u]/dvec[u]
    end

    return weighted_local_sweep_cut_curve(A, x, dvec, Gvol)
end


function seed_set_grow_one(
    A::SparseMatrixCSC{T,Int},
    seed::Int,
    alpha::Float64,
    eps::Float64,
) where {T}
    maxpush = round(Int, max(1.0/(eps*(1.0-alpha)), 2.0*10^9))
    dvec = vec(sum(A, dims = 2))
    @assert eltype(dvec)==Int
    Gvol = sum(dvec)
    ppr = weighted_ppr_push(A, seed, alpha, eps, maxpush, dvec, 0)[1]
    return weighted_degree_normalized_sweep_cut!(A, ppr, dvec, Gvol)
end


struct setstats
    size::Int
    volume_seed::Int #
    cut::Int
    seed::Int
    volume_other::Int # the volume on the other side
    cond::Float64
    support::Int # total vertices evaluated
    work::Int # total number of edges "pushed"
end


function setstats()
    return setstats(0, 0, 0, 0, 0, 1.0, 0, 0)
end


"""
`weighted_ncp`
--------------

Compute the NCP of a weighted graph. This runs a standard
personalized PageRank-based NCP except on a graph where
the edges are weighted. The weighted edges must be integers
at the moment.

Usage
-----
weighted_ncp(A)
weighted_ncp(A; ...)
- `alpha::Float64` default value 0.99, the pagerank value of alpha
- `maxcond::Float64` ignore clusters with conductance larger than maxcond
- `minsize::Int` the minimum size cluster to consider
- `maxvol::Float64' the maximum volume (if maxvol <= 1, then it's a ratio, if
    maxvol > 1, then it's a total number of edges)
- `maxvisits::Int` don't start a new cluster from a node if it's
  already been in k clusters
Returns
-------
The function returns an Array of setstats
"""


function parallel_weighted_ncp(
    A::SparseMatrixCSC{Int,Int};
    alpha::Float64 = 0.99,
    maxcond::Float64 = 1.0,
    minsize::Int = 5,
    maxvol::Float64 = 1.0,
    maxvisits::Int = 10,
)

    #n = size(A,1)
    n = checksquare(A)
    visits = SharedArray{Int}((n,))

    for i = 1:n
        visits[i] = 0
    end

    eps=1.e-5

    l = ReentrantLock()

    dvec = vec(sum(A, dims = 2)) # weighted degree vectors
    @assert eltype(dvec)==Int
    Gvol = sum(dvec)

    vset = randperm(n)
    sz = 10000

    ncpdata = @showprogress pmap(1:sz) do i
        begin
            # check if the node has already been handled
            v = vset[i]
            if visits[v] >= maxvisits
                return setstats()
            end

            seed = v

            maxpush = round(Int, max(1.0/(eps*(1.0-alpha)), 2.0*10^9))
            ppr =
                weighted_ppr_push(A, seed, alpha, eps, maxpush, dvec, 0)[1]
            bestset, bestcond, setnums =
                weighted_degree_normalized_sweep_cut!(A, ppr, dvec, Gvol)

            volseed = setnums[2]
            volother = Gvol-volseed
            if seed ∉ bestset
                volother = volseed
                volseed = Gvol - volseed
            end

            lock(l)
            for u in bestset
                visits[u] += 1
            end
            unlock(l)

            # todo add support/work
            return setstats(
                length(bestset),
                volseed,
                setnums[1],
                seed,
                volother,
                bestcond,
                0,
                0,
            )
        end
    end
    return ncpdata
end


function create_ncpdata()
    return DataFrame(
        seed = Int64[],
        eps = Float64[],
        size = Int64[],
        cond = Float64[],
        cut = Float64[],
        volume_seed = Int64[],
        volume_other = Int64[],
        ComputeTime = Float64[],
    )
end


function ncp_entry(n, seed, eps, bestset, bestcond, setnums, dt)
    setsize = setnums[2]
    volseed = setnums[3]
    volother = setnums[4]

    return [seed, eps, setsize, bestcond, setnums[1], volseed, volother, dt]
end


function serial_weighted_ncp(
    A::SparseMatrixCSC{Int,Int};
    alpha::Float64 = 0.99,
    maxcond::Float64 = 1.0,
    minsize::Int = 5,
    maxvol::Float64 = 1.0,
    maxvisits::Int = 10,
    maxlarge::Int = 10,
    largethresh::Float64 = 0.8,
    type::Int = 1,
    timelimit::Float64 = 3600.0,
)

    n = checksquare(A)

    epsseq = [2, 5, 10, 25, 50, 100]
    epsvals = [0.01; 1.0 ./ (100*epsseq); 1.0 ./ (10000*epsseq); 1.0e-7; 1.0e-8]
    epsbig = 1.0e-4

    dvec = vec(sum(A, dims = 2)) # weighted degree vectors
    @assert eltype(dvec)==Int
    Gvol = sum(dvec)
    meand = mean(dvec)

    ncpdata = create_ncpdata()

    lasteps = false

    for eps in epsvals
        println(eps)
        # reset visited
        visits = zeros(n)
        maxpush = round(Int, min(1.0/(eps*(1.0-alpha)), 2.0*10^9))
        if maxpush >= 100*Gvol
            # these are getting too big, let's stop here
            lasteps = true
            break
        end
        vset = randperm(n)
        @assert n > 10 "graph size is too small(<10)"
        dt = 0
        if timelimit > 0.0
            # estimate exec time
            for i = 1:10
                seed = vset[i]
                if type == 2
                    dt += @elapsed begin
                        ppr = weighted_ppr_push(
                            A,
                            seed,
                            alpha,
                            eps,
                            maxpush,
                            dvec,
                            0,
                        )[1]
                        bestset, bestcond, setnums =
                            weighted_degree_normalized_sweep_cut_curve!(
                                A,
                                ppr,
                                dvec,
                                Gvol,
                            )
                    end
                else
                    dt += @elapsed begin
                        ppr = weighted_ppr_push(
                            A,
                            seed,
                            alpha,
                            eps,
                            maxpush,
                            dvec,
                            0,
                        )[1]
                        bestset, bestcond, setnums =
                            weighted_degree_normalized_sweep_cut!(
                                A,
                                ppr,
                                dvec,
                                Gvol,
                            )
                    end
                end
            end
            dtpersample = dt / 10
            nsamples = min(n, Int(floor(timelimit / dtpersample)))
            vset = vset[1:nsamples]
        else
            if maxpush*meand^2 > Gvol
                vset = vset[1:min(ceil(Int64, 0.1*n), 10^5)]
                if maxpush*meand > 50 * Gvol
                    vset = vset[1:min(ceil(Int64, 0.1*length(vset)))]
                end
            end
        end
        nlarge = 0
        for v in vset # randomize the order
            seed = v
            if visits[v] >= maxvisits
                continue
            end

            dt = @elapsed ppr =
                weighted_ppr_push(A, seed, alpha, eps, maxpush, dvec, 0)[1]
            #@show (eps, v, dt)
            if type == 1
                # for each seeded PageRank, we take the set with smallest conductance
                # and we don't visit the same vertex too many times
                dt += @elapsed bestset, bestcond, setnums =
                    weighted_degree_normalized_sweep_cut!(A, ppr, dvec, Gvol)

                if length(bestset) <= minsize
                    continue
                end
                push!(
                    ncpdata,
                    ncp_entry(n, seed, eps, bestset, bestcond, setnums, dt),
                )

                if length(bestset) > largethresh*n
                    nlarge += 1
                    if nlarge >= maxlarge
                        break
                    end
                end
                if eps < epsbig
                    for u in bestset
                        visits[u] += 1
                    end
                end
                #println(@sprintf("%8.3e  %7i  %8i  %8i  %5.3f  %6.2f",
                #    eps, ncpdata[!,:size][end], ncpdata[!,:volume_seed][end], ncpdata[!,:volume_other][end], bestcond, dt))
            elseif type == 2
                # for each seeded PageRank, we take a list of sets with small conductance
                dt += @elapsed volS_curve, condS_curve, stats_curve =
                    weighted_degree_normalized_sweep_cut_curve!(
                        A,
                        ppr,
                        dvec,
                        Gvol,
                    )
                for i = 1:length(volS_curve)
                    if volS_curve[i] < minsize
                        continue
                    end
                    push!(
                        ncpdata,
                        [
                            seed,
                            eps,
                            stats_curve[i][2],
                            condS_curve[i],
                            stats_curve[i][1],
                            stats_curve[i][3],
                            stats_curve[i][4],
                            dt,
                        ],
                    )
                end
            else
                # same with type 1 but without visiting constraint
                dt += @elapsed bestset, bestcond, setnums =
                    weighted_degree_normalized_sweep_cut!(A, ppr, dvec, Gvol)

                push!(
                    ncpdata,
                    ncp_entry(n, seed, eps, bestset, bestcond, setnums, dt),
                )

            end

        end

        if lasteps
            break
        end
    end
    return ncpdata
end


function test_delocalization(
    A::SparseMatrixCSC,
    alpha::Float64,
    eps::Float64,
    dvec::Vector{Int},
    Gvol::Int,
)
    maxpush = round(Int, max(1.0/(eps*(1.0-alpha)), 2.0*10^9))
    ntrials = 5
    for i = 1:ntrials
        seed = rand(1:size(A, 1))
        ppr, r, flag = weighted_ppr_push(
            A,
            seed,
            alpha,
            eps,
            maxpush,
            dvec,
            round(Int, Gvol*0.8),
        )
        #@show eps, length(ppr), length(r)
        if flag == -2
            return true
        end

        bestset, bestcond, setnums =
            weighted_degree_normalized_sweep_cut!(A, ppr, dvec, Gvol)
        #@show eps, setnums, length(bestset)
        if setnums[3] > setnums[4]
            return true
        end
    end

    return false
end



function estimate_delocalization(A::SparseMatrixCSC{Int}, alpha::Float64)

    dvec = vec(sum(A, dims = 2))
    Gvol = sum(dvec)

    epscur = 1.0/((1-alpha)*Gvol)
    @assert epscur > eps(1.0)

    rval = test_delocalization(A, alpha, epscur, dvec, Gvol)

    if rval
        epsrange = (epscur, 0.1)
        # already delocalized, increase eps
        for i = 1:20
            epscur = epscur*2.0
            if test_delocalization(A, alpha, epscur, dvec, Gvol)
                epsrange = (epscur/2.0, epscur)
                break
            end
        end
    else
        # decrease eps to delocalize
        epsrange = (eps(1.0), epscur)
        for i = 1:20
            epscur /= 2.0
            if test_delocalization(A, alpha, epscur, dvec, Gvol)
                epsrange = (epscur, epscur*2.0)
                break
            end
        end
    end

    # run 5 steps of bisection
    for i = 1:5
        mid = epsrange[1]*0.5 + epsrange[2]*0.5
        if test_delocalization(A, alpha, mid, dvec, Gvol)
            # midpoint is delocalized, use upper points
            epsrange = (mid, epsrange[2])
        else
            # could get bigger...
            epsrange = (epsrange[1], mid)
        end
    end

    return epsrange
end


struct NCPProblem
    """ The matrix to run on. """
    A::SparseMatrixCSC{Int,Int}
    dvec::Vector{Int}
    Gvol::Int

    """ The set of alpha values to use. """
    alphas::Vector{Float64}

    """ The largest conductance to consider reporting. """
    maxcond::Float64
    minsize::Int

    """ The maximum volume to consider in a diffusion. """
    maxvol::Float64

    """ The range of eps values to consider. """
    epsrange::Tuple{Float64,Float64}
end

"""
test
"""
function ncp_problem(A::SparseMatrixCSC{Int,Int})
    epsrange = estimate_delocalization(A, 0.99) # epsrange[1] is delocalized, epsrange[2] isn't
    dvec = vec(sum(A, dims = 2))
    Gvol = sum(dvec)

    return NCPProblem(A, dvec, Gvol, [0.99], 1.0, 5, 1.0, epsrange)
end

randlog(a, b) = exp(rand(Float64)*(log(b)-log(a))+log(a))

function random_sample_ncp(p::NCPProblem, N::Int)
    n = size(p.A, 1)
    N = min(N, n)
    verts = randperm(n)
    ncpdata = create_ncpdata()

    for i = 1:N
        seed = verts[i]
        epsmax = 1/(p.dvec[seed]+1) # just go beyond trivial work
        eps = randlog(p.epsrange[1], epsmax)
        for alpha in p.alphas
            maxpush = round(Int, max(1.0/(eps*(1.0-alpha)), 2.0*10^9))
            dt = @elapsed ppr = weighted_ppr_push(
                p.A,
                seed,
                alpha,
                eps,
                maxpush,
                p.dvec,
                0,
            )[1]
            dt += @elapsed bestset, bestcond, setnums =
                weighted_degree_normalized_sweep_cut!(p.A, ppr, p.dvec, p.Gvol)

            push!(
                ncpdata,
                ncp_entry(n, seed, eps, bestset, bestcond, setnums, dt),
            )

            println(
                @sprintf(
                    "%8.3e  %7i  %8i  %8i  %5.3f  %6.2f",
                    eps,
                    ncpdata[:size][end],
                    ncpdata[:volume_seed][end],
                    ncpdata[:volume_other][end],
                    bestcond,
                    dt
                )
            )

        end
    end

    return ncpdata
end


function make_ACL_jobs(n, jobs, nworkers)
    vsets = [Int64[] for i = 1:nworkers]
    vperm = randperm(n)
    for i = 1:n
        push!(vsets[(i%nworkers)+1], vperm[i])
    end
    for i = 1:nworkers
        put!(jobs, vsets[i])
    end
    for _ = 1:nworkers
        put!(jobs, [])
    end
end


function solve_ACL_one_worker(
    A::SparseMatrixCSC{Int,Int},
    vset::Vector{Int},
    alpha = 0.99,
    epsvals = [],
    setsizes = [],
    type = 2,
    minvol = 10,
)
    n = checksquare(A)

    if length(epsvals) == 0 || length(setsizes) === 0
        epsseq = [2, 5, 10, 25, 50, 100]
        epsvals =
            [0.01; 1.0 ./ (100*epsseq); 1.0 ./ (10000*epsseq); 1.0e-7; 1e-8]
        setsizes = length(vset) * ones(Int, length(epsvals))
        for (i, eps) in enumerate(epsvals)
            if eps >= 1e-4
                setsizes[i] = min(setsizes[i], 10^5)
            elseif eps >= 1e-6
                setsizes[i] = min(setsizes[i], 5*10^3)
            else
                setsizes[i] = min(setsizes[i], 100)
            end
        end
    end

    dvec = vec(sum(A, dims = 2)) # weighted degree vectors
    @assert eltype(dvec)==Int
    Gvol = sum(dvec)

    ncpdata = create_ncpdata()

    lasteps = false

    for (i, eps) in enumerate(epsvals)
        println(eps)
        # reset visited
        maxpush = round(Int, min(1.0/(eps*(1.0-alpha)), 2.0*10^9))
        if maxpush >= 100000 * Gvol
            # these are getting too big, let's stop here
            lasteps = true
            break
        end
        # randomize the order
        seedset = randperm(length(vset))
        seeds = vset[seedset[1:setsizes[i]]]
        @assert length(seeds) == setsizes[i]
        totaldt = 0
        @show eps, length(seeds)
        for v in seeds
            seed = v

            dt = @elapsed ppr =
                weighted_ppr_push(A, seed, alpha, eps, maxpush, dvec, 0)[1]
            if type == 1
                # take one set per seeded PageRank
                dt += @elapsed begin
                    bestset, bestcond, setnums =
                        weighted_degree_normalized_sweep_cut!(
                            A,
                            ppr,
                            dvec,
                            Gvol,
                        )
                    volseed = setnums[3]
                    volother = setnums[4]
                    if min(volseed, volother) < minvol
                        continue
                    end
                    push!(
                        ncpdata,
                        ncp_entry(n, seed, eps, bestset, bestcond, setnums, dt),
                    )
                end
            else
                # take multiple sets per seeded PageRank
                dt += @elapsed begin
                    volS_curve, condS_curve, stats_curve =
                        weighted_degree_normalized_sweep_cut_curve!(
                            A,
                            ppr,
                            dvec,
                            Gvol,
                        )
                    for i = 1:length(volS_curve)
                        volseed = stats_curve[i][3]
                        volother = stats_curve[i][4]
                        @assert min(volseed, volother) == volS_curve[i]
                        if volS_curve[i] < minvol
                            continue
                        end
                        push!(
                            ncpdata,
                            [
                                seed,
                                eps,
                                stats_curve[i][2],
                                condS_curve[i],
                                stats_curve[i][1],
                                stats_curve[i][3],
                                stats_curve[i][4],
                                dt,
                            ],
                        )
                    end
                end
            end
            totaldt += dt
        end
        @show eps, totaldt / length(vset)
        if lasteps
            break
        end
    end
    return ncpdata
end

function do_ACL_jobs(
    jobs,
    results,
    A::SparseMatrixCSC{Int,Int},
    alpha::Float64 = 0.99,
    epsvals = [],
    setsizes = [],
    type = 2,
    minvol = 10,
)
    while (true)
        vset = take!(jobs)
        if length(vset) == 0
            break
        end
        ncpdata = solve_ACL_one_worker(
            A,
            vset,
            alpha,
            epsvals,
            setsizes,
            type,
            minvol,
        )
        put!(results, ncpdata)
    end
end


function bulk_local_ACL(
    A::SparseMatrixCSC{Int,Int};
    alpha::Float64 = 0.99,
    epsvals = [],
    setsizes = [],
    filename::String = "livejournal.csv",
    type = 2,
    minvol = 10,
)
    n = size(A, 1)
    nworkers = length(workers())
    jobs = RemoteChannel(()->Channel{Vector{Int64}}(nworkers + nworkers))
    results = RemoteChannel(()->Channel{DataFrame}(nworkers))
    make_ACL_jobs(n, jobs, nworkers)
    for p in workers()
        remote_do(
            do_ACL_jobs,
            p,
            jobs,
            results,
            A,
            alpha,
            epsvals,
            setsizes,
            type,
            minvol,
        )
    end
    all_ncp = create_ncpdata()
    while nworkers > 0
        ncp = take!(results)
        nworkers -= 1
        all_ncp = vcat(all_ncp, ncp)
    end
    close(jobs)
    close(results)
    CSV.write(filename, all_ncp)
    return all_ncp
end

#=




function serial_weighted_ncp(A::SparseMatrixCSC{Int,Int};
        alpha::Float64=0.99,
        maxcond::Float64=1.0,
        minsize::Int=5,
        maxvol::Float64=1.,
        maxvisits::Int=10,
        maxlarge::Int=10,
        largethresh::Float64=0.8)
end

=#
function test_progress() end

end # end module
