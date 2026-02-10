"""
Solving the low-rank SDP program using Augmented Lagrangian Method

min  Tr(Y'LY)
s.t. Tr(Y'DY) = 1
     Y'd = 0
     Diag(Y'Y) + s = (1-mu)/mu * e/Vol(G)
     0 <= s <= (1 - 2mu) / mu(1 - mu) * e / Vol(G)
"""
module muconductance
using Random
using SparseArrays
using LinearAlgebra

using MAT
using FiniteDiff
using MatrixNetworks
using LBFGSB
using GenericArpack
using Distributed
using PrettyTables

"""
Augmented Lagrangian for the low-rank SDP

L_A = Tr(Y'LY) - lambda*(Tr(Y'DY) - 1) + sigma/2 * (Tr(Y'DY) - 1)^2 \\
      - beta' * (d'YY'd - 0) + sigma/2 * (d'YY'd - 0)'*(d'YY'd - 0) \\
      - gamma * (Diag(Y'Y) + s - (1-mu)/mu * e/Vol(G)) \\
      + sigma/2 * (Diag(Y'Y) + s - (1-mu)/mu * e/Vol(G))'
      * (Diag(Y'Y) + s - (1-mu)/mu * e/Vol(G)) 

Parameters  name           shape         description
            YS = [Y, s]    nx(k + 1)     variables to optimize
            L              nxn           Laplacian matrix
            d              n             degree vector 
            ubsqr          n             square of upper bound \\sqrt{(1 - mu)/mu* e/Vol(G)}
            sigma          1             penalty parameter
            lam            n + k + 1     tuple of Lagrangian multipliers 
            n, k           1, 1          shape
"""
function f(
    YS::Vector{Float64},
    L::SparseMatrixCSC{Int64},
    d::Vector{Int64},
    lam::Vector{Float64},
    ubsqr::Vector{Float64},
    sigma::Float64,
    n::Int64,
    k::Int64,
)
    Y = reshape(@view(YS[1:n*k]), n, k)
    s = @view(YS[n*k+1:n*(k+1)])
    v0 = sum(Y .* (L * Y))
    v1 = sum(Y .* (Y .* d))
    v2 = Y' * d
    v2 = v2' * v2
    v3 = Y .^ 2 * ones(k) + s - ubsqr
    return (
        v0 - lam[1] * (v1 - 1) + sigma / 2 * (v1 - 1)^2 - lam[2] * v2 +
        sigma / 2 * v2 * v2 - lam[3:n+2]' * v3 + sigma / 2 * v3' * v3
    )
end


"""
Gradient for Augmented Lagrangian

\\partial L_A / \\partial Y 
= 2(LY) - 2*(lambda - sigma*(Tr(Y'DY) - 1))* DY
    - 2d'dY * (beta - sigma * (d'YY'd)) - 
    2 * (gamma - sigma * (Diag(Y'Y) + s - (1-mu)/mu * e/Vol(G)) .* Y

\\partial L_A / \\partial s
= -gamma + sigma * (Diag(Y'Y) + s - (1-mu)/mu * e/Vol(G)
"""
function g!(
    G,
    YS::Vector{Float64},
    L::SparseMatrixCSC,
    d::Vector,
    lam::Vector,
    ubsqr::Vector,
    sigma::Float64,
    n::Int64,
    k::Int64,
)
    Y = reshape(@view(YS[1:n*k]), n, k)
    s = @view(YS[n*k+1:n*(k+1)])
    v1 = sum(Y .* (Y .* d))
    v2 = Y' * d
    v21 = v2' * v2
    v22 = d * v2'
    v3 = Y .^ 2 * ones(k) + s - ubsqr
    mG = reshape(@view(G[1:n*(k+1)]), n, k + 1)
    mG[:, 1:k] = (
        2 * L * Y - 2 * (lam[1] - sigma * (v1 - 1)) * (Y .* d) -
        2 * v22 * (lam[2] - sigma * v21) - 2 * ((lam[3:n+2] - sigma * v3) .* Y)
    )
    mG[:, k+1] = (-lam[3:n+2] + sigma * v3)
end


"""
Test whether the closed-form gradient we derive is correct using FiniteDiff
"""
function test_g(; n::Int64 = 10, k::Int64 = 10, ncases = 100)
    rel_errors = zeros(ncases)
    for i = 1:ncases
        A = rand((0, 1), n, n)
        A = max.(A, A')
        A = sparse(A)
        YS = randn(n * (k + 1))
        d = sum(A, dims = 1)[1, :]
        L = Diagonal(d) - A
        lam = randn(n + 2)
        ubsqr = rand(n)
        sigma = randn(1)[1]
        numeric_grad = FiniteDiff.finite_difference_jacobian(
            xs -> f(xs, L, d, lam, ubsqr, sigma, n, k),
            YS,
        )[1, :]
        analytic_grad = zeros(n * (k + 1))
        g!(analytic_grad, YS, L, d, lam, ubsqr, sigma, n, k)
        rel_error = norm(analytic_grad - numeric_grad) / norm(analytic_grad)
        rel_errors[i] = rel_error
    end
    return maximum(rel_errors), sum(rel_errors) / length(rel_errors)
end


"""
Objective Value of the low-rank SDP.
"""
function obj(YS::Vector{Float64}, L::SparseMatrixCSC{Int64}, n::Int64, k::Int64)
    Y = reshape(@view(YS[1:n*k]), n, k)
    v0 = sum(Y .* (L * Y))
    return v0
end


"""
Compute degree normalized Laplacian D^{-1/2} L D^{-1/2}
"""
function degree_normalized_Laplacian(A::SparseMatrixCSC)
    n = size(A)[1]
    d = vec(sum(A, dims = 1))
    d = sqrt.(d)
    ai, aj, av = findnz(A)
    L = sparse(ai, aj, -av ./ ((d[ai] .* d[aj])), n, n)
    L = L + sparse(2.0I, n, n)
    return L
end


"""
Primal feasibility for low-rank SDP
    Tr(Y'DY) = 1
    Y'd = 0
    Diag(YY') + s = (1-mu)/mu * e/Vol(G)
"""
function compute_primal_feasi(
    YS::Vector{Float64},
    d::Vector,
    ubsqr::Vector,
    n::Int64,
    k::Int64,
)
    Y = reshape(@view(YS[1:n*k]), n, k)
    s = @view(YS[n*k+1:n*(k+1)])
    cons = zeros(n + 2)
    cons[1] = sum(Y .* (Y .* d)) - 1
    v2 = Y' * d
    cons[2] = v2' * v2
    cons[3:n+2] = Y .^ 2 * ones(k) + s - ubsqr
    return cons
end


"""
KKT conditions for original SDP
    Primal feasibility:
        Tr(DX) = 1
        d'Xd = 0
        diag(X) + s = (1-mu)/mu * e / Vol(G) 
        0 <= s <= (1 - 2mu)/mu/(1-mu) * e /Vol(G) (naturally satisfied)
        X \\succeq 0(naturally satisfied)
    Dual feasibility: 
        L - lambda * D - beta dd' - Diagonal(gamma) \\succeq 0(Important)
    Complementary Slackness:
        Tr(X(L - lambda * D - beta dd' - Diagonal(gamma))) = 0
        s = Proj(s + gamma, 0, (1 - 2mu)/mu/(1-mu) * e / Vol(G)) (naturally satisfied)
"""
function KKT_sdp(
    YS::Vector{Float64},
    L::SparseMatrixCSC,
    d::Vector,
    ubsqr::Vector,
    lam::Vector,
    n::Int64,
    k::Int64;
    beta = -1e4,
    nev = 6,
    verbose = false,
)
    Y = reshape(@view(YS[1:n*k]), n, k)
    s = @view(YS[n*k+1:n*(k+1)])

    primal_feasi = zeros(n + 2)
    dual_feasi = zeros(nev)
    complement_slackness = zeros(1)


    primal_feasi[1] = sum(Y .* (Y .* d)) - 1
    v2 = Y' * d
    primal_feasi[2] = v2' * v2
    primal_feasi[3:n+2] = Y .^ 2 * ones(k) + s - ubsqr

    if verbose
        println("Primal feasibility: ")
        println(norm(primal_feasi, Inf))
    end

    # notice beta is free, no matter how you take it, the complement slackness Tr(ZX) = 0 is
    # always satisfied because Xd = 0. But this is numerical computation, we should make sure
    # |d' * X * d * beta| isn't too large, otherwise Tr(ZX) = 0 will be affected numerically.
    op = ArpackSimpleFunctionOp(
	    (y,x) -> begin 
		    mul!(y, L, x) # this will compute L*x
		    y .-= lam[1] * (d.*x )
		    y .-= (beta*(d'*x)) * (d) 
		    y .-= lam[3:n+2] .* x 
	    end, n)
    eigvals, eigvecs = symeigs(op, 1; which=:SA, ncv=min(100, n), maxiter=1000000)
    dual_feasi = real.(eigvals)


    if verbose
        println("Dual feasibility: ")
        println("Smallest eigenvalues of L - lambda*Diag(d) - beta*d*d' - Diag(gamma)")
        @show dual_feasi
    end

    V = L - lam[1] * Diagonal(d) - Diagonal(lam[3:n+2]) ## V is sparse
    complement_slackness[1] = sum(Y .* (V * Y)) - beta * sum((Y' * d).^2)

    if verbose
        println("Complementary Slackness:")
        @show complement_slackness[1]
    end

    return primal_feasi, dual_feasi, complement_slackness
end


function initialize(A::SparseMatrixCSC, d::Vector, mu::Float64, k::Int64, type::Int64)
    L = Diagonal(d) - A
    n = L.n
    Y = zeros(n, k + 1)
    Vol = sum(d)
    lb = sqrt(mu / (1 - mu) / Vol)
    ub = sqrt((1 - mu) / mu / Vol)
    ubsqr = ub .^ 2 * ones(n)
    if type == 1
        println("Initialization type is 1.")
        # rank k
        # Take k random sets with volume between mu Vol(G) and (1-mu) Vol(G)
        for j = 1:k
            ds = 0
            ord = randperm(n)
            for i = 1:n
                v = ord[i]
                ds += d[v]
                if ds >= mu * Vol
                    v1 = sqrt((Vol - ds) / ds / Vol)
                    v2 = sqrt(ds / (Vol - ds) / Vol)
                    for l = 1:n
                        u = ord[l]
                        if l <= i
                            Y[u, j] = -v1
                        elseif l > i
                            Y[u, j] = v2
                        end
                    end
                    break
                end
            end
        end
    elseif type == 2
        L = degree_normalized_Laplacian(A)

        eigvals, eigvecs = symeigs(L,  k + 1; which = :SA, maxiter = 10000)
        Y[:, 1:k] = eigvecs[:, 2:k+1]
    end
    Y = Y / sqrt(k)
    Y[:, k+1] = ubsqr - Y[:, 1:k] .^ 2 * ones(k)
    Y[:, k+1] = max.(Y[:, k+1], 0)
    Y[:, k+1] = min.(Y[:, k+1], ub^2 - lb^2)
    if type == 3 
        # use random Y and s
        Y = randn(n, k + 1)
    end
    return reshape(Y, n * (k + 1))
end


function print_info(mu, k, T, pobj, dobj, S_min_eigval, comp_slack, stationary, primal_norm, sigma,
                    eta, omega)
    header = ["μ", "k", "T", "pobj", "dobj", "S_min_eigval", "⟨YY', S⟩", "‖proj grad‖_∞", "‖pinfeas‖₂", "σₜ", "ηₜ", "ωₜ"]
    data = [mu k T pobj dobj S_min_eigval comp_slack stationary primal_norm sigma eta omega]
    pretty_table(data; column_labels=header, 
                 formatters = [fmt__printf("%.3E", [1]),
                               fmt__printf("%d", 2:3),
                               fmt__printf("%.3E", 4:12)])
end


function dual_value(n, lam, ub, lb, trace_bound, S_min_eigval)
    return (lam[1] + ub^2 * sum(lam[3:n+2]) - (ub^2 - lb^2) * sum(max.(0, lam[3:n+2])) + trace_bound * min(S_min_eigval, 0))
end


function _ALM(
    A::SparseMatrixCSC,
    mu::Float64,
    k::Int;
    maxiter = 10000,
    Ptol = 1e-5,
    Ktol = 1e-5,
    init_type = 2,
    maxfun = 150000,
    x0 = nothing,
)
    @show mu
    d = sum(A, dims = 1)[1, :]
    L = Diagonal(d) - A
    sigma = 10.0
    omega = 1 / sigma
    eta = 1 / sigma^0.1

    Vol = sum(d)
    n = L.n

    lb = sqrt(mu / (1 - mu) / Vol)
    ub = sqrt((1 - mu) / mu / Vol)

    trace_bound = sum(min.(ub^2,((1 - 2*mu) / (1 - mu) ./ d) .+ lb^2))
    trace_bound = min(trace_bound, 1)

    lam = zeros(n + 2)

    bounds = zeros(3, n * (k + 1))
    bounds[1, 1:n*k] .= 0
    bounds[1, n*k+1:n*(k+1)] .= 2
    bounds[2, 1:n*k] .= -Inf
    bounds[2, n*k+1:n*(k+1)] .= 0
    bounds[3, 1:n*k] .= Inf
    bounds[3, n*k+1:n*(k+1)] .= ub .^ 2 - lb .^ 2

    ubsqr = ub .^ 2 * ones(n)

    if init_type <= 3 
        YS = initialize(A, d, mu, k, init_type)
    else
        @assert x0 !== nothing "Please specify one initial Y."
        YS = zeros(n, k + 1)
        YS[:, 1:k] = x0
        YS[:, k+1] = ubsqr - YS[:, 1:k] .^ 2 * ones(k)  
        YS = reshape(YS, n * (k + 1))  
    end

    optimizer = L_BFGS_B(n * (k + 1), 3)

    cverg = false
    niter = 0
    for iter = 1:maxiter
        println("Iteration $iter:")

        _, xout = optimizer(
            xs -> f(xs, L, d, lam, ubsqr, sigma, n, k),
            (G, xs) -> g!(G, xs, L, d, lam, ubsqr, sigma, n, k),
            YS,
            bounds,
            m = 3,
            factr = 0,
            pgtol = omega,
            iprint = 0,
            maxfun = maxfun,
            maxiter = maxfun,
        )
        YS[:] = xout

        stationary = optimizer.dsave[13]
        primal_feasi = compute_primal_feasi(YS, d, ubsqr, n, k)
        primal_norm = norm(primal_feasi, Inf)

        KKT_dt = @elapsed begin
            primal_feasi, dual_feasi, comp_slack = KKT_sdp(YS, L, d, ubsqr, lam, n, k)
        end
        pobj = obj(YS, L, n, k)
        dobj = dual_value(n, lam, ub, lb, trace_bound, dual_feasi[1])
        print_info(mu, k, iter, pobj, dobj, dual_feasi[1], comp_slack, stationary,
                   norm(primal_feasi, 2), sigma, eta, omega)
        if norm(primal_feasi, Inf) <= eta
            if norm(primal_feasi, Inf) <= Ptol && stationary <= Ktol
                cverg = true
                niter = iter
            elseif iter < maxiter
                lam -= sigma * primal_feasi
                eta /= sigma^0.9
                omega /= sigma
            end
        elseif iter < maxiter
            sigma *= 10
            eta = 1 / sigma^0.1
            omega = 1 / sigma
        end
        eta = max(eta, Ptol)
        omega = max(omega, Ktol)
        if cverg
            break
        end
    end
    return YS, lam, sigma, ubsqr, cverg, n, k, niter
end


function ALM(
    mu::Float64,
    A::SparseMatrixCSC,
    k::Int,
    init_type::Int,
    Ktol::Float64,
    Ptol::Float64,
    maxfun::Int,
    filename::String,
    resultfolder::String,
)
    n = size(A)[1]
    d = sum(A, dims=1)[1, :]
    L = Diagonal(d) - A
    ALM_dt = @elapsed begin
        vec_YS, lam, sigma, ubsqr, cverg, n, k, niter = _ALM(
            A,
            mu,
            k;
            init_type = init_type,
            Ptol = Ptol,
            Ktol = Ktol,
            maxfun = maxfun,
        )
    end
    Gvol = sum(d)
    objval = obj(vec_YS, L, n, k) 
    KKT_dt = @elapsed begin
        primal_feasi, dual_feasi, comp_slack = KKT_sdp(vec_YS, L, d, ubsqr, lam, n, k)
    end
    theta = min(0, dual_feasi[1])
    YS = reshape(@views(vec_YS), n, k + 1)
    filepath = resultfolder * filename * "-$Ktol-$Ptol-$k-$mu-$(init_type).mat"
    matwrite(
        filepath,
        Dict(
            "ALM_dt" => ALM_dt,
            "mu" => mu,
            "init_type" => init_type,
            "Ktol" => Ktol,
            "Ptol" => Ptol,
            "YS" => YS,
            "lam" => lam,
            "sigma" => sigma,
            "ubsqr" => ubsqr,
            "cverg" => cverg,
            "n" => n,
            "k" => k,
            "niter" => niter,
            "primal_feasi" => primal_feasi,
            "dual_feasi" => dual_feasi, 
            "comp_slack" => comp_slack,
            "KKT_dt" => KKT_dt,
            "objval" => objval,
            "Gvol" => Gvol,
        );
    )
    delta = theta * sum(YS[:,1:k].^2)
    dt = ALM_dt + KKT_dt
    return dt, objval, delta
end


function make_jobs(params::Vector{Tuple{Float64,Int64}}, jobs::RemoteChannel)
    for param in params
        put!(jobs, param)
    end
    for i = 1:length(workers())
        put!(jobs, (-1, -1))
    end
end


function do_jobs(
    jobs::RemoteChannel,
    results::RemoteChannel,
    A::SparseMatrixCSC,
    init_type::Int,
    Ktol::Float64,
    Ptol::Float64,
    maxfun::Int,
    filename::String,
    resultfolder::String,
)
    while true
        mu, k = take!(jobs)
        if mu == -1
            break
        end
        exec_time, objval, delta = ALM(mu, A, k, init_type, Ktol, Ptol, maxfun, filename, resultfolder)
        put!(results, (mu, k, exec_time, objval, delta, myid()))
    end
end


function bulk_eval_network_profile(
    A::SparseMatrixCSC,
    mus::Vector{Float64},
    ks::Vector{Int64},
    init_type::Int,
    Ktol::Float64,
    Ptol::Float64,
    maxfun::Int,
    filename::String,
    resultfolder::String,
)
    @assert length(mus) == length(ks)
    jobs = RemoteChannel(() -> Channel{Tuple}(length(mus) + length(workers())))
    results = RemoteChannel(() -> Channel{Tuple}(length(mus)))
    make_jobs([(mus[i], ks[i]) for i = 1:length(mus)], jobs)
    for p in workers()
        remote_do(
            do_jobs,
            p,
            jobs,
            results,
            A,
            init_type,
            Ktol,
            Ptol,
            maxfun,
            filename,
            resultfolder,
        )
    end
    njob = length(mus)
    while njob > 0
        mu, k, exec_time, objval, delta, pid = take!(results)
        println(
            "Computation for mu = $mu, k = $k is done on worker $pid, which takes $exec_time secs. The objval is $objval,
            the delta is $delta, the final lower bound is $(objval + delta).",
        )
        njob -= 1
    end
end

end


using Test

#@show muconductance.test_g(n = 1000, k = 1000, ncases = 10) 
#@test muconductance.test_g(n = 20, k = 10, ncases = 10) <= 1e-6
