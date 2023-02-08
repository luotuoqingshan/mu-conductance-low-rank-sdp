using LinearAlgebra
using SparseArrays
using MatrixNetworks
using Convex
using MosekTools
using SCS
using GenericArpack
const MOI = Convex.MOI


"""
Regularized Spectral Cut 
"""
function reg_spectral_cut(A::SparseMatrixCSC, tau::Float64)
    f,fval = fiedler_vector(sparse(A+tau*(ones(size(A)...))))
    return f, fval
end


"""
Computing rank-1 approximation x which minimizes ||xx' - X||. 
"""
function rank1_approx(X)
    F = svd(X)
    U, S, _ = F
    f = U[:, 1]
    return f 
end


"""
Solving the following convex program
min   Tr(LX)
s.t.  X \\circ (D - d * d' / Vol G) = 1 
      ||Diag(X)||_Inf <= rho^2
      X \\succeq 0
"""
function spectral_balanced_cut1(A::SparseMatrixCSC, rho::Float64)
    n = size(A, 1)
    d = sum(A, dims=1)[1, :] 
    D = Diagonal(d)
    L = D - A
    X = Semidefinite(n)
    vol_G = sum(d)
    problem = minimize(sum(L .* X), sum(X .* (D - d * d' / vol_G)) == 1, 
      norm_inf(diag(X)) <= rho^2, isposdef(X))
    opt = MOI.OptimizerWithAttributes(SCS.Optimizer, 
                                      "max_iters" => 1000000, "verbose" => 0,
                                      "eps" => 1e-4)
    solve!(problem, opt)
    return problem.optval, rank1_approx(X.value) 
end


"""
Solving the following program
min   Tr(LX) 
s.t.  Tr(DX) = 1 
      d'Xd = 0
      ||diag(X)||_Inf <= rho^2 
      X \\succeq 0
"""
function spectral_balanced_cut2(A::SparseMatrixCSC, rho::Float64)
    n = size(A, 1)
    d = sum(A, dims=1)[1, :]
    D = Diagonal(d)
    L = D - A 
    X = Semidefinite(n)
    problem = minimize(sum(L .* X), sum(D .* X) == 1, d' * X * d == 0, 
      norm_inf(diag(X)) <= rho^2, isposdef(X))
    opt = MOI.OptimizerWithAttributes(SCS.Optimizer,
                                      "max_iters" => 1000000, "verbose" => 0,
                                      "eps" => 1e-4)
    solve!(problem, opt)
    @show problem
    return problem.optval, rank1_approx(X.value) 
end


"""
Solving the following convex program
min L \\circ X 
s.t. D \\circ X = 1
     d'Xd = 0 
     mu / ((1 - mu) Vol)<= Diag(X) <= (1 - mu) / (mu Vol) 
     X \\succeq 0
"""
function spectral_balanced_cut3(A::SparseMatrixCSC, mu::Float64;
                                eps=1e-4, solver::String="SCS")
    n = size(A, 1)
    Vol = sum(A)
    D = Diagonal(vec(sum(A, dims=1)))
    L = sparse(D) - A 
    X = Semidefinite(n)
    d = diag(D)
    constraint1 = (sum(D .* X) == 1) 
    constraint2 = (sum(d' * X * d) == 0)
    constraint3 = (diag(X) <= (1 - mu) / (mu * Vol))
    constraint4 = (diag(X) >= mu / (1 - mu) / Vol) 
    constraint5 = isposdef(X)
    problem = minimize(sum(L .* X), 
                       constraint1, constraint2, constraint3, constraint4, constraint5)
    if solver == "SCS"
        opt = MOI.OptimizerWithAttributes(SCS.Optimizer,
                                         "max_iters" => 1000000, "verbose" => 1,
                                         "eps_abs" => eps, "eps_rel" => eps)
    elseif solver == "Mosek"
        opt = MOI.OptimizerWithAttributes(Mosek.Optimizer,
                                         "QUIET" => false, 
                                         "INTPNT_CO_TOL_DFEAS" => eps, 
                                         "INTPNT_CO_TOL_PFEAS" => eps,
                                         )
    end
    solve!(problem, opt)
    return problem.optval, X.value 
end