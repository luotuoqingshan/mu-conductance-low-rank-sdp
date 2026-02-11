using Plots
using TOML
using Random
using CSV
using DataFrames
using Printf
using MAT


function plotgraph(A, xy)
    px = Float64[]
    py = Float64[]
    for (ei, ej) in zip(findnz(triu(A, 1))[1:2]...)
        push!(px, xy[1, ei])
        push!(px, xy[1, ej])
        push!(px, NaN)

        push!(py, xy[2, ei])
        push!(py, xy[2, ej])
        push!(py, NaN)
    end
    plot(
        px,
        py,
        markersize = 6,
        linecolor = 1,
        linealpha = 0.8,
        linewidth = 0.7,
        markercolor = colorant"black",
        markerstrokecolor = colorant"white",
        framestyle = :none,
        legend = false,
        marker = :dot,
    )
end


function logpolar(xy)
    z = vec(xy[1, :]) + 1.0im .* vec(xy[2, :])
    lz = log.(1.15 .+ abs.(z)) .* exp.(1.0im .* angle.(z))
    return [real(lz)'; imag(lz)']
end


function showvector(x)
    vperm = sortperm(x .* sign(x[end]))
    # normalize sign
    x .*= sign(x[end])
    p = plotgraph(A, xy)
    scatter!(
        p,
        xy[1, vperm],
        xy[2, vperm],
        marker_z = x[vperm],
        markerstrokecolor = :black,
        alpha = 1.0,
        markerstrokewidth = 0,
        markersize = 6,
        border = :none,
    )
    p2 = scatter(
        -(1:length(x)),
        (0.5*x[vperm]),
        label = "",
        border = :none,
        markersize = 6,
        markerstrokewidth = 0.0,
        marker_z = x[vperm],
        colorbar = false,
    )
    plot(p, p2, layout = (2, 1))
end


function lrsdpobjplot(
    filename::String,
    muvals::Vector{Float64},
    k::Int,
    Ptol::Float64 = 1e-5,
    Ktol::Float64 = 1e-5,
    init_type::Int = 2;
    label::String = "1",
    xlabel::String = "Volume",
    ylabel::String = "Conductance",
    datafolder::String = pwd()*"/../data/output/",
    minvol::Int = 10,
)
    xs = Float64[]
    ys = Float64[]
    cnt = 0
    Gvol = 0
    for mu in muvals
        path2res =
            datafolder*filename*"/"*filename*"-$Ktol-$Ptol-$k-$mu-$(init_type).mat"
        if isfile(path2res)
            @show mu
            res = matread(path2res)
            YS = res["YS"]
            k = res["k"]
            n = res["n"]
            Y = YS[:, 1:k]

            objval = res["objval"]
            Gvol = res["Gvol"]

            @show objval
            @assert "dual_feasi" in keys(res) "KKT conditions haven't been evaluated"

            push!(xs, mu)
            dual_feasi = res["dual_feasi"]
            theta = min(0, dual_feasi[1])
            objval += min(1.0, (1 - mu) / mu * n / Gvol) * theta
            @show objval
            push!(ys, objval)
            cnt += 1
        end
    end
    # Post processing to smooth the curve,
    # This step is reasonable because by definition
    # mu-conductance is non-decreasing
    # however our lower bound doesn't, since it's aposteriori
    # so taking stepwise maximum is meaningful
    # however in most cases the original curve itself is non-decreasing
    for i = 1:(length(ys)-1)
        ys[i+1] = max(ys[i+1], ys[i])
    end

    # We ignore super small mu
    xvals = []
    yvals = []
    for i = 1:length(xs)
        if xs[i] * Gvol < minvol
            continue
        end
        push!(xvals, xs[i])
        push!(yvals, ys[i])
    end

    plot!(
        xvals * Gvol,
        yvals ./ 2,
        linewidth = 2,
        xscale = :log10,
        yscale = :log10,
        xlabel = xlabel,
        ylabel = ylabel,
        legend = false,
        label = label,
        marker = :d,
    )
    return cnt
end
