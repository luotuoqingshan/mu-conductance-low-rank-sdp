#=
NCPPlots

Goal, make some nicer looking NCP plots like we had in the LocalGraphClustering repo.

An NCP plot in the LGC repo had:

- hexbin for histogram2d with log-spaced x, y values
- but real labels.

(Copyright note, I did these all from memory, and did not check the
LGC code...)

https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.cut.html
=#

# https://stackoverflow.com/questions/47391489/group-dataframe-by-binning-a-columnfloat64-in-julia
# Also Categorical Arrays cuts
##
#cut(log10.(ncpdata.cond), 50)

include("hexbins.jl")
using StatsBase, Random
using CategoricalArrays
import Plots
function myhexbin(x,y;nbins=100)
  lx,ly = log10.(x), log10.(y)
  lxmin,lxmax,lymin,lymax = extrema(lx)..., extrema(ly)...
  bmin = min(lxmin,lymin)
  bmax = max(lymax,lymax)

  hexhist = fit(HexBinPlots.HexHistogram,lx,ly,
    6/nbins,6/nbins,
    #(bmax-bmin)/nbins,(bmax-bmin)/nbins;
    boundingbox=[bmin,bmax,bmin,bmax])
  h,vh = HexBinPlots.make_shapes(hexhist)
  vmax = maximum(vh)
  xsh = Vector{Float64}()
  ysh = Vector{Float64}()
  for k in eachindex(h)
    append!(xsh,h[k].x)
    push!(xsh,h[k].x[1])
    push!(xsh,NaN)
    append!(ysh,h[k].y)
    push!(ysh,h[k].y[1])
    push!(ysh,NaN)
  end
  color = log.(vh.+1)
  zz = repeat(color, inner=8)
  Plots.plot(10.0.^xsh, 10.0.^ysh, fill_z = zz, linecolor=nothing,
        seriestype=:shape,xscale=:log10,yscale=:log10,label="",colorbar=false)
  #Plots.plot(10.0.^xsh,10.0.^ysh,fill_z=log10.(vh.+1),linecolor=nothing,
  #      seriestype=:shape,xscale=:log10,yscale=:log10,label="",colorbar=false)
end

##
#using Plots
#myhexbin(x, y, nbins=30)
#
#myhexbin(randn(1000).^2,randn(1000).^2, nbins=50)
##
using CategoricalArrays, Statistics
# regarding groupby
# https://github.com/JuliaLang/julia/issues/32331
function myncpplot(x,y;nbins=100,plotmin::Bool=true,plotmedian::Bool=true)
  myhexbin(x,y,nbins=nbins)
  #scatter!(x,y,alpha=0.25)

  minx = Vector{Float64}()
  miny = Vector{Float64}()
  medianx = Vector{Float64}()
  mediany = Vector{Float64}()

  logx = log10.(x)
  breaks = range(0, stop=nextfloat(maximum(logx)), length=floor(Int, nbins))
  cv = CategoricalArrays.cut(log10.(x),breaks;allowempty=true)
  p = sortperm(cv)
  firstindex = 1
  while firstindex <= length(p)
    first = cv[p[firstindex]]
    lastindex = firstindex + 1
    while lastindex <= length(p) && cv[p[lastindex]] == first
      lastindex += 1
    end
    # get the index of the minimizing element of y
    imin = p[firstindex + argmin(@view y[p[firstindex:lastindex-1]]) - 1]
    #println(first, " ", firstindex, " ", lastindex, " ", imin)
    push!(minx, x[imin])
    push!(miny, y[imin])

    push!(medianx, median(@view x[p[firstindex:lastindex-1]]))
    push!(mediany, median(@view y[p[firstindex:lastindex-1]]))
    firstindex = lastindex # setup for next
  end
  if plotmin
    Plots.plot!(minx,miny,label="", color=1, linewidth=2, xlabel="Volume", ylabel="Conductance")
  end
  if plotmedian
    Plots.plot!(medianx,mediany,label="", color=1)
  end
end
#myncpplot(ncpdata.size, ncpdata.cond)


function myncpplot_min(x,y;nbins=100, color=1, label="mu=0.1")
    #scatter!(x,y,alpha=0.25)
  
    minx = Vector{Float64}()
    miny = Vector{Float64}()
  
    logx = log10.(x)
    breaks = range(0, stop=nextfloat(maximum(logx)), length=nbins)
    cv = CategoricalArrays.cut(log10.(x),breaks;allowempty=true)
    p = sortperm(cv)
    firstindex = 1
    while firstindex <= length(p)
      first = cv[p[firstindex]]
      lastindex = firstindex + 1
      while lastindex <= length(p) && cv[p[lastindex]] == first
        lastindex += 1
      end
      # get the index of the minimizing element of y
      imin = p[firstindex + argmin(@view y[p[firstindex:lastindex-1]]) - 1]
      #println(first, " ", firstindex, " ", lastindex, " ", imin)
      push!(minx, x[imin])
      push!(miny, y[imin])
  
      firstindex = lastindex # setup for next
    end
    Plots.plot!(minx,miny,label=label, color=color, linewidth=2, xscale=:log10,yscale=:log10, xlabel="Volume", ylabel="Conductance")
end


using DataFrames


function diffusion_ncpplot(
    ncp::DataFrame;
    samples::Int=1000000,
    nbins::Int=100,
)
    vols = min.(ncp.volume_seed, ncp.volume_other)
    conds = ncp.cond
    total = length(conds)
    perm = randperm(total)[1:min(samples, total)]     
    myncpplot(vols[perm], conds[perm]; nbins=nbins, plotmedian=false)
end




