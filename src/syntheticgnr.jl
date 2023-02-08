using Random
using NearestNeighbors
using SparseArrays
using LinearAlgebra
using MatrixNetworks


function gnr(n,m;r=0.1,k=5)
    xy = [1.5*randn(2,n) r.*randn(2,m)]
    T = BallTree(xy)
    idxs = knn(T, xy, k)[1]
    # form the edges for sparse
    ei = Int[]
    ej = Int[]
    for i=1:n+m
      for j=idxs[i]
        if i > j
          push!(ei,i)
          push!(ej,j)
        end
      end
    end
    # symmetrized edges
    A = sparse(ei,ej,1,n+m,n+m)
    A = dropzeros!(max.(A,A'))
    return xy, A
end

xy, A = gnr(90, 10)
A, filt = largest_component(A)
xy = xy'
xy = xy[filt, :]
d = vec(sum(A, dims=1))
@show size(A)[1] 
@show sum(A)