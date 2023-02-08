using CSV
using MAT
using DataFrames
using DelimitedFiles
using SparseArrays
using MatrixNetworks
using LinearAlgebra

function del_selfloops(A::SparseMatrixCSC)
    colptr = A.colptr
    rowval = A.rowval
    nzval = A.nzval

    n = size(A, 1)
    I = Array{Int}(undef, 0) 
    J = Array{Int}(undef, 0)
    for u = 1:n
        for nzi in colptr[u]:(colptr[u+1]-1)
            v = rowval[nzi]
            if u != v
                push!(I, u)
                push!(J, v)
            end
        end
    end
    A = sparse(I, J, ones(Int64, length(I)))
    return A
end


function kcore(A::SparseMatrixCSC, k::Int)
    d, _ = corenums(A)
    S_ind = (d .>= k)
    A = A[S_ind, S_ind]
    A_lcc, _ = largest_component(A)
    return A_lcc
end


function load_graph_txt(path2file::String)
    E = readdlm(path2file, Int64)
    I = E[:, 1]
    J = E[:, 2]
    A = sparse(I, J, ones(Int64, length(I)))
    A, _ = largest_component(A)
    return A
end


function load_graph_smat(path2file::String)
    E = readdlm(path2file, Int64)
    I = E[2:end, 1] .+ 1
    J = E[2:end, 2] .+ 1
    V = E[2:end, 3]
    A = sparse(I, J, V)
    A, _ = largest_component(A) 
    return A
end


function load_graph_edge(path2file::String)
    E = readdlm(path2file, Int64)
    I = E[:, 1]  
    J = E[:, 2]
    A = sparse(I, J, ones(Int64,length(I))) 
    A_lcc, _ = largest_component(A)
    return A_lcc 
end


function load_graph_edge_xy(path2file::String)
    E = readdlm(path2file*".edges", Int64)
    I = E[:, 1]  
    J = E[:, 2]
    A = sparse(I, J, ones(Int64,length(I))) 
    xy = readdlm(path2file*".xy", Float64)
    A_lcc, filt = largest_component(A)
    xy = xy[filt, :]
    return A_lcc, xy 
end


function load_graph(dataset="ca-HepPh", 
                    datafolder=homedir()*"/mu-cond/data/input/")
    if dataset in ["facebook", "deezer"]
        if dataset == "facebook"
            path2file = datafolder*"musae_facebook_edges.csv"
            n = 22470 
        elseif dataset == "deezer"
            path2file = datafolder*"deezer_europe_edges.csv"
            n = 28281
        end
        E = DataFrame(CSV.File(path2file))
        I = E[1:end, 1] .+ 1
        J = E[1:end, 2] .+ 1
        A = sparse(I, J, ones(length(I)), n, n)
        A = max.(A, A')

        A, _ = largest_component(A)
    elseif dataset in ["ca-HepPh", "ca-AstroPh", "email-Enron"]
        I = Array{Int}(undef, 0) 
        J = Array{Int}(undef, 0)
        linecounter = 0
        path2file = datafolder*dataset*".txt"
        open(path2file) do file
            for l in eachline(file)  
                linecounter += 1
                if linecounter >= 5                 
                    u, v = split(l, '\t')
                    u = parse(Int, u)
                    v = parse(Int, v)
                    push!(I, u)
                    push!(J, v)
                end
            end
        end
        I .+= 1
        J .+= 1
        A = sparse(I, J, ones(Int64, length(I)))
        A, _ = largest_component(A)
    elseif dataset in ["soc-LiveJournal1"]
        path2file = datafolder*dataset*".txt"
        I = Array{Int}(undef, 0) 
        J = Array{Int}(undef, 0)
        linecounter = 0
        open(path2file) do file
            for l in eachline(file)  
                linecounter += 1
                if linecounter >= 5                 
                    u, v = split(l, '\t')
                    u = parse(Int, u)
                    v = parse(Int, v)
                    push!(I, u)
                    push!(J, v)
                    if u != v
                        push!(I, v)
                        push!(J, u)
                    end
                end
            end
        end
        vertices = unique([I; J])
        vid = Dict(vertices[i] => i for i = 1:length(vertices))
        for i = 1:length(I)
            I[i] = vid[I[i]]
            J[i] = vid[J[i]]
        end
        A = sparse(I, J, ones(Int64, length(I)))
        A, _ = largest_component(A)
    elseif dataset in ["dblp"]
        path2file = datafolder*dataset*"-cc.smat"
        A = load_graph_smat(path2file)
        A, _= largest_component(A)
    elseif dataset in ["email-Enron-core2", "email-Enron-core5", "email-Enron-core10", "email-Enron-core7"]
        path2file = datafolder*dataset*".edges"
        A = load_graph_edge(path2file)
    else
        path2file = datafolder*dataset
        strs = split(path2file, '.')
        @assert length(strs) > 1 "For datasets not listed, please include file extension(e.g. txt, smat)." 
        ext = strs[end]
        if ext == "edges"
            A = load_graph_edge(path2file)
        elseif ext == "smat"    
            A = load_graph_smat(path2file)
        else
            @assert false "File type not supported."
        end
    end
    #@assert A == A'

    ##remove self-loops
    A = del_selfloops(A)

    println("Name of dataset: $dataset")
    println("Number of vertices: $(size(A)[1])")
    println("Number of edges: $(div(sum(A) + sum(diag(A)), 2))")
    return A
end

using Test 

#@testset begin 
#    A = load_graph("ca-HepPh") 
#    n = size(A, 1)
#    # undirected
#    @test A == A' 
#    # self-loops removed
#    @test diag(A) == zeros(n) 
#end
