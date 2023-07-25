using MultivariateStats, Parquet2, JLD2, HDF5, LinearAlgebra, SimilaritySearch
using Downloads: download
using Base.Iterators, Dates


include("io.jl")
include("laion-gold-standards.jl")

function select_queries(;
        n = 10_000,
        k = 1000,
        dbname="SISAP23-Challenge/laion2B-en-clip768v2-n=10M.h5",
        queriesfromfile="/data1/sadit/laion2B-en/img_emb/img_emb_1000.jld2")
    
    Q = jldopen(queriesfromfile) do f
        x = normalize_vectors!(Float32, f["emb"])
        sample = sort!(unique(rand(1:length(x), 2n))) # at least 10M
        MatrixDatabase(x[sample])
    end

    knns, dists = compute_knns(Float32, dbname, Q, NormalizedCosineDistance(), k, 10^6)
    Q, knns, dists
end

function filter_gold_standard(dists; k=10, minsum=0.1)  # compute this for largest dataset
    [(c[k] !== c[k+1] && sum(c[1:k]) > minsum) for c in eachcol(dists)]
end

function main()
    n = 10_000
    k = 10
    klarge = 1000

    nsize = "100M"
    earlygoldfile = "Private-Queries-and-Gold-clip768v2-$nsize.jld2"
    # select_queries perform faster since it uses an optimized distance function but uses less precision
    Q, knns, dists = if isfile(earlygoldfile)
        jldopen(earlygoldfile) do f
            f["Q"], f["knns"], f["dists"]
        end
    else
        Q, knns, dists = select_queries(; dbname="SISAP23-Challenge/laion2B-en-clip768v2-n=$nsize.h5", n, k=klarge)
        jldsave(earlygoldfile; Q, knns, dists)
        Q, knns, dists
    end

    let
        dlist = [i for (i, x) in enumerate(filter_gold_standard(dists; k)) if x]
        @show length(dlist)
        @assert length(dlist) >= n
        resize!(dlist, n)
        
        emb, knns, dists = Q.matrix[:, dlist], knns[:, dlist], dists[:, dlist]
        @assert emb isa Matrix{Float32}
        @show typeof(emb), typeof(knns), typeof(dists)
        @show size(emb), size(knns), size(dists)
        jldsave("private-queries-gold-10k-clip768v2.h5"; emb)
        # jldsave("laion2B-en-private-gold-standard-v2-$nsize-F64-IEEE754.h5"; knns, dists) # b

        @info "--- 10M"
        knns, dists = compute_knns(Float64, "SISAP23-Challenge/laion2B-en-clip768v2-n=10M.h5", MatrixDatabase(emb), NormCostDist(), k, 10^6)
        jldsave("laion2B-en-private-gold-standard-v2-10M-F64-IEEE754.h5"; knns, dists)

        @info "--- 30M"
        knns, dists = compute_knns(Float64, "SISAP23-Challenge/laion2B-en-clip768v2-n=30M.h5", MatrixDatabase(emb), NormCostDist(), k, 10^6)
        jldsave("laion2B-en-private-gold-standard-v2-30M-F64-IEEE754.h5"; knns, dists)
       
        @info "--- 100M"
        knns, dists = compute_knns(Float64, "SISAP23-Challenge/laion2B-en-clip768v2-n=100M.h5", MatrixDatabase(emb), NormCostDist(), k, 10^6)
        jldsave("laion2B-en-private-gold-standard-v2-100M-F64-IEEE754.h5"; knns, dists)
        # dlist
    end 
end
