using MultivariateStats, JLD2, HDF5, LinearAlgebra, SimilaritySearch
using Downloads: download
using Base.Iterators, Dates

function eval_queries!(dist::SemiMetric, KNN::Vector{KnnResult}, Q::AbstractDatabase, X::AbstractDatabase, r)
    Threads.@threads for qID in eachindex(Q)
        q = Q[qID]
        for (i, objID) in enumerate(r)
            d = SimilaritySearch.evaluate(dist, q, X[i])
            push_item!(KNN[qID], IdWeight(objID, d))
        end
    end
end

function normalize_vectors!(X)
    for c in eachcol(X)
        normalize!(c)
    end

    MatrixDatabase(X)
end

"""
    gold_standard(; dbname, qname, s, k, outname)

Computes the gold standard of `k` nearest neighbors of dbname and qname (searching in batches of size `s`)
"""
function gold_standard(; dbname, qname, s, k, outname)
    dist = NormalizedCosineDistance()
    Q = jldopen(qname) do f
        normalize_vectors!(Float64.(f["emb"]))
    end

    KNN = [KnnResult(k) for _ in eachindex(Q)]

    h5open(dbname) do f
        E = f["emb"]
        @show typeof(E)
        n = size(E, 2)
        
        @info "working on $dbname (size $(n)) -- $(Dates.now())"
        
        for r in Iterators.partition(1:n, s)
            @info "advance $(r) -- $outname -- step: $(s) -- $(Dates.now())"
            X = normalize_vectors!(Float64.(E[:, r]))
            eval_queries!(dist, KNN, Q, X, r)
        end
    end

    @info "done $outname $(Dates.now()), now saving"

    knns = zeros(Int32, k, length(KNN))
    dists = zeros(Float32, k, length(KNN))

    for (i, res) in enumerate(KNN)
        knns[:, i] .= IdView(res)
        dists[:, i] .= DistView(res)
    end

    jldsave(outname; knns, dists)
end

function main()
    k = 1000
    s = 10^6
    qname = "SISAP23-Challenge/public-queries-10k-clip768v2.h5"

    for sizename in ["300K", "10M", "30M", "100M"]
        dbname = "SISAP23-Challenge/laion2B-en-clip768v2-n=$sizename.h5"
        outname = "laion2B-en-public-gold-standard-v2-$sizename-F64-IEEE754.h5"
        @info "start $(Dates.now()) $outname"
        gold_standard(; dbname, qname, s, k, outname)
    end
end
