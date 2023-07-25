using MultivariateStats, JLD2, HDF5, LinearAlgebra, SimilaritySearch
using Downloads: download
using Base.Iterators, Dates

struct NormCostDist <: SemiMetric end
function SimilaritySearch.evaluate(::NormCostDist, u::T, v) where T
    d = zero(eltype(T))
    @inbounds @simd for i in eachindex(u)
        d += u[i] * v[i]
    end

    one(eltype(T)) - d
end

function eval_queries!(dist::SemiMetric, KNN::Vector{KnnResult}, Q::AbstractDatabase, X::AbstractDatabase, r)
    Threads.@threads for qID in eachindex(Q)
        q = Q[qID]
        for (i, objID) in enumerate(r)
            d = SimilaritySearch.evaluate(dist, q, X[i])
            push_item!(KNN[qID], IdWeight(objID, d))
        end
    end
end

function normalize_vectors!(FloatType, X)
    X_ = FloatType.(X)
    for c in eachcol(X_)
        normalize!(c)
    end

    MatrixDatabase(X_)
end

"""
    gold_standard(FT, dist; dbname, qname, s, k, outname)

Computes the gold standard of `k` nearest neighbors of dbname and qname (searching in batches of size `s`)
"""
function gold_standard(::Type{FT}, dist; dbname, qname, s, k, outname) where FT
    Q = jldopen(qname) do f
        normalize_vectors!(FT, f["emb"])
    end

    compute_knns(FT, dbname, Q, dist, k, s)

    jldsave(outname; knns, dists)
end

function compute_knns(::Type{FT}, dbname::String, Q, dist::SemiMetric, k::Integer, s::Integer) where {FT<:AbstractFloat}
    KNN = [KnnResult(k) for _ in eachindex(Q)]

    h5open(dbname) do f
        E = f["emb"]
        @show typeof(E)
        n = size(E, 2)
        
        @info "working on $dbname (size $(n)) -- $(Dates.now())"
        
        for r in Iterators.partition(1:n, s)
            @info "advance $(r) --- step: $(s) -- $(Dates.now())"
            X = normalize_vectors!(FT, E[:, r])
            eval_queries!(dist, KNN, Q, X, r)
        end
    end

    @info "done $(Dates.now()), now saving"

    knns = zeros(Int32, k, length(KNN))
    dists = zeros(Float32, k, length(KNN))

    for (i, res) in enumerate(KNN)
        knns[:, i] .= IdView(res)
        dists[:, i] .= DistView(res)
    end

    knns, dists
end

function main()
    k = 1000
    s = 10^6
    #dist = NormalizedCosineDistance()
    dist = NormCostDist()
    FloatType = Float32
    qname = "SISAP23-Challenge/public-queries-10k-clip768v2.h5"

    for sizename in ["300K", "10M", "30M", "100M"]
        dbname = "SISAP23-Challenge/laion2B-en-clip768v2-n=$sizename.h5"
        outname = "laion2B-en-public-gold-standard-v2-$sizename-F32-IEEE754-explicit-distance.h5"
        if isfile(outname)
            @info "found $outname, skipping to the next setup"
        else
            @info "start $(Dates.now()) $outname"
            gold_standard(FloatType, dist; dbname, qname, s, k, outname)
        end
    end
end
