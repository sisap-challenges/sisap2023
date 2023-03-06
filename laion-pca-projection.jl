using MultivariateStats, JLD2, HDF5, LinearAlgebra
using Downloads: download
using Base.Iterators, Dates

function process_big_matrix_pca_32_96(; dbname, key, s, pca32name, pca96name)
    pca32model = load("model-pca32-laion2B-10M.jld2", "pca32")
    pca96model = load("model-pca96-laion2B-10M.jld2", "pca96")

    h5open(dbname) do f
        E = f[key]
        n = size(E, 2)
        
        @info "working on $dbname (size $n) -- $(Dates.now())"
        pca32 = Matrix{Float32}(undef, 32, n)
        pca96 = Matrix{Float32}(undef, 96, n)
        
        for r in Iterators.partition(1:n, s)
            @info "advance $(r) -- $dbname -- step: $(s) -- $(Dates.now())"
            X = E[:, r]
            for c in eachcol(X)
                normalize!(c)
            end
            pca32[:, r] = predict(pca32model, X)
            pca96[:, r] = predict(pca96model, X)
        end

        jldsave(pca32name; pca32)
        jldsave(pca96name; pca96)
        @info "done $pca96name"
    end
end

function main_queries()
    @info "start $(Dates.now())"
    dbname = "SISAP23-Challenge/clip768/en-queries/public-queries-10k-clip768.h5"
    key = "emb"
    pca32name = "SISAP23-Challenge/pca32/en-queries/public-queries-10k-pca32v2.h5"
    pca96name = "SISAP23-Challenge/pca96/en-queries/public-queries-10k-pca96v2.h5"
    s = 10_000
    process_big_matrix_pca_32_96(; dbname, key, s, pca32name, pca96name)
end

function main_datasets()
    @info "start $(Dates.now())"
    for sizename in ["100K", "300K", "10M", "30M", "100M"]
        dbname = "SISAP23-Challenge/clip768/en-bundles/laion2B-en-clip768-n=$(sizename).h5"
        key = "emb"
        pca32name = "SISAP23-Challenge/pca32/en-bundles/laion2B-en-pca32v2-n=$(sizename).h5"
        pca96name = "SISAP23-Challenge/pca96/en-bundles/laion2B-en-pca96v2-n=$(sizename).h5"
        s = 10^6
        process_big_matrix_pca_32_96(; dbname, key, s, pca32name, pca96name)
    end
end