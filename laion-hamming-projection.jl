using SimilaritySearch, JLD2, HDF5, SurrogateDistanceModels 
using Downloads: download
using Base.Iterators, Dates, LinearAlgebra

#=
function get_binwalk(paths, dist, nrefs, permsize)
    modelname = @sprintf "%s/binwalk-model-nrefs=%04d-permsize=%02d.jld2" paths.models nrefs permsize

    if isfile(modelname)
        load(modelname, "model")
    else
        R = MatrixDatabase(load("LAION/refs-$nrefs.npy"))
        B = BinWalk(dist, R; permsize)
        jldsave(modelname, model=B)
        B
    end
end
=#

function process_big_matrix_hamming(; dbname, s, outfile)
    model = load("binwalk-models/binwalk-model-nrefs=1024-permsize=64.jld2", "model")

    h5open(dbname) do f
        E = f["emb"]
        n = size(E, 2)
        
        @info "working on $dbname (size $n) -- $(Dates.now())"
        hamming = Matrix{UInt64}(undef, 16, n)
        
        for r in Iterators.partition(1:n, s)
            @info "advance $(r) -- $dbname -- step: $(s) -- $(Dates.now())"
            X = E[:, r]
            for c in eachcol(X)
                normalize!(c)
            end

            db = StrideMatrixDatabase(X)
            hamming[:, r] = encode_database(model, db).matrix
        end

        jldsave(outfile * ".tmp"; hamming)
        mv(outfile * ".tmp", outfile)
        @info "done $outfile"
    end
end

function main_queries()
    @info "start $(Dates.now())"
    dbname = "SISAP23-Challenge/clip768/en-queries/public-queries-10k-clip768.h5"
    outfile = "SISAP23-Challenge/hamming/en-queries/public-queries-10k-hammingv2.h5"
    s = 10_000
    process_big_matrix_hamming(; dbname, s, outfile)
end

function main_datasets()
    @info "start $(Dates.now())"
    for sizename in ["100K", "300K", "10M", "30M", "100M"]
        dbname = "SISAP23-Challenge/clip768/en-bundles/laion2B-en-clip768-n=$(sizename).h5"
        outfile = "SISAP23-Challenge/hamming/en-bundles/laion2B-en-hammingv2-n=$(sizename).h5"
        s = 10^6
        process_big_matrix_hamming(; dbname, s, outfile)
    end
end