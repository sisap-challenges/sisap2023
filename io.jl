using JLD2, HDF5, LinearAlgebra
using Base.Iterators


function normalize_vectors!(FloatType, X)
    X_ = FloatType.(X)
    for c in eachcol(X_)
        normalize!(c)
    end

    MatrixDatabase(X_)
end

function load_laion(dbname)
    X = load(dbname, "emb")
    normalize_vectors!(X)
end

