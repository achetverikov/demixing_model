# Explicit initialization avoids the package's k-means and sample-variance setup.
using GaussianMixtures, DelimitedFiles, Logging
global_logger(ConsoleLogger(stderr, Logging.Warn))

parameters(g) = hcat(g.w, g.μ, sqrt.(g.Σ))
function snapshot(g, x)
    # em!'s returned history is PRE-update; recompute the final total likelihood.
    hcat(parameters(g), fill(avll(g, x)*length(x), 2))
end

for name in ARGS[2:end]
    base = joinpath(ARGS[1], name)
    x = readdlm(base*"_data.csv", ',', Float64)
    theta = readdlm(base*"_initial.csv", ',', Float64)
    g = GMM(theta[:, 1], theta[:, 2:3], theta[:, 4:5].^2, History[], 0)
    for iteration in 1:5000
        before = parameters(g)
        em!(g, x; nIter=1, varfloor=0.0, parallel=false, debug=0)
        iteration == 1 && writedlm(base*"_julia_one.csv", snapshot(g, x), ',')
        if maximum(abs.(parameters(g)-before)) < 1e-10
            writedlm(base*"_julia_converged.csv", snapshot(g, x), ',')
            em!(g, x; nIter=1, varfloor=0.0, parallel=false, debug=0)
            writedlm(base*"_julia_extra.csv", snapshot(g, x), ',')
            break
        end
        iteration == 5000 && error("Reference EM did not converge")
    end
end
