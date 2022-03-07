using Distributions
using Plots
using Random


function f_x(x)
    # return 1 / (1 + exp(-x))
    return x
end


function normal_dist(μ=0.0, σ=1.0)
    return Normal(μ, σ)
end


function monte_carlo(n, p)
    samples = rand(p, n)
    estimates = f_x.(samples)
    return samples, estimates
end


function importance_sampling(n, q, p)
    samples = rand(q, n)
    pdf_q = pdf.(q, samples)
    pdf_p = pdf.(p, samples)
    estimates = f_x.(samples) .* (pdf_p ./ pdf_q)
    return samples, estimates
end


function plot_estimates(n_mc_estimates, n_is_estimates, delta_n)

    function x_range(l)
        total_n = length(l)
        ret = delta_n:delta_n:total_n
        if (last(ret) != total_n); push!(ret, total_n); end
        return ret
    end

    function n_mean(n_estimates, xl)
        ret = []
        for x in xl
            x_n_estimates = [mean(estimates[1:x]) for estimates in n_estimates]
            push!(ret, x_n_estimates)
        end
        return ret
    end

    # TODO(kykim): Do a cumsum type of optimization.
    x_mc = x_range(first(n_mc_estimates))
    n_y_mc = n_mean(n_mc_estimates, x_mc)
    min_y_mc, max_y_mc = minimum.(n_y_mc), maximum.(n_y_mc)
    mid_y_mc = (max_y_mc + min_y_mc) ./ 2.0

    x_is = x_range(first(n_is_estimates))
    n_y_is = n_mean(n_is_estimates, x_is)
    min_y_is, max_y_is = minimum.(n_y_is), maximum.(n_y_is)
    mid_y_is = (max_y_is + min_y_is) ./ 2.0
    
    p = plot(x_mc, mid_y_mc, ribbon=(max_y_mc - mid_y_mc), fillalpha=0.15,
             label="MC", lw=2, xlabel="no. of samples", ylabel="estimates")
    plot!(p, x_is, mid_y_is, ribbon=(max_y_is - mid_y_is), fillalpha=0.15, label="IS", lw=2)
    # savefig("myplot.png")
    return p
end


Random.seed!(0)

p = normal_dist(2.0, 1.0)
q = normal_dist(1.0, 1.0)

mc_n = 10_000
is_n = 10_000
delta_n = 100

n_trials = 5
n_mc_estimates, n_is_estimates = [], []
for _ in 1:n_trials
    mc_samples, mc_estimates = monte_carlo(mc_n, p)
    is_samples, is_estimates = importance_sampling(is_n, q, p)
    push!(n_mc_estimates, mc_estimates)
    push!(n_is_estimates, is_estimates)
end

plt = plot_estimates(n_mc_estimates, n_is_estimates, delta_n)
display(plt)
