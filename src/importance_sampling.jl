"""
TODOs:
- Try other f's, distributions (see Distributions.jl).
- Try a simple adaptive importance sampling algorithm.
"""

using Distributions
using PyPlot
using Random


function monte_carlo(f_x, p_x, n)
    samples = rand(p_x, n)
    estimates = f_x.(samples)
    return samples, estimates
end


function importance_sampling(f_x, p_x, q_x, n)
    samples = rand(q_x, n)
    pdf_q = pdf.(q_x, samples)
    pdf_p = pdf.(p_x, samples)
    estimates = f_x.(samples) .* (pdf_p ./ pdf_q)
    return samples, estimates
end


function importance_sampling(f_x, p_x, pdf_q_fn, is_samples)
    pdf_q = pdf_q_fn.(is_samples)
    pdf_p = pdf.(p_x, is_samples)
    estimates = f_x.(is_samples) .* (pdf_p ./ pdf_q)
    return estimates
end


function plot_fs(f_x, p_x, q_x, lb, ub)
    xl = [x for x in lb:0.1:ub]
    f_xl = f_x.(xl)
    pdf_p_xl, pdf_q_xl = pdf.(p_x, xl), pdf.(q_x, xl)

    plot(xl, pdf_p_xl, label="p")
    plot(xl, pdf_q_xl, label="q")
    plot(xl, f_xl, label="f")
    plot(xl, f_xl .* pdf_p_xl, label="f*p")
    legend();
end


function plot_fs(f_x, p_x, pdf_q_fn, lb, ub)
    xl = [x for x in lb:0.1:ub]
    f_xl = f_x.(xl)
    pdf_p_xl, pdf_q_xl = pdf.(p_x, xl), pdf_q_fn.(xl)

    plot(xl, pdf_p_xl, label="p")
    plot(xl, pdf_q_xl, label="q")
    plot(xl, f_xl, label="f")
    plot(xl, f_xl .* pdf_p_xl, label="f*p")
    legend();
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

    # For Plots.jl:
    #  plot(x_mc, mid_y_mc, ribbon=(max_y_mc - mid_y_mc), fillalpha=0.15,
    #       label="MC", lw=2, xlabel="no. of samples", ylabel="estimates")
    #  plot(x_is, mid_y_is, ribbon=(max_y_is - mid_y_is), fillalpha=0.15, label="IS", lw=2)
    plot(x_mc, mid_y_mc, label="MC")
    xlabel("no. samples"); ylabel("estimates");
    fill_between(x_mc, min_y_mc, max_y_mc, alpha=0.1)
    plot(x_is, mid_y_is, label="IS")
    fill_between(x_is, min_y_is, max_y_is, alpha=0.1)
    legend();
    # savefig("myplot.png")
end


function n_estimates(f_x, p_x, q_x, mc_n, is_n, n_trials)
    n_mc_estimates, n_is_estimates = [], []
    for _ in 1:n_trials
        mc_samples, mc_estimates = monte_carlo(f_x, p_x, mc_n)
        is_samples, is_estimates = importance_sampling(f_x, p_x, q_x, is_n)
        push!(n_mc_estimates, mc_estimates)
        push!(n_is_estimates, is_estimates)
    end
    return n_mc_estimates, n_is_estimates
end


function n_estimates(f_x, p_x, pdf_q_fn, mc_n, is_samples, n_trials)
    n_mc_estimates, n_is_estimates = [], []
    for _ in 1:n_trials
        mc_samples, mc_estimates = monte_carlo(f_x, p_x, mc_n)
        is_estimates = importance_sampling(f_x, p_x, pdf_q_fn, is_samples)
        push!(n_mc_estimates, mc_estimates)
        push!(n_is_estimates, is_estimates)
    end
    return n_mc_estimates, n_is_estimates
end


# Below are various experiment functions.
function simple_exp(mc_n, is_n, delta_n, n_trials)
    Random.seed!(0)

    f_x = (x) -> (1. / (1. + exp(-(x - 4.5))))
    p_x = Normal(2.0, 1.0)
    q_x = Normal(3.0, 0.75)  # Compare σ=0.75 vs. σ=0.5
    figure(1, figsize=(6.4, 4.8))
    plot_fs(f_x, p_x, q_x, -3.0, 9.0)

    n_mc_estimates, n_is_estimates = n_estimates(
        f_x, p_x, q_x, mc_n, is_n, n_trials)
    figure(2, figsize=(6.4, 4.8))
    plot_estimates(n_mc_estimates, n_is_estimates, delta_n)

    # all_plt = plot(fs_plt, est_plt, layout=(1, 2))
    # display(all_plt)
end


# This is an example given in the chapter 6 importance sampling note.
function normal_exponential_exp(mc_n, is_n, delta_n, n_trials)
    Random.seed!(0)

    f_x = (x) -> (x >= 4 ? 1.0 : 0.0)
    p_x = Normal(2.5, 1.0)
    # Using this to sample from an exponential shifted to the right by 4.
    q_x_tmp = Exponential(1.0)
    shift = 4.0
    pdf_q_fn = (x) -> pdf(q_x_tmp, x - shift)
    is_samples = rand(q_x_tmp, is_n) .+ shift
    figure(1, figsize=(6.4, 4.8))
    plot_fs(f_x, p_x, pdf_q_fn, -3.0, 9.0)

    n_mc_estimates, n_is_estimates = n_estimates(
        f_x, p_x, pdf_q_fn, mc_n, is_samples, n_trials)
    figure(2, figsize=(6.4, 4.8))
    plot_estimates(n_mc_estimates, n_is_estimates, delta_n)
end

# simple_exp(10_000, 10_000, 100, 5)
normal_exponential_exp(10_000, 10_000, 100, 5)
