"""Curriculum sampling experiments.

TODOs:
- Sigmoid based sampling at α.
- Simgoid based sampling at α' > α.
- Sigmoid based samlping with linear α schedule.
- Sigmoid based sampling with exponential α schedule.
"""

using Distributions
using PyPlot
using Random
using Statistics
using StatsBase


function _output_filename(filename)
    # return joinpath("/Users/kykim/dump", filename)
    return joinpath("/Users/kykim/Desktop", filename)
end


function _sigmoid(mu, s)
    return (x) -> (1 / (1 + exp(-(x - mu) / s)))
end


function _range(vec::Vector)
    max_e = maximum(vec); min_e = minimum(vec)
    return round((max_e + min_e) / 2.0, digits=3), round((max_e - min_e) / 2.0, digits=3)
end


function plot_p_sigmoid(; bounds, alpha, filename=nothing)
    chisq_dist = Chisq(3.0)
    lb, ub = bounds
    xs = collect(lb:0.05:ub)

    plt.figure(figsize=(9.0, 6.0))
    pdfs = pdf.(chisq_dist, xs)
    plt.plot(xs, 5.0 * pdfs, label="cost", lw=1.0)

    alpha_quantile = quantile(chisq_dist, 1.0 - alpha)
    ys = collect(0.0:0.05:1.2)
    plt.plot(fill(alpha_quantile, length(ys)), ys, label="α-quantile", linestyle="--")

    sigmoid_lw = 0.7
    sigmoid_1_0 = _sigmoid(alpha_quantile, 1.0)
    plt.plot(xs, sigmoid_1_0.(xs), label="sigmoid", lw=sigmoid_lw)
    sigmoid_2_0 = _sigmoid(alpha_quantile, 2.0)
    plt.plot(xs, sigmoid_2_0.(xs), label="sigmoid-lo", lw=sigmoid_lw)
    sigmoid_0_5 = _sigmoid(alpha_quantile, 0.5)
    plt.plot(xs, sigmoid_0_5.(xs), label="sigmoid-hi", lw=sigmoid_lw)

    plt.legend()

    if !isnothing(filename)
        plt.savefig(_output_filename(filename), dpi=500)
    end
end


function plot_estimates(n_estimates::Vector, xl::Vector, label::String)
    isempty(n_estimates) && return

    min_y, max_y = minimum.(n_estimates), maximum.(n_estimates)
    mid_y = (max_y + min_y) ./ 2.0

    plt.plot(xl, mid_y, label=label, lw=0.7)
    plt.fill_between(xl, min_y, max_y, alpha=0.1)
end


# Computes empirical (1-α)-quantile.
#
# Set normalize_w to true for self-normalized importance sampling.
function _ecdf_quantile(samples, alpha; normalize_w=false)
    raw_samples = samples; weights = []
    if first(samples) isa Tuple
        raw_samples = first.(samples)
        weights = last.(samples)
        if normalize_w
            weights = weights ./ sum(weights)
        end
    end
    weights = isempty(weights) ? Weights(Float64[]) : Weights(Float64.(weights))

    emp_cdf = StatsBase.ecdf(raw_samples; weights)
    # TODO: Binary search might be faster.
    x_last = emp_cdf.sorted_values[end]
    for x in Iterators.reverse(emp_cdf.sorted_values)
        if emp_cdf(x) < (1.0 - alpha)
            return x_last
        end
        x_last = x
    end
    return emp_cdf.sorted_values[begin]
end


# Samples according to q(x) \propto p_x * Pr(x) where Pr(x) = sigmoid_fn(x).
#
# This returns the sample drawn using rejection sampling along with its
# unnormalized weight.
function _sigmoid_sample_continuous(p_x, sigmoid_fn::Function, n_samples::Integer)
    samples = []
    # Note: This may be slow especially for the kinds of p_x that we deal with.
    while length(samples) < n_samples
        p_samples = rand.(p_x, (n_samples - length(samples)) * 100)
        pdfs = pdf.(p_x, p_samples)
        sample_probs = sigmoid_fn.(p_samples)
        accept_probs = rand(length(p_samples))
        for (idx, (s_pr, a_pr)) in enumerate(zip(sample_probs, accept_probs))
            if s_pr > a_pr
                q_u = pdfs[idx] * s_pr
                u_weight = pdfs[idx] / q_u
                push!(samples, (p_samples[idx], u_weight))
            end
        end
    end
    return samples[1:n_samples]
end


# Discrete version of _sigmoid_sample_continuous above.
#
# The input distribution is assumed to be an instance of DiscreteNonParametric.
function _sigmoid_sample(p_x, is_sigmoid_fn::Function, n_samples::Integer)
    # Construct the proposal distribution (discrete) q(x) \propto p_x * Pr(x).
    q_xs = support(p_x)
    q_ps = probs(p_x) .* is_sigmoid_fn.(q_xs); q_ps /= sum(q_ps)
    q_x = DiscreteNonParametric(q_xs, q_ps)

    q_samples = rand(q_x, n_samples)
    q_weights = pdf.(p_x, q_samples) ./ pdf.(q_x, q_samples)
    q_s_w = [(q_sample, q_weight) for (q_sample, q_weight) in zip(q_samples, q_weights)]
    return q_x, q_s_w
end


# Computes sample weights using the adaptive MIS approach.
#
# In this scheme, the DM weighting with a mixture consisting of the current and
# all past proposal distributions is used.
function _amis_weight(q_x_vec::Vector, sample_weight_vec::Vector)
    isempty(q_x_vec) && return sample_weight_vec
    new_sample_weight_vec = []
    curr_samples = first.(sample_weight_vec)
    curr_weights = last.(sample_weight_vec)
    prev_weights_all = [pdf.(q_x, curr_samples) for q_x in q_x_vec]
    for (idx, (sample, weight)) in enumerate(sample_weight_vec)
        prev_sample_weights = [prev_weights[idx] for prev_weights in prev_weights_all]
        new_weight = weight / (sum(prev_sample_weights) + weight)
        push!(new_sample_weight_vec, (sample, new_weight))
    end
    return new_sample_weight_vec
end


# Uses a sigmoid function with mu set to ecdf at the current target alpha.
#
# Fixed no. of samples drawn for each alpha based on the max no. of samples.
function _sigmoid_sample_mu(p_x, n_samples, mu, s, alphas; mix_weights=false)
    alpha_n = trunc(Int, n_samples / length(alphas))
    samples = []
    q_x_vec = []
    for alpha in alphas
        alpha_mu = isempty(samples) ? mu : _ecdf_quantile(samples, alpha)
        alpha_sigmoid_fn = _sigmoid(alpha_mu, s)
        alpha_n_samples = alpha == alphas[end] ? n_samples - length(samples) : alpha_n
        q_x, alpha_samples = _sigmoid_sample(p_x, alpha_sigmoid_fn, alpha_n_samples)
        if mix_weights
            alpha_samples = _amis_weight(q_x_vec, alpha_samples)
        end
        push!(q_x_vec, q_x)
        append!(samples, alpha_samples)
    end
    return samples
end


# Similar to _sigmoid_sample_mu but with adaptive slope.
function _sigmoid_sample_mu_s(p_x, n_samples, mu, s, alphas; mix_weights=false)
    alpha_n = trunc(Int, n_samples / length(alphas))
    samples = []
    q_x_vec = []
    equantiles = Dict(alpha => [] for alpha in Set(alphas))
    estds = Dict(alpha => [] for alpha in Set(alphas))
    alpha_mu = mu; alpha_s = s
    for alpha in alphas
        if !isempty(samples)
            alpha_mu = _ecdf_quantile(samples, alpha)
            push!(equantiles[alpha], alpha_mu)
            alpha_s = s - (2.0 - s) / (n_samples) * length(samples)
            if false && length(equantiles[alpha]) > 1
                xs = Float64.(equantiles[alpha])
                evar = StatsBase.var(xs, FrequencyWeights(ones(length(xs))); corrected=true)
                p = 0.95
                # Slope for which Pr(x_p) = p: s = -(x_p - \mu) / ln((1 - p) / p).
                alpha_s = max(-sqrt(evar) / log((1 - p) / p), 1.0)
                push!(estds[alpha], sqrt(evar))
            end
        end
        alpha_sigmoid_fn = _sigmoid(alpha_mu, alpha_s)
        alpha_n_samples = alpha == alphas[end] ? n_samples - length(samples) : alpha_n
        q_x, alpha_samples = _sigmoid_sample(p_x, alpha_sigmoid_fn, alpha_n_samples)
        if mix_weights
            alpha_samples = _amis_weight(q_x_vec, alpha_samples)
        end
        push!(q_x_vec, q_x)
        append!(samples, alpha_samples)
    end
    return samples, estds
end


function exp_mixture_p(; n_trials, n_samples, alphas, plot_delta_n)
    p_x = MixtureModel(
        [LogNormal(0.0, 0.75), Normal(10.0, 1.00), Normal(16.0, 0.75)],
        [0.70, 0.20, 0.10])
    t_alpha = alphas[end]
    alpha_quantile = quantile(p_x, 1.0 - t_alpha)

    mu = 5.0; s = 5.0
    is_sigmoid_fn = _sigmoid(mu, s)

    # Plot the distribution and the sigmoid fn.
    xs = collect(0.0:0.1:20.0)
    plt.figure(figsize=(9.0, 6.0))
    plt.plot(xs, pdf.(p_x, xs), label="cost", lw=0.7)
    # plt.plot(xs, is_sigmoid_fn.(xs), label="sigmoid", lw=0.7)
    ys = collect(0.0:0.05:0.50)
    plt.plot(fill(alpha_quantile, length(ys)), ys, label="α-quantile", linestyle="--")
    plt.xlabel("cost"); plt.ylabel("density"); plt.legend();
    plt.tight_layout()
    # plt.savefig(_output_filename("graphs.png"), dpi=500)

    n_mc_samples = []; n_is_samples = []
    for _ in 1:n_trials
        mc_samples = rand(p_x, n_samples)
        push!(n_mc_samples, mc_samples)

        is_samples = _sigmoid_sample_continuous(p_x, is_sigmoid_fn, n_samples)
        push!(n_is_samples, is_samples)
    end

    n_mc_estimates = []; n_is_estimates = []
    xl = collect(plot_delta_n:plot_delta_n:min(n_samples, n_samples + plot_delta_n - 1))
    for x in xl
        mc_quantiles = [_ecdf_quantile(mc_samples[1:x], t_alpha)
                        for mc_samples in n_mc_samples]
        push!(n_mc_estimates, mc_quantiles)

        is_quantiles = [_ecdf_quantile(is_samples[1:x], t_alpha, normalize_w=true)
                        for is_samples in n_is_samples]
        push!(n_is_estimates, is_quantiles)
    end

    # Plot MC and IS estimates w/ confidence regions.
    plt.figure(figsize=(9.0, 6.0))
    plot_estimates(n_mc_estimates, xl, "MC")
    plot_estimates(n_is_estimates, xl, "IS")
    plt.plot(1:n_samples, fill(alpha_quantile, n_samples), label="α-quantile", linestyle="--")
    plt.xlabel("samples"); plt.ylabel("estimates"); plt.legend();
    plt.tight_layout()
    # plt.savefig(_output_filename("convergence-$(mu)-$(s).png"), dpi=500)
end


function _mixture_distrib(; plot_graphs=true, alpha)
    # mix_p_x = MixtureModel(
    #     [LogNormal(0.0, 0.75), Normal(10.0, 1.00), Normal(16.0, 0.75)],
    #     [0.85, 0.075, 0.075])
    mix_p_x = MixtureModel(
        [LogNormal(0.0, 1.0), Normal(10.0, 1.00), Normal(20.0, 1.0)],
        [0.90, 0.07, 0.03])
    xs = collect(0.0:0.05:25.0)
    ps = pdf.(mix_p_x, xs); ps /= sum(ps)
    p_x = DiscreteNonParametric(xs, ps)

    alpha_quantile = quantile(p_x, 1.0 - alpha)

    # Plot the distribution.
    if plot_graphs
        plot_xs = collect(0.0:0.05:25.0)
        plt.figure(figsize=(9.0, 6.0))
        # plt.plot(plot_xs, pdf.(mix_p_x, plot_xs), label="cost")
        plt.bar(xs, ps, width=0.4, label="cost", alpha=0.7)
        # plt.plot(xs, is_sigmoid_fn.(xs), label="sigmoid", lw=0.7)
        plot_ys = collect(0.0:0.005:0.03)
        plt.plot(fill(alpha_quantile, length(plot_ys)), plot_ys, label="α-quantile",
                 linestyle="--", color="orange")
        plt.xlabel("cost"); plt.ylabel("p"); plt.legend();
        plt.tight_layout()
        # plt.savefig(_output_filename("graphs-$(mu)-$(s).png"), dpi=500)
    end

    return p_x, alpha_quantile
end


function exp_discrete_mixture_p(; n_trials, n_samples, alphas, mu, s,
                                plot_delta_n, plot_graphs=true)
    # Create a discrete version of the mixture density.
    p_x, alpha_quantile = _mixture_distrib(plot_graphs=plot_graphs, alpha=alphas[end])

    is_sigmoid_fn = _sigmoid(mu, s)

    estds = Dict()
    n_mc_samples = []; n_fixed_is_samples = []; n_mu_is_samples = []; n_mu_s_is_samples = []
    for _ in 1:n_trials
        mc_samples = rand(p_x, n_samples)
        push!(n_mc_samples, mc_samples)

        # q_x, fixed_is_samples = _sigmoid_sample(p_x, is_sigmoid_fn, n_samples)
        # push!(n_fixed_is_samples, fixed_is_samples)

        # mu_is_samples = _sigmoid_sample_mu(p_x, n_samples, mu, s, alphas)
        # push!(n_mu_is_samples, mu_is_samples)

        mu_s_is_samples, estds = _sigmoid_sample_mu_s(p_x, n_samples, mu, s, alphas)
        push!(n_mu_s_is_samples, mu_s_is_samples)
    end

    # plt.figure(figsize=(9.0, 6.0))
    # alpha_estds = estds[alphas[end]]
    # plt.plot(1:length(alpha_estds), alpha_estds)
    # plt.tight_layout()

    n_mc_estimates = []; n_fixed_is_estimates = []; n_mu_is_estimates = []; n_mu_s_is_estimates = []
    xl = collect(plot_delta_n:plot_delta_n:min(n_samples, n_samples + plot_delta_n - 1))
    for x in xl
        mc_quantiles = [_ecdf_quantile(mc_samples[1:x], alphas[end])
                        for mc_samples in n_mc_samples]
        push!(n_mc_estimates, mc_quantiles)

        if !isempty(n_fixed_is_samples)
            fixed_is_quantiles = [_ecdf_quantile(fixed_is_samples[1:x], alphas[end])
                                  for fixed_is_samples in n_fixed_is_samples]
            push!(n_fixed_is_estimates, fixed_is_quantiles)
        end

        if !isempty(n_mu_is_samples)
            mu_is_quantiles = [_ecdf_quantile(mu_is_samples[1:x], alphas[end])
                               for mu_is_samples in n_mu_is_samples]
            push!(n_mu_is_estimates, mu_is_quantiles)
        end

        if !isempty(n_mu_s_is_samples)
            mu_s_is_quantiles = [_ecdf_quantile(mu_s_is_samples[1:x], alphas[end])
                                 for mu_s_is_samples in n_mu_s_is_samples]
            push!(n_mu_s_is_estimates, mu_s_is_quantiles)
        end
    end

    # Print stats.
    println("No. of samples: $(n_samples). No. of trials: $(n_trials). Alpha: $(alphas[end]). Quantile: $(alpha_quantile)")
    delta_idx = trunc(Int, length(xl) / 5)
    for idx in delta_idx:delta_idx:length(xl)
        print("[n=$(xl[idx])]")
        mc_med, mc_rng = _range(n_mc_estimates[idx])
        print("  MC: $(mc_med)±$(mc_rng)")

        if !isempty(n_fixed_is_samples)
            fixed_is_med, fixed_is_rng = _range(n_fixed_is_estimates[idx])
            print("  IS-fixed: $(fixed_is_med)±$(fixed_is_rng)")
        end

        if !isempty(n_mu_is_estimates)
            mu_is_med, mu_is_rng = _range(n_mu_is_estimates[idx])
            print("  IS-mu: $(mu_is_med)±$(mu_is_rng)")
        end

        if !isempty(n_mu_s_is_estimates)
            mu_s_is_med, mu_s_is_rng = _range(n_mu_s_is_estimates[idx])
            print("  IS-mu-s: $(mu_s_is_med)±$(mu_s_is_rng)")
        end

        println()
    end

    # Plot MC and IS estimates w/ confidence regions.
    plt.figure(figsize=(9.0, 6.0))
    plot_estimates(n_mc_estimates, xl, "MC")
    plot_estimates(n_fixed_is_estimates, xl, "IS-fixed")
    plot_estimates(n_mu_is_estimates, xl, "IS-mu")
    plot_estimates(n_mu_s_is_estimates, xl, "IS-mu-s")
    plt.plot(1:n_samples, fill(alpha_quantile, n_samples), label="α-quantile", linestyle="--", lw=0.5)
    # plt.ylim([10.0, 25.0])
    # plt.ylim(12.5)
    plt.xlabel("no. of samples"); plt.ylabel("estimate"); plt.legend(loc=7);
    plt.tight_layout()
    plt.savefig(_output_filename("convergence-$(mu)-$(s).png"), dpi=500)
end


# plot_p_sigmoid(bounds=(0.0, 15.0), alpha=0.05, filename="sigmoid.png")

# exp_mixture_p(; n_trials=3, n_samples=1_000_000, alphas=[0.1], plot_delta_n=10_000)
# exp_mixture_p(; n_trials=3, n_samples=100_000, alphas=[0.1], plot_delta_n=1_000)

# _mixture_distrib(; plot_graphs=true, alpha=0.05);
exp_discrete_mixture_p(; n_trials=10, n_samples=500, alphas=fill(0.01, 250),
                       mu=10.0, s=4.0, plot_delta_n=2, plot_graphs=false)
