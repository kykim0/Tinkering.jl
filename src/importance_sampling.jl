"""Importance sampling examples.

TODOs:
- Try other f's, distributions (see Distributions.jl).
- Try a simple adaptive importance sampling algorithm.
"""

using Distributions
using PyPlot
using Random


# Plots f, p, q, and fp in the domain [lb, ub].
#
# Args:
# - f_x: a Function.
# - p_x: a Distribution.
# - q_x: a Function, a Distribution, or a Vector of the types.
# - lb/ub: Floats.
function plot_fs(f_x, p_x, q_x, lb, ub; lw=0.75)
    xl = [x for x in lb:0.05:ub]

    pdf_p_xl = pdf.(p_x, xl)
    plt.plot(xl, pdf_p_xl, label=raw"$p$", lw=lw)

    q_x_l = (isa(q_x, Vector) ? q_x : [q_x])
    for (idx, q_x_i) in enumerate(q_x_l)
        pdf_q_xl = (isa(q_x_i, Function) ? q_x_i.(xl) : pdf.(q_x_i, xl))
        label = string(raw"$q$", (idx == 1 ? "" : string(idx)))
        plt.plot(xl, pdf_q_xl, label=label, lw=lw)
    end

    f_xl = f_x.(xl)
    plt.plot(xl, f_xl, label=raw"$f$", lw=lw)
    plt.plot(xl, f_xl .* pdf_p_xl, label=raw"$f \times p$", lw=lw)
    plt.legend();
end


# Plots incremental sampling estimates with a confidence region.
#
# Args:
# - n_estimates: a Vector of estimates of length n.
# - delta_n: compute incremental estimates every this many steps.
# - label: a String label to use for the plot.
function plot_estimates(n_estimates::Vector, delta_n::Integer, label::String)

    function _x_range(l)
        total_n = length(l)
        ret = collect(delta_n:delta_n:total_n)
        if (last(ret) != total_n); push!(ret, total_n); end
        return ret
    end

    function _n_mean(n_estimates, xl)
        ret = []
        # TODO(kykim): Do a cumsum type of optimization.
        for x in xl
            x_n_estimates = [mean(estimates[1:x]) for estimates in n_estimates]
            push!(ret, x_n_estimates)
        end
        return ret
    end

    xl = _x_range(first(n_estimates))
    n_y = _n_mean(n_estimates, xl)
    min_y, max_y = minimum.(n_y), maximum.(n_y)
    mid_y = (max_y + min_y) ./ 2.0

    # For Plots.jl:
    #  plot(x_mc, mid_y_mc, ribbon=(max_y_mc - mid_y_mc), fillalpha=0.15,
    #       label="MC", lw=2, xlabel="no. of samples", ylabel="estimates")
    #  plot(x_is, mid_y_is, ribbon=(max_y_is - mid_y_is), fillalpha=0.15, label="IS", lw=2)
    plt.plot(xl, mid_y, label=label, lw=0.5)
    plt.fill_between(xl, min_y, max_y, alpha=0.1)
end


# Computes standard weights.
function is_standard_weights(p_x, q_x, samples)
    weights = pdf.(p_x, samples) ./ pdf.(q_x, samples)
    return weights
end


# Computes mixture weights based on all proposals.
function is_mixture_weights(p_x, q_x_l, samples)
    q_x_l = (isa(q_x_l, Vector) ? q_x_l : [q_x_l])
    denom = .+([pdf.(q_x_i, samples) for q_x_i in q_x_l]...) / length(q_x_l)
    weights = pdf.(p_x, samples) ./ denom
    return weights
end


# Simple experiment comparing Gaussian nominal and propsal distributions.
#
# Importance sampling with the proposal used leads to slightly smaller variance
# as it is better configured w.r.t to f * p.
function simple_exp(mc_n, is_n, delta_n, n_trials)
    Random.seed!(0)

    f_x = (x) -> (1. / (1. + exp(-(x - 4.5))))
    p_x = Normal(2.0, 1.0)
    # Note. Using a smaller σ (e.g., 0.5) leads to higher variance as q now has
    # lighter tails than p. This case of using a Gaussian proposal when the
    # nominal is also Gaussian is discussed in Owen 9.1.
    q_x = Normal(3.0, 0.50)  # Compare σ=1.0 vs. σ=0.5

    # Plot f, p, fp, q.
    figure(figsize=(9.0, 6.0))
    plot_fs(f_x, p_x, q_x, -3.0, 9.0)
    plt.tight_layout()
    # plt.savefig("distribs.png", dpi=500)

    # Compute n_trials many estimates.
    n_mc_estimates, n_is_estimates = [], []
    for _ in 1:n_trials
        mc_estimates = f_x.(rand(p_x, mc_n))
        push!(n_mc_estimates, mc_estimates)

        is_samples = rand(q_x, is_n)
        pdf_p, pdf_q = pdf.(p_x, is_samples), pdf.(q_x, is_samples)
        is_estimates = f_x.(is_samples) .* (pdf_p ./ pdf_q)
        push!(n_is_estimates, is_estimates)
    end

    # Plot MC and IS estimates w/ confidence regions.
    figure(figsize=(9.0, 6.0))
    plot_estimates(n_mc_estimates, delta_n, "MC")
    plot_estimates(n_is_estimates, delta_n, "IS")
    plt.xlabel("samples"); plt.ylabel("estimates"); plt.legend();
    plt.tight_layout()
    # plt.savefig("estimates.png", dpi=500)
end


# This is an example given in the chapter 6 importance sampling note.
#
# The exponential proposal distribution is so well-suited for the given f * p
# that importance sampling done with the distribution leads to extremely small
# variance.
#
# An alternate importance sampling with a "better" Gaussian also works well.
function normal_exponential_exp(mc_n, is_n, delta_n, n_trials)
    Random.seed!(0)

    f_x = (x) -> (x >= 4 ? 1.0 : 0.0)
    p_x = Normal(2.5, 1.0)

    # Using this to sample from an exponential shifted to the right by 4.
    q_x_tmp = Exponential(1.0)
    shift = 4.0
    pdf_q_fn = (x) -> pdf(q_x_tmp, x - shift)
    is_samples1 = rand(q_x_tmp, is_n) .+ shift

    q_x2 = Normal(3.5, 1.0)
    is_samples2 = rand(q_x2, is_n)

    # Plot f, p, fp, q.
    plt.figure(figsize=(9.0, 6.0))
    plot_fs(f_x, p_x, [pdf_q_fn, q_x2], -3.0, 9.0)

    # Compute n_trials many estimates.
    n_mc_estimates, n_is_estimates1, n_is_estimates2 = [], [], []
    for _ in 1:n_trials
        mc_estimates = f_x.(rand(p_x, mc_n))
        push!(n_mc_estimates, mc_estimates)

        pdf_p1, pdf_q1 = pdf.(p_x, is_samples1), pdf_q_fn.(is_samples1)
        is_estimates1 = f_x.(is_samples1) .* (pdf_p1 ./ pdf_q1)
        push!(n_is_estimates1, is_estimates1)

        pdf_p2, pdf_q2 = pdf.(p_x, is_samples2), pdf.(q_x2, is_samples2)
        is_estimates2 = f_x.(is_samples2) .* (pdf_p2 ./ pdf_q2)
        push!(n_is_estimates2, is_estimates2)
    end

    # Plot MC and IS estimates w/ confidence regions.
    plt.figure(figsize=(9.0, 6.0))
    plot_estimates(n_mc_estimates, delta_n, "MC")
    for (idx, n_is_estimates) in enumerate([n_is_estimates1, n_is_estimates2])
        label = string("IS", (idx == 1 ? "" : string(idx)))
        plot_estimates(n_is_estimates, delta_n, label)
    end
    plt.xlabel("samples"); plt.ylabel("estimates"); plt.legend();
end


# Experiment to demonstrate a simple multiple importance sampling use case.
#
# The f and p in the experiment are such that when a Gaussian proposal with a
# smaller variance is used for importance sampling, it can lead to a higher
# variance in estimate than the Monte Carlo estimate. This experiment
# demonstrates that using multiple importance sampling can be an option in such
# cases.
function exp_gaussian_mis(mc_n, is_n, delta_n, n_trials, weight_type="mixture")
    Random.seed!(0)

    f_x = (x) -> (1. / (1. + exp(-(x - 4.5))))
    p_x = Normal(2.0, 1.0)
    q_x1 = Normal(3.0, 0.75)  # Compare σ=0.75 vs. σ=0.5
    q_x2 = Normal(2.25, 0.75)
    q_x3 = Normal(3.75, 0.75)
    q_x_l = [q_x1, q_x2, q_x3]
    is_n_i = is_n ÷ length(q_x_l)  # Draw an equal no. of samples for each.

    # Plot f, p, fp, q.
    plt.figure(figsize=(9.0, 6.0))
    plot_fs(f_x, p_x, [q_x1, q_x2, q_x3], -3.0, 9.0)

    # Compute n_trials many estimates.
    n_mc_estimates, n_is_estimates = [], []
    for _ in 1:n_trials
        mc_estimates = f_x.(rand(p_x, mc_n))
        push!(n_mc_estimates, mc_estimates)

        is_estimates = []
        for q_x_i in q_x_l
            is_samples = rand(q_x_i, is_n_i)
            if weight_type == "standard"
                weights = is_standard_weights(p_x, q_x_i, is_samples)
            elseif weight_type == "mixture"
                weights = is_mixture_weights(p_x, q_x_l, is_samples)
            end
            append!(is_estimates, f_x.(is_samples) .* weights)
        end
        shuffle!(is_estimates)
        push!(n_is_estimates, is_estimates)
    end

    # Plot MC and IS estimates w/ confidence regions.
    plt.figure(figsize=(9.0, 6.0))
    plot_estimates(n_mc_estimates, delta_n, "MC")
    plot_estimates(n_is_estimates, delta_n, "IS")
    plt.xlabel("samples"); plt.ylabel("estimates"); plt.legend();
    plt.tight_layout()
    # plt.savefig("myplot.png", dpi=500)
end


# Experiment to compare a single Gaussian proposal and multiple proposals.
#
# The nominal distribution is a multimodal distribution designed in such a way
# that also leads to a multimodal f * p. Importance sampling with a single
# Gaussian can work reasonably well when configured properly to cover the
# regions of f * p, but the multimodal nature makes this somewhat tricky.
# Instead, importance sampling done with multiple Gaussians makes it easier to
# achieve the desired "coverage" and leads to lower variance.
function exp_mixture_gaussian_mis(mc_n, is_n, delta_n, n_trials,
                                  weight_type="mixture")
    Random.seed!(0)

    f_x = (x) -> (1. / (1. + exp(-(x - 1.5))))
    p_x = MixtureModel(
        Normal[Normal(-2.0, 1.0), Normal(1.0, 0.50), Normal(3.0, 0.5)],
        [0.6, 0.25, 0.15])

    # Compare a single Guassian proposal vs. multiple proposals.
    q_x = Normal(0.0, 1.5)
    q_x_mis1 = Normal(3.0, 0.75)
    q_x_mis2 = Normal(1.0, 1.0)
    q_x_mis3 = Normal(-1.0, 1.5)
    q_x_misl = [q_x_mis1, q_x_mis2, q_x_mis3]
    is_n_i = is_n ÷ length(q_x_misl)  # Draw an equal no. of samples for each.

    # Plot f, p, fp, q.
    plt.figure(figsize=(9.0, 6.0))
    plot_fs(f_x, p_x, [q_x, q_x_misl...], -6.0, 7.0)
    plt.tight_layout()
    # plt.savefig("mis_graphs.png", dpi=500)

    # Compute n_trials many estimates.
    n_mc_estimates, n_is_estimates, n_mis_estimates = [], [], []
    for _ in 1:n_trials
        mc_estimates = f_x.(rand(p_x, mc_n))
        push!(n_mc_estimates, mc_estimates)

        is_samples = rand(q_x, is_n)
        pdf_p, pdf_q = pdf.(p_x, is_samples), pdf.(q_x, is_samples)
        is_estimates = f_x.(is_samples) .* (pdf_p ./ pdf_q)
        push!(n_is_estimates, is_estimates)

        mis_estimates = []
        for q_x_i in q_x_misl
            mis_samples = rand(q_x_i, is_n_i)
            if weight_type == "standard"
                weights = is_standard_weights(p_x, q_x_i, mis_samples)
            elseif weight_type == "mixture"
                weights = is_mixture_weights(p_x, q_x_misl, mis_samples)
            end
            append!(mis_estimates, f_x.(mis_samples) .* weights)
        end
        shuffle!(mis_estimates)
        push!(n_mis_estimates, mis_estimates)
    end

    # Plot MC and IS estimates w/ confidence regions.
    plt.figure(figsize=(9.0, 6.0))
    plot_estimates(n_mc_estimates, delta_n, "MC")
    plot_estimates(n_is_estimates, delta_n, "IS")
    plot_estimates(n_mis_estimates, delta_n, "MIS-$(weight_type)")
    plt.xlabel("Samples"); plt.ylabel("Estimates"); plt.legend();
    plt.tight_layout()
    plt.savefig("mis.png", dpi=500)
end


# simple_exp(10_000, 10_000, 100, 5)

# normal_exponential_exp(10_000, 10_000, 100, 5)

# Compare the "standard" and "mixture" weightings. Mixture weights does
# noticeably better presumably because it is more robust against small q which
# can significantly increase variance.
# exp_gaussian_mis(9000, 9000, 100, 5, "mixture")

PyPlot.matplotlib[:rc]("font", family="serif")
PyPlot.matplotlib[:rc]("text", usetex=true)
PyPlot.matplotlib[:rc]("pgf", rcfonts=false)

exp_mixture_gaussian_mis(5000, 5000, 20, 5, "mixture")
