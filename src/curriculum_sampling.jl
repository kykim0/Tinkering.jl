using Distributions
using PyPlot
using Random
using Statistics


function plot_p_sigmoid(; lb, ub, alpha, filename=nothing)
    chisq_dist = Chisq(3.0)
    xs = [x for x in lb:0.05:ub]

    figure(figsize=(9.0, 6.0))
    pdfs = pdf.(chisq_dist, xs)
    plt.plot(xs, 5.0 * pdfs, label="cost", lw=1.0)


    alpha_quantile = quantile(chisq_dist, 1.0 - alpha)
    ys = [y for y in 0.0:0.05:1.2]
    plt.plot(fill(alpha_quantile, length(ys)), ys, label="α-quantile", linestyle="--")

    sigmoid_lw = 0.7
    sigmoid_1_0 = (x) -> 1 / (1 + exp.(-(x - alpha_quantile)))
    plt.plot(xs, sigmoid_1_0.(xs), label="sigmoid1", lw=sigmoid_lw)
    sigmoid_0_5 = (x) -> 1 / (1 + exp.(-0.5 * (x - alpha_quantile)))
    plt.plot(xs, sigmoid_0_5.(xs), label="sigmoid2", lw=sigmoid_lw)
    sigmoid_1_5 = (x) -> 1 / (1 + exp.(-2.0 * (x - alpha_quantile)))
    plt.plot(xs, sigmoid_1_5.(xs), label="sigmoid3", lw=sigmoid_lw)

    plt.legend()

    if !isnothing(filename)
        plt.savefig(filename, dpi=500)
    end
end

plot_p_sigmoid(lb=0.0, ub=15.0, alpha=0.05, filename="/Users/kykim/Desktop/sigmoid.png")
