using Distributions
using PyPlot
using Random
using Statistics


function sigmoid_slope(; bounds)
    lb, ub = bounds
    xs = [x for x in lb:0.05:ub]

    μ = 0.0; s = 0.1
    f_x = (x) -> 1 / (1 + exp(-(x - μ) / s))

    plt.figure(figsize=(9.0, 6.0))

    plt.plot(xs, f_x.(xs))

    x_l = -s * log(1 - 2 / (1 + sqrt(1 + 4 * s))) + μ
    x_r = -s * log(1 - 2 / (1 - sqrt(1 + 4 * s))) + μ
    plt.plot(x_l, f_x(x_l), marker="o")
    plt.plot(x_r, f_x(x_r), marker="o")

    plt.xlim(bounds); plt.ylim(bounds)
end


sigmoid_slope(bounds=(-2.0, 2.0))
