"""Gaussian process examples."""

using Distributions
using LinearAlgebra
using PyPlot
using Random


μ(X, m) = [m(x) for x in X]
Σ(X, k) = [k(x, xp) for x in X, xp in X]
K(X, Xp, k) = [k(x, xp) for x in X, xp in Xp]


# Mean functions.
mean_const(β) = (x) -> β
mean_lin(β) = (x) -> dot(x, β)
mean_zero() = (x) -> 0.0

# TODO(kykim): More mean functions.
# - Periodic mean: m(x)=a cos(2πx/p)+b sin(2πx/p).
# - Polynomical mean.


# Covariance functions.
cov_const(σ) = (x, xp) -> σ
cov_exp(l) = (x, xp) -> exp(-(x - xp) / l)
cov_gamma_exp(γ, l) = (x, xp) -> exp(-((x - xp) / l)^γ)
cov_sq_exp(l) = (x, xp) -> exp(-(x - xp)^2 / (2 * l^2))


mutable struct GaussianProcess
    m  # mean
    k  # covariance function
    X  # design points
    y  # objective values
    ν  # noise variance
end


function mvnrand(μ, Σ, inflation=1e-6)
    N = MvNormal(μ, Σ + inflation * I)
    return rand(N)
end
Base.rand(gp, X) = mvrand(μ(X, gp.m), Σ(X, gp.k))


function predict(gp, X_pred)
    X_pred = (isa(X_pred, Vector) ? X_pred : [X_pred])
    m, k, ν = gp.m, gp.k, gp.ν
    tmp = K(X_pred, gp.X, k) / (K(gp.X, gp.X, k) + ν * I)
    μ_p = μ(X_pred, m) + tmp * (gp.y - μ(gp.X, m))
    S = K(X_pred, X_pred, k) - tmp * K(gp.X, X_pred, k)
    ν_p = diag(S) .+ eps()
    return (μ_p, ν_p)
end


function plot_gp(gp, lb, ub, true_y::Function=nothing)
    xl = [x for x in lb:0.05:ub]

    # Plot observations.
    plot(gp.X, gp.y, "o", markersize=3, label="obs")

    # Plot the true y if available.
    if true_y != nothing
        plot(xl, [true_y(x) for x in xl], label="true y", lw=0.5)
    end

    # Plot predicted function mean with confidence interval.
    gp_pred = predict(gp, xl)
    gp_pred_mean, gp_pred_var = first(gp_pred), last(gp_pred)
    plot(xl, gp_pred_mean, label="pred mean", lw=0.5)
    gp_pred_c95 = 1.96 * gp_pred_var
    fill_between(xl, gp_pred_mean - gp_pred_c95, gp_pred_mean + gp_pred_c95,
                 alpha=0.1)
    xlabel("x"); ylabel("y"); legend();
    # savefig("myplot.png")
end


################################# 
# Below are experiment functions.
#################################

# An example from stor-i.github.io/GaussianProcesses.jl/latest/Regression.
function simple_regression()
    Random.seed!(0)

    n = 10
    x = 2π * rand(n)
    y = sin.(x) + 0.05 * rand(n)

    gp = GaussianProcess(mean_zero(), cov_sq_exp(1.0), x, y, 1.0)
    true_y = (x) -> sin(x)
    figure(1, figsize=(9.0, 6.0))
    plot_gp(gp, -2.0, 8.0, true_y)
end

simple_regression()
