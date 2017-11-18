using Distributions, BayesianNonparametricStatistics, PyPlot


sde = SDEWithConstantVariance(x->sinpi(2*x), 1.0, 0.0, 10000.0, 0.01)
X = rand(sde)
distribution = MvNormal([k^(-1.0) for k in 1:10])
Π = GaussianProcess([fourier(k) for k in 1:10], distribution)
M = SDEModel(1.0, 0.0)
postΠ = calculateposterior(Π, X, M)
clf()
x=0.0:0.01:1.0
for k in 1:10
    f = rand(postΠ)
    plot(x, f.(x))
end
