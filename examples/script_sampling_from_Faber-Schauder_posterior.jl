# Script

using Distributions, PyPlot, BayesianNonparametricStatistics

β=0.5
θ = sumoffunctions(vcat([faberschauderone],[faberschauder(j,k) for j in 0:4 for k in 1:2^j]),vcat([1.0],[(-1)^(j*k)*2^(-β*j) for j in 0:4 for k in 1:2^j]))

x = 0.0:0.001:1.0

y = θ.(x)

# Uncomment the following lines to plot θ.
# clf()
# plot(x,y)

sde = SDEWithConstantVariance(θ, 1.0, 0.0, 1000.0, 0.01)

X = rand(sde)

# Uncomment the following lines to plot a sample from sde.
# clf()
# plot(X)


M = SDEModel(1.0, 0.0)

Π = FaberSchauderExpansionWithGaussianCoefficients([2^(β*j) for j in 0:4])
postΠ = calculateposterior(Π, X, M )

for k in 1:100
    f = rand(postΠ)
    y = f.(x)
    plot(x,y)
end
