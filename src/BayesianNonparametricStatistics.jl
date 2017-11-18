# TO DO: PyPlot.plot aanpassen zodat die meer argumenten kan oppakken met ...
#

__precompile__(true)

module BayesianNonparametricStatistics

using Distributions, PDMats, PyPlot

import Base: length, step

import Distributions: rand, ZeroVector

import PyPlot.plot

export AbstractModel, SDEModel, AbstractSDE, AbstractSamplePath,
AbstractGaussianProcess, SDE, SDEWithConstantVariance, SamplePathRange,
SamplePath, GaussianProcess, FaberSchauderExpansionWithGaussianCoefficients,
calculateposterior, fourier, faberschauder, faberschauderone,
fourierseriesprior, faberschauderprior,
calculatedependencystructurefaberschauderbasis, sumoffunctions

# model.jl implements AbstractModel and SDEModel.
include("model.jl")
# samplepath.jl implements SamplePath, SamplePathRange, AbstractSamplePath and
# extends Base.step and Base.length to the appropriate types.
include("samplepath.jl")
# basisfunctions.jl implements the Fourier and Faber-Schauder basis.
include("basisfunctions.jl")
# Implements AbstractSDE, SDE, SDEWithConstantVariance and extends
#Distribution.rand to sample an SDE sample path.
include("SDE.jl")
# Implements AbstractGaussianProcess, GaussianProcess,
# FaberSchauderExpansionWithGaussianCoefficients types and methods to calculate
# the posterior, with associated methods calculateLebesgueintegral,
# calculatestochasticintegral, calculatedependencystructurefaberschauderbasis.
include("gaussianprocess.jl")

end # End of module.
