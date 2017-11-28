# gaussianprocess.jl

"""
    abstract type AbstractGaussianProcess end

Subtypes: GaussianProcess and FaberSchauderExpansionWithGaussianCoefficients.
Supertype: Any

Both subtypes represent a Gaussian process expressed as the inner product of a
Gaussian vector of real-valued coefficients and and a vector of functions. The
distribution is represented by a subtype of AbstractMvNormal from the
Distribution package.

See also: GaussianProcess and FaberSchauderExpansionWithGaussianCoefficients.
"""
abstract type AbstractGaussianProcess end

"""
    struct GaussianProcess{T<:AbstractMvNormal} <: AbstractGaussianProcess
        basis::Vector{<:Function}
        distribution::T
    end

Implements a Gaussian process defined as an inner product of a Gaussian vector
with distribution 'distribution' of type T<:AbstractMvNormal of the
Distributions package and a vector of functions, the basis of the function
space. For the use of a Gaussian process expanded in Faber-Schauder functions up
to level j, we recommend using FaberSchauderExpansionWithGaussianCoefficients,
as this makes efficient use of the sparsity structure of Faber-Schauder functions.

See also: FaberSchauderExpansionWithGaussianCoefficients, AbstractGaussianProcess

# Example
```julia
using Distributions, PyPlot
α = 0.5
Π = GaussianProcess([fourier(k) for k in 1:10], MvNormalCanon([k^(α+0.5) for k in 1:10]))
X = rand(Π)
clf()
plot(X)
```
"""
struct GaussianProcess{T<:AbstractMvNormal} <: AbstractGaussianProcess
  basis::Vector{<:Function}
  # distribution is the distribution of the coefficients corresponding to the
  # basis. distribution is a AbstractMvNormal subtype of the Distributions
  # package.
  distribution::T
  function GaussianProcess(basis::Vector{<:Function}, distribution::T) where
        T<:AbstractMvNormal
      length(basis)==length(distribution) || error("The basis and distribution are not of equal length.")
      new{T}(basis, distribution)
  end
end

"""
    calculatedependencystructurefaberschauderbasis(higestlevel::Int64)

Higest level should be at least zero. Returns a
2^(higestlevel+1) x 2^(higestlevel+1)-matrix A (say) with A[i,i'] = true when
Faber-Schauder function i=(j,k) (with i=2^j+k) and i'=(j',k') have essentially
non-empty support, and false otherwise. When an entry A[i,i']=false,
\int_0^1ψ_i(X_t)ψ_{i'}(X_t)dX_t=0 and \int_0^1ψ_i(X_t)ψ_{i'}(X_t)dt are zero,
which speeds up calculations in the calculateposterior method.

# Examples
```
calculatedependencystructurefaberschauderbasis(0)
calculatedependencystructurefaberschauderbasis(2)
```
"""
function calculatedependencystructurefaberschauderbasis(higestlevel::Int64)
    numberofbasisfunctions = 2^(higestlevel+1)
    dependentcoefficients = Array{Bool,2}(numberofbasisfunctions,
        numberofbasisfunctions)
    for k in 1:numberofbasisfunctions
        dependentcoefficients[k,k] = true
    end
    for k in 2:numberofbasisfunctions
        dependentcoefficients[1, k] = true
    end
    for k in 3:numberofbasisfunctions
        dependentcoefficients[2, k] = true
    end
    for j in 1:higestlevel
        for k in 1:2^j
            for l in k+1:2^j
                dependentcoefficients[2^j+k, 2^j+l] = false
            end
        end
    end
    for jone in 1:higestlevel
        for kone in 1:2^jone
            for jtwo in jone+1:higestlevel
                jdifference = jtwo - jone
                for ktwo in 1:2^jtwo
                    if (kone-1)*2^jdifference < ktwo ≤ kone*2^jdifference
                        dependentcoefficients[2^jone+kone, 2^jtwo+ktwo] = true
                    else
                        dependentcoefficients[2^jone+kone, 2^jtwo+ktwo] = false
                    end
                end
            end
        end
    end
    return Symmetric(dependentcoefficients)
end

"""
    struct FaberSchauderExpansionWithGaussianCoefficients <: AbstractGaussianProcess
        higestlevel::Int64
        basis::Vector{<:Function}
        distribution::MvNormalCanon{Float64, PDSparseMat{Float64, SparseMatrixCSC{Float64, Int64}}, S} where S<:Union{Vector{Float64}, ZeroVector{Float64}}
        dependentcoefficients::Symmetric{Bool, Array{Bool, 2}}
    end

Implements a Gaussian Process with Faber-Schauder functions and optimally exploits
the sparsity structure of Faber-Schauder functions.

Constructors:
    FaberSchauderExpansionWithGaussianCoefficients(higestlevel, distribution)
    FaberSchauderExpansionWithGaussianCoefficients(inversevariancesperlevel::Vector{Float64})

The user is not allowed to set basis and dependentcoefficients, they are calculated
from the input. The length of distribution should be equal to 2^(higestlevel+1).
The second constructor defines a Gaussian process with Faber-Schauder basis with
independent coefficients with length(inversevariancesperlevel)-1 levels, so
2^(length(inversevariancesperlevel)) number of basis functions. Where the variance
of the coefficients belonging to level k is inversevariancesperlevel[k+1] (we start with level 0).

# Example
```julia
using PyPlot
clf()
α = 0.5
Π = FaberSchauderExpansionWithGaussianCoefficients([2^(2*j*α) for j in 0:5])
f = rand(Π)
x = -1.0:0.01:2.0
plot(x, f.(x))
```
"""
struct FaberSchauderExpansionWithGaussianCoefficients <: AbstractGaussianProcess
    higestlevel::Int64
    basis::Vector{<:Function}
    distribution::MvNormalCanon{Float64, PDSparseMat{Float64, SparseMatrixCSC{Float64, Int64}}, S} where S<:Union{Vector{Float64}, ZeroVector{Float64}}
    dependentcoefficients::Symmetric{Bool, Array{Bool, 2}}
    function FaberSchauderExpansionWithGaussianCoefficients(higestlevel::Int64,
        distribution::MvNormalCanon{Float64, PDSparseMat{Float64,
        SparseMatrixCSC{Float64,Int64}}, T}) where T<:Union{Vector{Float64},ZeroVector{Float64}}
        length(distribution) == 2^(higestlevel+1) || error("The length of the distribution is not equal to 2^(higestlevel+1).")
        dependentcoefficients = calculatedependencystructurefaberschauderbasis(higestlevel)
        basis = vcat(faberschauderone, [faberschauder(j,k) for j in 0:higestlevel for k in 1:2^j])
        new(higestlevel, basis, distribution, dependentcoefficients)
    end
end

# Constructor
function FaberSchauderExpansionWithGaussianCoefficients(inversevariancesperlevel::Vector{Float64})
    lenghtinversevariancesperlevel = length(inversevariancesperlevel)
    lenghtinversevariancesperlevel == 0 && error("inversevariancesperlevel is of zero length.")
    # We start with level zero.
    higestlevel = lenghtinversevariancesperlevel - 1
    # There are two functions of level zero.
    vectorofinversevariances = repmat(inversevariancesperlevel[1:1],2)
    # and 2^k of level k, k=1,2,...
    for k in 1:higestlevel
        vectorofinversevariances = vcat(vectorofinversevariances, repmat(inversevariancesperlevel[k+1:k+1],2^k))
    end
    distribution = MvNormalCanon(PDSparseMat(spdiagm(vectorofinversevariances)))
    return FaberSchauderExpansionWithGaussianCoefficients(higestlevel, distribution)
end

# Extends Base.length to objects of a subtypes of AbstractGaussianProcess.
"""
    length(Π::AbstractGaussianProcess)

Returns the number of basis functions == length of the distribution of the
coefficients. So for a FaberSchauderExpansionWithGaussianCoefficients object it
this is equal to 2^(higestlevel+1).

# Example
```julia
Π = GaussianProcess([sin, cos], MvNormal([1.,1.]))
length(Π)
#
Π = FaberSchauderExpansionWithGaussianCoefficients([2.0^j for j in 0:3])
length(Π)
```
"""
length(Π::AbstractGaussianProcess)=length(Π.basis)

"""
    sumoffunctions(vectoroffunctions::Vector{<:Function}, vectorofscalars::Vector{Float64})

Calculates the 'inner product' of the function vector and the scalar vector. In
other words the sum of the functions weigthed by vectorofscalars. Returns a
function.
"""
function sumoffunctions(vectoroffunctions::Vector{<:Function},
    vectorofscalars::Vector{Float64})
  n = length(vectoroffunctions)
  n == length(vectorofscalars) || error("The vector of functions and the vector of scalars should be of equal length")
  return function(x::Float64)
    value = 0.0;
    for i in 1:n
      value += vectorofscalars[i]*vectoroffunctions[i](x)
    end
    return value
  end
end

# Extend Distributions.rand to objects which are subtypes of
# AbstractGaussianProcess. Returns a function.
"""
    rand(Π::AbstractGaussianProcess)

Returns a random function, where the coefficients have distribution Π.distribution
and the basis functions are defined in Π.basis.

# Example
```julia
using Distributions, PyPlot
distribution = MvNormalCanon(collect(1:10))
Π = GaussianProcess([fourier(k) for  k in 1:10], distribution)
f = rand(Π)
x = -1.0:0.01:2.0
clf()
plot(x, f.(x))
```
"""
function rand(Π::AbstractGaussianProcess)
  Z = rand(Π.distribution)
  return sumoffunctions(Π.basis,Z)
end

"""
    rand(Π::FaberSchauderExpansionWithGaussianCoefficients)

Samples a random function from Gaussian process Π.

# Examples

```julia
using PyPlot, Distributions
clf()
Π = FaberSchauderExpansionWithGaussianCoefficients([2.0^j for j in 0:3])
f = rand(Π)
x = -1.0:0.01:2.0
plot(x,f.(x))
```
"""
function rand(Π::FaberSchauderExpansionWithGaussianCoefficients)
    h = Π.distribution.h
    if typeof(h) <: ZeroVector
        h = zeros(length(h))
    end
    J = full(Π.distribution.J)
    Z = rand(MvNormalCanon(h,J))
    return sumoffunctions(Π.basis, Z)
end

# Calculates the stochastic Ito integral.
# Is not exported. Used in calculateposterior.
"""
    calculatestochasticintegral(f::Function, X::AbstractSamplePath)

Internal function. Calculates the stochastic (Ito) integral \int_0^T f(X_t)dX_t,
where T=X.timeinterval[end].
"""
function calculatestochasticintegral(f::Function, X::AbstractSamplePath)::Float64
  return sum(map(f, X.samplevalues[1:end-1]).*(X.samplevalues[2:end] -
    X.samplevalues[1:end-1]))
end

# Calculates the Lebesgue integral.
# Is not exported. Used in calculateposterior.
"""
    calculateLebesgueintegral(f::Function, X::SamplePathRange)
    calculateLebesgueintegral(f::Function, X::AbstractSamplePath)

Is an internal function. Calculates the Lebesgue integral \int_0^T f(X_t)dt,
with T=X.timeinterval[end].
"""
function calculateLebesgueintegral(f::Function, X::SamplePathRange)::Float64
  return sum(map(f, X.samplevalues[1:end-1]))*step(X)
end

# Calculates the Lebesgue integral for SamplePath objects, but would also work
# for SamplePathRange objects.
function calculateLebesgueintegral(f::Function, X::AbstractSamplePath)::Float64
  return sum(map(f, X.samplevalues[1:end-1]).*(X.timeinterval[2:end]-
    X.timeinterval[1:end-1]))
end

# plot(X::SamplePath) extends PyPlot.plot.
# TO DO: needs to be extended, so that other PyPlot options may be used.
"""
    plot(X::AbstractSamplePath)

Equivalent to plot(X.timeinterval, X.samplevalues).
"""
plot(X::AbstractSamplePath) = plot(X.timeinterval, X.samplevalues)

function dividefunctions(f::Function, g::Function)
    return function(x) return f(x)/g(x) end
end

function dividefunctions(f::Function, g::Number)
    return function(x) return f(x)/g end
end

dividefunctions(f::Number, g::Number) = f*g

function multiplyfunctions(f::Function, g::Function)
    return function(x) return f(x)*g(x) end
end

function multiplyfunctions(f::Function, g::Number)
    return function(x) return g*f(x) end
end

multiplyfunctions(f::Number, g::Function) = multiplyfunctions(g, f)

multiplyfunctions(f::Number, g::Float64) = f*g

function calculateGirsanovmatrix(Π::AbstractGaussianProcess,
        X::AbstractSamplePath, M::SDEModel)
    d = length(Π)
    girsanovmatrix = Array{Float64}(d, d)
    for k in 1:d
        for l in k:d
            girsanovmatrix[k,l] = calculateLebesgueintegral(dividefunctions(multiplyfunctions(Π.basis[k], Π.basis[l]),multiplyfunctions(M.σ, M.σ)), X)
        end
    end
    return Symmetric(girsanovmatrix)
end

function calculateGirsanovvector(Π::AbstractGaussianProcess, X::AbstractSamplePath, M::SDEModel)
    return [calculatestochasticintegral(dividefunctions(Π.basis[k], multiplyfunctions(M.σ,M.σ)),X) for k in 1:length(Π)]
end
# The following function calculates the posterior, with prior Π and data X. Π is
# a Gaussian process and X is data satisfying an unknown stochastic differential
# equation dX_t=θ(X_t)dt+dt.
"""
    calculateposterior(Π::GaussianProcess, X::AbstractSamplePath, M::SDEModel{Float64})
    calculateposterior(Π::GaussianProcess, X::AbstractSamplePath, M::SDEModel{<:Function})
    calculateposterior(Π::FaberSchauderExpansionWithGaussianCoefficients, X::AbstractSamplePath, M::SDEModel{Float64})
    calculateposterior(Π::FaberSchauderExpansionWithGaussianCoefficients, X::AbstractSamplePath, M::SDEModel{<:Function})

Calculates the posterior distribution Π(⋅∣X), and returns it as a
GaussianProcess object. Uses M to determine the right likelihood.

# Example

```julia
using PyPlot
clf()
sde = SDEWithConstantVariance(x->sinpi(2*x), 1.0, 0.0, 10000.0, 0.01)
X = rand(sde)
Π = FaberSchauderExpansionWithGaussianCoefficients([2.0^j for j in 0:5])
M = SDEModel(1.0, 0.0)
postΠ = calculateposterior(Π, X, M)
x=0.0:0.01:1.0
for k in 1:10
    f = rand(postΠ)
    plot(x, f.(x))
end
```
"""
function calculateposterior(Π::GaussianProcess,
        X::AbstractSamplePath, M::SDEModel)::GaussianProcess
    girsanovvector = calculateGirsanovvector(Π, X, M)
    girsanovmatrix = calculateGirsanovmatrix(Π, X, M)
    precisionmatrixposterior = girsanovmatrix + invcov(Π.distribution)
    potentialposterior = girsanovvector + mean(Π.distribution)
    posteriordistribution = MvNormalCanon(potentialposterior,
        precisionmatrixposterior)
    return GaussianProcess(Π.basis, posteriordistribution)
end

function calculateposterior(Π::FaberSchauderExpansionWithGaussianCoefficients,
        X::AbstractSamplePath, M::SDEModel)::FaberSchauderExpansionWithGaussianCoefficients
    girsanovvector = calculateGirsanovvector(Π, X, M)
    girsanovmatrix = calculateGirsanovmatrix(Π, X, M)
    girsanovmatrix = sparse(girsanovmatrix)
    girsanovmatrix = PDSparseMat(girsanovmatrix)
    precisionmatrixposterior = girsanovmatrix + Π.distribution.J
    potentialposterior = girsanovvector + mean(Π.distribution)
    meanposterior = precisionmatrixposterior \ potentialposterior
    posteriordistribution = MvNormalCanon(meanposterior, potentialposterior,
        precisionmatrixposterior)
    return FaberSchauderExpansionWithGaussianCoefficients(Π.higestlevel,
        posteriordistribution)
end
