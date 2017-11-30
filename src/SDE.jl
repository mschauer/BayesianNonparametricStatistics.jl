# SDE.jl

"""
    abstract type AbstactSDE <: Any

subtypes: SDE and SDEWithConstantVariance.

Both subtypes implement a stochastic differential equation
dX_t=θ(X_t)dt + σ(X_t)dW_t.
"""
abstract type AbstractSDE end

"""
    struct SDE{S<:Function, T<:Function} <: AbstractSDE
      b::S
      σ::T
      beginvalue::Float64
      endtime::Float64
      dt::Float64
    end

Implements a stochastic differential equation dX_t=b(X_t)dt+σ(X_t)dW_t on time
interval [0,endtime], with W_t a Brownian motion. Both b and σ are functions
R→R. Beginvalue X_0=beginvalue and dt>0 is the precision with which we
discretise.

See also: SDEWithConstantVariance.

# Example
```julia
using PyPlot
clf()
sde = SDE(sin, x -> 2 + sin(x), 0, 10, 0.001)
X = rand(sde)
#
plot(X)
```
"""
struct SDE{S<:Function, T<:Function} <: AbstractSDE
  b::S
  σ::T
  beginvalue::Float64
  endtime::Float64
  dt::Float64

  function SDE(b::S, σ::T, beginvalue, endtime, dt) where {S<:Function, T<:Function}
    dt > 0 || error("The time discretisation (dt) should be positive")
    endtime > 0 || error("The endtime should be positive")
    new{S, T}(b, σ, beginvalue, endtime, dt)
  end
end

"""
    struct SDEWithConstantVariance{S<:Function} <: AbstractSDE
        b::S
        σ::Float64
        beginvalue::Float64
        endtime::Float64
        dt::Float64
    end

Implements the stochastic differential equation dX_t=b(X_t)dt+ σdW_t, with
b:R→R a function and σ a real number. Is a special case of SDE type with field σ
set equal to the lambda expression x->σ, but is faster.

# Example
```julia
using PyPlot
clf()
#
sde = SDEWithConstantVariance(cos, 1, 0, 10, 0.001)
X = rand(sde)
plot(X)
```
"""
struct SDEWithConstantVariance{S<:Function} <: AbstractSDE
    b::S
    σ::Float64
    beginvalue::Float64
    endtime::Float64
    dt::Float64

    function SDEWithConstantVariance(b::T, σ, beginvalue, endtime, dt) where
            T<:Function
        dt > 0 || error("The time discretisation (dt) should be positive")
        endtime > 0 || error("The endtime should be positive")
        new{T}(b, σ, beginvalue, endtime, dt)
    end
end

# Extends Distributions.rand for SDEWithConstantVariance types.
"""
    rand(sde::SDEWithConstantVariance)
    rand(sde::SDE)

Returns a SamplePath object which represents a sample path from an
AbstractSDE subtype sde. From time 0.0 to time sde.endtime, discretised with
precision sde.dt.

# Examples

```julia
using PyPlot
clf()
sde = SDEWithConstantVariance(sin, 1.0, 0.0, 10.0, 0.01)
X = rand(sde)
plot(X)
```

```julia
using PyPlot
clf()
sde = SDE(x->0.5*x, identity, 1, 1, 0.001)
X = rand(sde)
plot(X)
```
"""
function rand(sde::SDEWithConstantVariance)::SamplePath
  timeinterval = 0.0:sde.dt:sde.endtime
  lengthoftimeinterval = length(timeinterval)
  # Brownian motion increments multiplied with variance.
  increments = sde.σ * rand(Normal(0, sqrt(sde.dt)), lengthoftimeinterval)
  samplevalues = Array{Float64}(lengthoftimeinterval)
  samplevalues[1] = sde.beginvalue
  prevXval = sde.beginvalue
  for k in 2:lengthoftimeinterval
    samplevalues[k] = prevXval + sde.b(prevXval)*sde.dt + increments[k]
    prevXval = samplevalues[k]
  end
  return SamplePath(timeinterval, samplevalues)
end

# Extends Distributions.rand for SDE types.
function rand(sde::SDE)::SamplePath
  timeinterval = 0.0:sde.dt:sde.endtime
  lengthoftimeinterval = length(timeinterval)
  BMincrements = rand(Normal(0, sqrt(sde.dt)), lengthoftimeinterval)
  samplevalues = Array{Float64}(lengthoftimeinterval)
  samplevalues[1] = sde.beginvalue
  prevXval = sde.beginvalue
  for k in 2:lengthoftimeinterval
    samplevalues[k] = prevXval + sde.b(prevXval)*sde.dt +
      sde.σ(prevXval)*BMincrements[k]
    prevXval = samplevalues[k]
  end
  return SamplePath(timeinterval, samplevalues)
end
