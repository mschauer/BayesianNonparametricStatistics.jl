# samplepath.jl

"""
    abstract type AbstractSamplePath <: Any

subtypes: SamplePath and SamplePathRange.

Both subtypes implement a continuous samplepath (necessarily discretised).
"""
abstract type AbstractSamplePath end

"""
    SamplePathRange(timeinterval::T, samplevalues::Vector{Float64}) <:
        AbstractSamplePath where T<:StepRangeLen{Float64}

Implements a continuous stochastic process, where the time interval is
equidistance.

Constructors:

SamplePathRange(timeinterval, samplevalues), with timeinterval a Float64 range
object, and samplevalues a Float64 vector of the same length.

SamplePathRange(timeinterval, f), with timeinterval a Float64 range object and f
a function that takes Float64 and returns a Float64. Returns
SamplePathRange(timeinterval, f.(timeinterval)).

See also: AbstractSamplePath, SamplePath, SDE, SDEWithConstantVariance.

# Examples

```julia
using PyPlot
clf()
x = 0.0:0.001:10.0
y = sin.(x)
X = SamplePathRange(x,y)
plot(X)
```

```julia
using PyPlot
sde = SDEWithConstantVariance(sin, 1, 0, 10, 0.001);
X = rand(sde)
plot(X)
```
"""
struct SamplePathRange{T<:StepRangeLen{Float64}} <: AbstractSamplePath
  timeinterval::T
  samplevalues::Vector{Float64}

  function SamplePathRange(timeinterval::T, samplevalues) where
      T<:StepRangeLen{Float64}
    length(timeinterval) == length(samplevalues) ||
        error("Length timeinterval should be equal to length samplevalues")
    step(timeinterval) > 0 || error("Time interval should be increasing.")
    new{T}(timeinterval, samplevalues)
  end
end
# Constructor
SamplePathRange(timeinterval::StepRangeLen{Float64}, f::Function) =
    SamplePathRange(timeinterval, f.(timeinterval))

"""
    isincreasing(x::AbstractVector{<:Number})::Bool

Tests whether a vector of numbers is strictly increasing. Is internal to
NonparametricBayesForDiffusions.
"""
function isincreasing(x::AbstractVector{<:Number})
  for i in 1:length(x)-1
    x[i+1] <= x[i] && return false
  end
  return true
end

"""
    SamplePath(timeinterval::S, samplevalues::T) where
        {S<:AbstractVector{Float64}, T<:AbstractVector{Float64}}

 Implements a continuous samplepath where the timeinterval is not necessarily
 equally spaced. Sample value samplevalues[k] is the value of the process at
 time timeinterval[k]. timeinterval is an increasing vector. timeinterval and
 samplevalues are of equal length.

 Constructors:
 SamplePath(timeinterval, samplevalues) (as above)
 SamplePath(timeinterval, f::Function) =
    SamplePath(timeinterval, f.(timeinterval))


 See also: AbstractSamplePath, SamplePathRange.

 # Examples

```julia
using PyPlot
clf()
t = logspace(-5,5,10)
v = exp.(t)
X = SamplePath(t, v)
plot(X)
```
"""
struct SamplePath{S<:AbstractVector{Float64}, T<:AbstractVector{Float64}} <:
    AbstractSamplePath
  timeinterval::S
  samplevalues::T

  function SamplePath(timeinterval::S, samplevalues::T) where
      {S<:AbstractVector{Float64}, T<:AbstractVector{Float64}}
    length(timeinterval) == length(samplevalues) ||
      error("Length of timeinterval should be equal to the length of the samplevalues vector")
    isincreasing(timeinterval) ||
      error("Timeinterval should be increasing")
    new{S,T}(timeinterval, samplevalues)
  end
end

#constructor
SamplePath(timeinterval::T, f::Function) where T<:AbstractVector{Float64} =
SamplePath(timeinterval, f.(timeinterval))

"""
    step(X::SamplePathRange)

Returns the step of the timeinterval field; the discretisation step of the
sample path.

# Example

```julia
X = SamplePathRange(0.:0.1:2.0, x->x^2)
length(X)
```
"""
step(X::SamplePathRange)=step(X.timeinterval)

# Returns the length of timeinterval == samplevalues vectors, not the endtime!
# TO DO: Answers to examples
"""
    length(X::AbstractSamplePath)

Returns the length of the vector timeinterval == length vector samplevalues.

# Examples
```julia
X = SamplePath([0.,1.,2.], [3.,5., -1.])
length(X)
```

```julia
X = SamplePathRange(0.:0.1:2.0, x->x^2)
length(X)
```
"""
length(X::AbstractSamplePath)=length(X.timeinterval)
