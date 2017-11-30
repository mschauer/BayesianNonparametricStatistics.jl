# samplepath.jl

"""
    abstract type AbstractSamplePath <: Any

subtype: SamplePath.

Its subtype implements a continuous samplepath.
"""
abstract type AbstractSamplePath end

"""
    isincreasing(x::AbstractVector{T})::Bool where T <: Number
    isincreasing(x::Range{T})::Bool where T <: Number

Tests whether a vector of numbers is strictly increasing. Is internal to
NonparametricBayesForDiffusions.
"""
function isincreasing(x::AbstractVector{T}) where T <: Number
  for i in 1:length(x)-1
    x[i+1] <= x[i] && return false
  end
  return true
end

isincreasing(x::Range{T}) where T <: Number = step(x) > 0

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

 # Examples

```julia
using PyPlot
clf()
t = 0.0:0.1:2.0
v = sinpi.(t)
X = SamplePath(t, v)
plot(X)
```
"""
struct SamplePath{S<:AbstractVector{Float64},
        T<:AbstractVector{Float64}} <: AbstractSamplePath
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
SamplePath(timeinterval, f::Function) = SamplePath(timeinterval, f.(timeinterval))

step(X::SamplePath{S}) where S<:Range{<:Number} = step(X.timeinterval)

# Returns the length of timeinterval == samplevalues vectors, not the endtime!
# TO DO: Answers to examples
"""
    length(X::SamplePath)

Returns the length of the vector timeinterval == length vector samplevalues.

# Examples
```julia
X = SamplePath([0.,1.,2.], [3.,5., -1.])
length(X)
```

```julia
X = SamplePath(0.0:0.1:2π, sin)
length(X) == length(0.0:0.1:2π)
```
"""
length(X::SamplePath)=length(X.timeinterval)
