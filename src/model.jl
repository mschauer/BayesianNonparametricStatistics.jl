# model.jl

"""
    abstract type AbstractModel end

Implements a statistical model. Has so far only SDEModel as subtype.
"""
abstract type AbstractModel end

"""
    type SDEModel{T<:Union{Float64, <:Function}} <: AbstractModel
    σ::T
    beginvalue::Float64
    end

Implements an SDE model with begin value beginvalue and σ.

# Warning
It is assumed that
for every b under consideration, the laws of dX_t=b(X_t)dt+σ(X_t)dW_t are
equivalent.

# Example

```julia
SDEModel(1.0, 0.0)
# Implements the model dX_t=b(X_t)dt+σ(X_t)dW_t, with X_0=0.0.
#
SDEModel(identity, 0.0)
# Implements the model dX_t=b(X_t)+X_tdW_t, with X_0=0.0.
```
"""
type SDEModel{T<:Union{Float64, <:Function}} <: AbstractModel
    σ::T
    beginvalue::Float64
end
