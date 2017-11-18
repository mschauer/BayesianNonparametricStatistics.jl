# basisfunctions.jl

"""
    fourier(k::Int)

Implements the Fourier basis of functions ϕ_k, defined, if k is odd, by
ϕ_k(x)=sqrt(2)sin((k+1)π*x) and if k is even, by ϕ_k(x)=sqrt(2)cos(k*x).

# Examples

```julia
using PyPlot
clf()
x = -1.0:0.01:2.0
y = fourier(3).(x)
plot(x,y)
```
"""
function fourier(k::Int)
  if k < 1
    error("k should be positive")
  end
  sqrttwo = sqrt(2.0)
  if k % 2 == 1
    return function(x::Float64)
      return sqrttwo * sinpi((k+1)*x)
    end
  else
    return function(x::Float64)
      return sqrttwo * cospi(k*x)
    end
  end
end

"""
    faberschauderone(x::Float64)

Implements the first Faber-Schauder function defined by 1-2x for 0≤x≤1/2 and
-1+2x for 0.5≤x≤1, and is 1-periodic extended to all x∈R.

See also: faberschauder

#Example
```julia
x=-2.:0.001:2
y=faberschauderone.(x)
using PyPlot
plot(x,y)
```
"""
function faberschauderone(x::Float64)
    y = mod(x, 1.0)
    if 0≤y≤0.5
        return 1.0 - 2*y
    else
        return -1.0 + 2*y
    end
end

"""
    faberschauder(j::Int, k::Int)

Implements the k-th Faber-Schauder function of level j. Here, j≥0 and 1≤k≤2^j.
It is a one periodic function and defined on [0,1] by 2^(j+1)(x-(k-1)2^j) on
(k-1)2^j≤x≤(2k-1)2^(j+1) and 1 - 2^(j+1)(x-(2k-1)2^(j+1)) on
[(2k-1)2^(j+1), k2^j] and zero outside these intervals.

#Example
```julia
j=3
x=-2.:0.001:2.
using PyPlot
clf()
for k in 1:2^j
    y = faberschauder(j,k).(x)
    plot(x,y)
end
```
"""
function faberschauder(j::Int,k::Int)
  if j < 0
    error("j should be a nonnegative integer.")
  elseif !(1≤k≤2^j)
    error("k should be an integer between 1 and 2^j.")
  end
  j_float = Float64(j)
  return function(x::Float64)
    y = mod(x,1.0)
    if y ≤ (k-1)*2.0^(-j_float) || y ≥ k*2.0^(-j_float)
      return 0.0
    elseif ((k-1)*2.0^(-j_float) < y ≤ (2k-1)*2.0^(-j_float-1.0))
      return 2^(j+1)*(y-(k-1)*2.0^(-j_float))
    else
      return 1.0 - 2^(j+1)*(y-(2k-1)*2.0^(-j_float-1.0))
    end
  end
end
