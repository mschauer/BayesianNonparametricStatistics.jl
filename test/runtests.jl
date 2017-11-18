using Base.Test

using BayesianNonparametricStatistics, Distributions, PDMats

srand(123)

@testset "BayesianNonparametricStatistics package" begin
    @testset "model.jl" begin
        @test Set(subtypes(AbstractModel)) == Set([SDEModel])
        @test supertype(AbstractModel) == Any
        for σ in -10.0:0.1:10.0
            for x in -10.0:0.1:10.0
                M = SDEModel(σ, x)
                @test M.σ == σ
                @test M.beginvalue == x
            end
        end
    end

    @testset "samplepath.jl" begin
        @test Set(subtypes(AbstractSamplePath)) ==
            Set([SamplePath, SamplePathRange])

        @test supertype(AbstractSamplePath) == Any

        X = SamplePathRange(0.0:0.1:1.0, 1:11)

        @test X.timeinterval == 0.0:0.1:1.0
        @test X.samplevalues == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]

        Y = SamplePathRange(0.0:0.1:1.0, x->2*x)
        @test Y.timeinterval == 0.0:0.1:1.0
        @test Y.samplevalues == collect(0.0:0.2:2.0)

        @test BayesianNonparametricStatistics.isincreasing(0:0.1:1)
        @test !BayesianNonparametricStatistics.isincreasing(0:-0.1:-1.0)
        @test !BayesianNonparametricStatistics.isincreasing([0.0, 0.1, 0.2, 0.1, 0.3, 0.4])
        @test !BayesianNonparametricStatistics.isincreasing([0.0, -0.1, 0.0, 0.2, 0.3])
        @test !BayesianNonparametricStatistics.isincreasing([1, 2, 3, 4, -10, 11, 12])
        @test BayesianNonparametricStatistics.isincreasing([1, 2, 3, 4, 500, 600, 2000, 10000])
        @test BayesianNonparametricStatistics.isincreasing([1., 2., 2.00000001, 2.00000002])
        @test !BayesianNonparametricStatistics.isincreasing([2.,2.,2.,2.,2.])
        @test !BayesianNonparametricStatistics.isincreasing([2.,2., 3.0, 3.0001])

        Z = SamplePath([1., 2.], [1.1, 2.1])
        Z.timeinterval == [1., 2.]
        Z.samplevalues == [1.1, 2.1]

        W = SamplePathRange(0.0:0.1:1.0, sin)
        @test W.timeinterval == 0.0:0.1:1.0
        @test W.samplevalues == sin.(0.0:0.1:1.0)

        @test step(Y) == 0.1

        @test length(Y) == 11
        @test length(Z) == 2

        X = SamplePath([1.0, 3.0, 5.0, 9.0], x->x^2)
        @test X.timeinterval == [1.0, 3.0, 5.0, 9.0]
        @test X.samplevalues == [1.0, 9.0, 25.0, 81.0]

        X = SamplePath(collect(-2.0:0.01:2.0), sin)
        X.timeinterval == collect(-2.0:0.01:2.0)
        X.samplevalues == sin.(collect(-2.0:0.01:2.0))

        X = SamplePath(-2.0:0.01:2.0, sin)
        X.timeinterval == collect(-2.0:0.01:2.0)
        X.samplevalues == sin.(collect(-2.0:0.01:2.0))
    end

    @testset "basisfunctions.jl" begin
        # The Fourier functions are orthogonal.
        n = 3
        X = SamplePathRange(0.:.01:1.0, 0.:.01:1.0)
        for k in 1:n-1
            for l in k+1:n
                @test abs(BayesianNonparametricStatistics.calculateLebesgueintegral(x->fourier(k)(x)*fourier(l)(x), X))<0.1
            end
        end
        # And the square integrates to one.
        for k in 1:n
            @test abs(BayesianNonparametricStatistics.calculateLebesgueintegral(x->fourier(k)(x)^2,X)-1.) <0.1
        end

        # They are one-periodic.
        x = 0.0:0.1:0.9
        for k in 1:n
            for y in x
                value = fourier(k)(y)
                for m in -10.0:10.0
                    valuetwo = fourier(k)(y+m)
                    @test value ≈ valuetwo
                end
            end
        end

        # They have minimum -sqrt(2) and maximum sqrt(2).
        x = 0.0:0.01:1.0
        for k in 1:n
            values = map(fourier(k), x)
            @test abs(minimum(values)+sqrt(2.0)) < 0.1
            @test abs(maximum(values)-sqrt(2.0)) < 0.1
        end

        # Below we test Faber-Schauder functions on their mathematical properities, up
        # to level n, defined below.
        n = 2
        X=SamplePathRange(0.:0.001:1.0,0.:0.001:1.0)

        # Faber-Schauder functions are level-wise orthogonal, for levels j≥1.
        for j in 1:n
            for k in 1:2^j-1
                for l in k+1:2^j
                    @test abs(BayesianNonparametricStatistics.calculateLebesgueintegral(x->faberschauder(j,k)(x)*faberschauder(j,l)(x), X))<0.01
                end
            end
        end

        # They integrate to 2^(-j-1)
        @test abs(BayesianNonparametricStatistics.calculateLebesgueintegral(faberschauderone, X)-0.5) <0.01
        for j in 1:n
            for k in 1:2^j
                @test abs(BayesianNonparametricStatistics.calculateLebesgueintegral(faberschauder(j,k), X)- 2.0^(-j-1))<0.01
            end
        end

        # They have minimum 0 and maximum 1.
        x = 0.0:0.001:1.0
        y = map(faberschauderone, x)
        @test abs(minimum(y)) < 0.01
        @test abs(maximum(y)-1.0) < 0.01

        for j in 1:n
            for k in 1:2^j
                y = map(faberschauder(j,k), x)
                @test abs(minimum(y)) < 0.01
                @test abs(maximum(y)-1.0) < 0.01
            end
        end

        # They are one-periodic.
        x = 0.0:0.01:0.99
        for y in x
            value = faberschauderone(y)
            for k in -10.0:10.0
                valuetwo = faberschauderone(y+k)
                @test value≈valuetwo
            end
        end

        for j in 0:n
            for k in 1:2^j
                for y in x
                    value = faberschauder(j,k)(y)
                    for m in -10.:10.
                        valuetwo = faberschauder(j,k)(y+m)
                        @test value≈valuetwo
                    end
                end
            end
        end
        # End of tests for mathematical properities of Faber-Schauder functions.

    end

    @testset "SDE.jl" begin
        @test Set(subtypes(AbstractSDE)) == Set([SDE, SDEWithConstantVariance])

        @test supertype(AbstractSDE) == Any

        # The following should represent a Brownian motion.
        sde_SDE_type = SDE(x->0, x->1, 0, 1, 0.001)

        lengthvector = 1000
        x = Vector{Float64}(lengthvector)

        for k in 1:lengthvector
            X = rand(sde_SDE_type)
            x[k] = X.samplevalues[end]
        end

        @test abs(mean(x)) < 0.1
        @test 0.9 < var(x) < 1.1

        X = rand(sde_SDE_type)

        @test length(X) == length(0.0:sde_SDE_type.dt:sde_SDE_type.endtime)
        @test X.timeinterval == 0.0:sde_SDE_type.dt:sde_SDE_type.endtime

        geometricbrownianmotion = SDE(x->0.5*x, identity, 1, 1, 0.001)

        for k in 1:lengthvector
            X = rand(geometricbrownianmotion)
            x[k] = X.samplevalues[end]
        end

        @test abs(e^(0.5)-mean(x)) < 0.1

        X = rand(geometricbrownianmotion)

        @test length(X) == length(0.0:geometricbrownianmotion.dt:geometricbrownianmotion.endtime)
        @test X.timeinterval ==  0.0:geometricbrownianmotion.dt:geometricbrownianmotion.endtime



        sde_SDEWithConstantVariance_type = SDEWithConstantVariance(x->0, 1, 0, 1, 0.001)

        for k in 1:lengthvector
            X = rand(sde_SDEWithConstantVariance_type)
            x[k] = X.samplevalues[end]
        end

        @test abs(mean(x)) < 0.1
        @test 0.9 < var(x) < 1.1

        X = rand(sde_SDEWithConstantVariance_type)
        @test X.timeinterval == 0.0:sde_SDEWithConstantVariance_type.dt:sde_SDEWithConstantVariance_type.endtime
        @test length(X) == length(0.0:sde_SDEWithConstantVariance_type.dt:sde_SDEWithConstantVariance_type.endtime)

        ornsteinuhlenbeckprocess = SDEWithConstantVariance(x-> -x, 1, e, 1, 0.001)

        for k in 1:lengthvector
            X = rand(ornsteinuhlenbeckprocess)
            x[k] = X.samplevalues[end]
        end

        @test abs(mean(x) - 1) < 0.1

        X = rand(ornsteinuhlenbeckprocess)

        @test X.timeinterval == 0.0:ornsteinuhlenbeckprocess.dt:ornsteinuhlenbeckprocess.endtime
        @test length(X) == length(0.0:ornsteinuhlenbeckprocess.dt:ornsteinuhlenbeckprocess.endtime)
    end

    @testset "gaussianprocess.jl" begin
        @test Set(subtypes(AbstractGaussianProcess)) ==
        Set([GaussianProcess, FaberSchauderExpansionWithGaussianCoefficients])

        @test supertype(AbstractGaussianProcess) == Any

        X = GaussianProcess([sin, cos], MvNormal([1.0, 1.0]))

        @test X.basis == [sin, cos]

        @test typeof(X.distribution) == MvNormal{Float64,PDMats.PDiagMat{Float64,Array{Float64,1}},Distributions.ZeroVector{Float64}}
        @test X.distribution.Σ.diag == [1.0, 1.0]
        @test mean(X.distribution) == [0.0,0.0]
        @test length(X) == 2
        @test typeof(rand(X)) <: Function

        n = 10000
        a = sinpi(1/4)
        x = Vector{Float64}(n)

        for k in 1:n
            f = rand(X)
            x[k] = f(π/4)
        end

        @test abs(mean(x)) < 0.01
        @test abs(var(x)-2*a^2) < 0.1

        X = GaussianProcess([sinpi, cospi], MvNormal([1.0,1.0], [1.0,1.0]))

        for k in 1:n
            f = rand(X)
            x[k] = f(1/4)
        end

        @test abs(mean(x)-2*a)<0.01
        @test abs(var(x)-2*a^2)<0.1

        x = 0.0:0.1:1.0
        y = sumoffunctions([sin, cos], [1., 1.]).(x)
        z = map(x-> sin(x)+cos(x), x)
        @test y ≈ z

        # Test for two cases whether calculatedependencystructurefaberschauderbasis
        # calculates the right dependency structure of the Faber-Schauder basis.
        A = [
        true true true true true true true true;
        true true true true true true true true;
        true true true false true true false false;
        true true false true false false true true;
        true true true false true false false false;
        true true true false false true false false;
        true true false true false false true false;
        true true false true false false false true
        ]

        A = Symmetric(A)

        @test calculatedependencystructurefaberschauderbasis(2) == A

        A = [
        true true true true true true true true true true true true true true true true; #1
        true true true true true true true true true true true true true true true true; #2
        true true true false true true false false true true true true false false false false; #3
        true true false true false false true true false false false false true true true true; #4
        true true true false true false false false true true false false false false false false; #5
        true true true false false true false false false false true true false false false false; #6
        true true false true false false true false false false false false true true false false; #7
        true true false true false false false true false false false false false false true true; #8
        true true true false true false false false true false false false false false false false; #9
        true true true false true false false false false true false false false false false false; #10
        true true true false false true false false false false true false false false false false; #11
        true true true false false true false false false false false true false false false false; #12
        true true false true false false true false false false false false true false false false; #13
        true true false true false false true false false false false false false true false false; #14
        true true false true false false false true false false false false false false true false; #15
        true true false true false false false true false false false false false false false true #16
        ]

        A = Symmetric(A)

        @test calculatedependencystructurefaberschauderbasis(3) == A

        A = PDSparseMat(sparse([1.0 0.0; 0.0 1.0]))

        distribution = MvNormalCanon(A)

        Π = FaberSchauderExpansionWithGaussianCoefficients(0,distribution)
        @test length(Π) == 2

        n = 10000
        x = Vector{Float64}(n)

        for k in 1:n
            f = rand(Π)
            x[k] = f(0.5)
        end

        @test abs(mean(x))<0.1
        @test abs(var(x)-1.0) <0.1

        for k in 1:n
            f = rand(Π)
            x[k] = f(0.0)
        end

        @test abs(mean(x))<0.1
        @test abs(var(x)-1.0) <0.1

        for k in 1:n
            f = rand(Π)
            x[k] = f(1.0)
        end

        @test abs(mean(x))<0.1
        @test abs(var(x)-1.0) <0.1

        distribution = MvNormalCanon([1.0, -2.0],[1.0, -2.0], A)
        Π = FaberSchauderExpansionWithGaussianCoefficients(0,distribution)

        @test length(Π) == 2

        n = 10000
        x = Vector{Float64}(n)

        for k in 1:n
            f = rand(Π)
            x[k] = f(0.5)
        end

        @test abs(mean(x)+2.0)<0.1
        @test abs(var(x)-1.0) <0.1

        for k in 1:n
            f = rand(Π)
            x[k] = f(0.0)
        end

        @test abs(mean(x)-1.0)<0.1
        @test abs(var(x)-1.0) <0.1

        for k in 1:n
            f = rand(Π)
            x[k] = f(1.0)
        end

        @test abs(mean(x)-1.0)<0.1
        @test abs(var(x)-1.0) <0.1

        Π = FaberSchauderExpansionWithGaussianCoefficients([1.0, 0.5])

        n = 10000
        x = Vector{Float64}(n)
        for k in 1:n
            f = rand(Π)
            x[k] = f(0.25)
        end

        @test abs(mean(x)) < 0.1
        @test abs(var(x) - 2.5) < 0.1

        A = PDSparseMat(sparse(eye(4)))

        distribution = MvNormalCanon([1.0,1.0,1.0,1.0], [1.0,1.0,1.0,1.0], A)

        Π = FaberSchauderExpansionWithGaussianCoefficients(1, distribution)

        n = 10000
        x = Vector{Float64}(n)
        for k in 1:n
            f = rand(Π)
            x[k] = f(0.25)
        end

        @test abs(mean(x)-2.0) < 0.1
        @test abs(var(x)-1.5) < 0.1

        X = SamplePathRange(0.0:10.0, [1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,11.])
        @test BayesianNonparametricStatistics.calculatestochasticintegral(identity, X) ≈ 55.0
        @test BayesianNonparametricStatistics.calculateLebesgueintegral(x->1., X) ≈ 10.0

        X = SamplePath(collect(0.0:10.0), [1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,11.])
        @test BayesianNonparametricStatistics.calculatestochasticintegral(identity, X) ≈ 55.0
        @test BayesianNonparametricStatistics.calculateLebesgueintegral(x->1., X) ≈ 10.0

        α = 0.5
        Π = FaberSchauderExpansionWithGaussianCoefficients([2^(α*j) for j in 1:5])

        sde = SDEWithConstantVariance(x->0.0, 1.0, 0.0, 10000.0, 0.01)
        X = rand(sde)

        M = SDEModel(1.0, 0.0)

        postΠ = calculateposterior(Π, X, M)

        n = 1000
        y = 0.0:0.01:1.0
        x = Array{Float64}(length(y),n)
        for k in 1:n
            f = rand(postΠ)
            x[:,k] = f.(y)
        end

        @test maximum(abs.(mean(x,1))) < 0.1

        M = SDEModel(1.0, 0.0)

        Π = GaussianProcess([fourier(k) for k in 1:40], MvNormal([k^(-1.0) for k in 1:40]))

        postΠ = calculateposterior(Π, X, M)

        n = 1000
        y = 0.0:0.01:1.0
        x = Array{Float64}(length(y),n)
        for k in 1:n
            f = rand(postΠ)
            x[:,k] = f.(y)
        end

        @test maximum(abs.(mean(x,1))) < 0.1

        M = SDEModel(x->1.0, 0.0)

        Π = GaussianProcess([fourier(k) for k in 1:40], MvNormal([k^(-1.0) for k in 1:40]))

        postΠ = calculateposterior(Π, X, M)

        n = 1000
        y = 0.0:0.01:1.0
        x = Array{Float64}(length(y),n)
        for k in 1:n
            f = rand(postΠ)
            x[:,k] = f.(y)
        end

        @test maximum(abs.(mean(x,1))) < 0.1

        M = SDEModel(x->1.0, 0.0)

        Π = FaberSchauderExpansionWithGaussianCoefficients([2^(α*j) for j in 1:5])

        postΠ = calculateposterior(Π, X, M)

        n = 1000
        y = 0.0:0.01:1.0
        x = Array{Float64}(length(y),n)
        for k in 1:n
            f = rand(postΠ)
            x[:,k] = f.(y)
        end

        @test maximum(abs.(mean(x,1))) < 0.1
    end
end
