using ITensors
using ITensorNetworkAD
using OptimKit
using Zygote
using Test

@testset "optimization" begin
  @testset "Energy minimization" begin
    N = 6
    s = siteinds("S=1/2", N; conserve_qns=true)
    os = OpSum()
    for n in 1:(N - 1)
      os .+= 0.5, "S+", n, "S-", n + 1
      os .+= 0.5, "S-", n, "S+", n + 1
      os .+= "Sz", n, "Sz", n + 1
    end
    Hmpo = MPO(os, s)
    ψ₀mps = randomMPS(s, n -> isodd(n) ? "↑" : "↓")
    H = prod(Hmpo)
    ψ₀ = prod(ψ₀mps)
    # The Rayleigh quotient to minimize
    function E(H::ITensor, ψ::ITensor)
      ψdag = dag(ψ)
      return (ψdag' * H * ψ)[] / (ψdag * ψ)[]
    end
    E(ψ::ITensor) = E(H, ψ)
    ∇E(ψ::ITensor) = E'(ψ)
    fg(ψ::ITensor) = (E(ψ), ∇E(ψ))
    linesearch = HagerZhangLineSearch(;
      c₁=0.1, c₂=0.9, ϵ=1e-6, θ=1 / 2, γ=2 / 3, ρ=5.0, verbosity=0
    )
    algorithm = LBFGS(3; maxiter=30, gradtol=1e-8, linesearch=linesearch)
    ψ, fψ, gψ, numfg, normgradhistory = optimize(fg, ψ₀, algorithm)
    D, _ = eigen(H; ishermitian=true)
    @test E(H, ψ) < E(H, ψ₀)
    @test E(H, ψ) ≈ minimum(D)
  end
  @testset "Energy minimization (MPS)" begin
    N = 4
    χ = 4
    s = siteinds("S=1/2", N; conserve_qns=true)
    os = OpSum()
    for n in 1:(N - 1)
      os .+= 0.5, "S+", n, "S-", n + 1
      os .+= 0.5, "S-", n, "S+", n + 1
      os .+= "Sz", n, "Sz", n + 1
    end
    Hmpo = MPO(os, s)
    ψ₀mps = randomMPS(s, n -> isodd(n) ? "↑" : "↓"; linkdims=χ)
    H = ITensors.data(Hmpo)
    ψ₀ = ITensors.data(ψ₀mps)
    # The Rayleigh quotient to minimize
    function E(H::Vector{ITensor}, ψ::Vector{ITensor})
      N = length(ψ)
      ψdag = dag.(addtags.(ψ, "bra"; tags="Link"))
      ψ′dag = prime.(ψdag)
      e = ITensor(1.0)
      for n in 1:N
        e = e * ψ′dag[n] * H[n] * ψ[n]
      end
      norm = ITensor(1.0)
      for n in 1:N
        norm = norm * ψdag[n] * ψ[n]
      end
      return e[] / norm[]
    end
    E(ψ) = E(H, ψ)
    ∇E(ψ) = E'(ψ)
    fg(ψ) = (E(ψ), ∇E(ψ))
    linesearch = HagerZhangLineSearch(; c₁=0.1, c₂=0.9, ϵ=1e-6, θ=1 / 2, γ=2 / 3, ρ=5.0)
    algorithm = LBFGS(5; maxiter=100, gradtol=1e-8, linesearch=linesearch, verbosity=0)
    ψ, fψ, gψ, numfg, normgradhistory = optimize(fg, ψ₀, algorithm)
    sweeps = Sweeps(5)
    setmaxdim!(sweeps, χ)
    fψmps, ψmps = dmrg(Hmpo, ψ₀mps, sweeps; outputlevel=0)
    time_Eψ = @elapsed E(ψ)
    time_∇Eψ = @elapsed E'(ψ)
    @test E(H, ψ) ≈ inner(ψmps, Hmpo, ψmps) / inner(ψmps, ψmps)
  end
end