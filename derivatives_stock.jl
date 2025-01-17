using LinearAlgebra

include("modules/stock_module.jl")
include("modules/spectral_stuff.jl")
px = println

function create_derivatives_stock(H,ϕ0,E0,K;debug=false,x0=1)
    # 1) Define the different kinds of recursive objects we will need
    kinds = ["H","E","K","h","q","Q","Φ","ϕ","N","x"]

    # 2) Define default parameters values
    p = Dict()
    p["N"] = length(H)-1 # H^N max, after they vanish
    p["H"] = H
    p["ϕ0"] = ϕ0
    p["E0"] = E0
    p["K"] = K
    p["x0"] = x0
    p["dim"] = size(H[1],1)

    # 3) Define how to compute the quantities
    function compute_H(n,S) # H^n
        if n>S.p["N"]
            return 0*I
        end
        S.p["H"][n+1]
    end

    function compute_E(c,S) # E^n_μ
        (n,μ) = c
        if n==0
            return p["E0"][μ+1]
        end
        ϕ0 = X("ϕ",(0,μ),S)
        ϕ0⋅ (X("q",c,S)*ϕ0)
        # sca(ϕ0, X("q",c,S)*ϕ0)
    end

    compute_K(μ,S) = S.p["K"][μ+1] # K_μ

    compute_h(c,S) = X("H",c[1],S)-X("E",c,S)*I # h^n_μ

    function compute_q(c,S)
        (n,μ) = c
        Hn = X("H",n,S)
        if n==0
            px("PROBLEM IN COMPUTE q")
        elseif n==1
            return Hn
        end
        Hn + sum([X("h",(n-s,μ),S)*X("K",μ,S)*X("Q",(s,μ),S) for s=1:n-1])
    end

    compute_Q(c,S) = X("q",c,S) - X("E",c,S)*I

    function compute_Φ(c,S)
        (n,μ) = c
        ϕ0 = X("ϕ",(0,c[2]),S)
        if n==0
            return ϕ0
        end
        X("K",μ,S)*X("Q",c,S)*ϕ0
    end

    function compute_ϕ(c,S)
        (n,μ) = c
        # px("compute ϕ0")
        if n==0
            # px("return ϕ0")
            return S.p["ϕ0"][μ+1]
        elseif n==1
            return X("Φ",c,S)
        end
        X("Φ",c,S) - sum([X("N",(n-s,μ),S) * X("ϕ",(s,μ),S) for s=0:n-2])
    end

    function compute_N(c,S)
        (n,μ) = c
        if n==0
            return 1
        elseif n==1
            return 0
        end
        # 0.5*sum([sca(X("Φ",(n-s,μ),S) , X("Φ",(s,μ),S)) - X("N",(n-s,μ),S)*X("N",(s,μ),S) for s=1:n-1])
        0.5*sum([X("Φ",(n-s,μ),S) ⋅ X("Φ",(s,μ),S) - X("N",(n-s,μ),S)*X("N",(s,μ),S) for s=1:n-1])
    end

    # x^n, model for understanding the divergence of energy derivatives
    function compute_x(n,S)
        if n in [0,1]
            return S.p["x0"]
        end
        x = Float64(0)
        N = Int(floor(n/2))
        for s=1:N-1
            x += 2*X("x",n-s,S)*X("x",s,S)
        end
        if n%2==0
            x += X("x",N,S)*X("x",N,S)
        else
            x += 2*X("x",N,S)*X("x",N+1,S)
        end
        x
    end

    # 4) Gather computation functions, same order than kinds list
    computation_functions = [compute_H,compute_E,compute_K,compute_h,compute_q,compute_Q,compute_Φ,compute_ϕ,compute_N,compute_x]
    
    # 5) Create stock
    S = Stock(kinds,computation_functions,p;debug=debug)
    S
end

function stock_at_H(Hs)
    # Solves exact at λ=0
    E0, ϕ0 = eigenmodes(Hs[1])
    # Computes the pseudo-inverses
    K = all_pseudo_inverses(E0,ϕ0)
    # Creates stock object
    S = create_derivatives_stock(Hs,ϕ0,E0,K;debug=false)
    S
end

# Estimate norms of derivatives for E and ϕ
function estimate_norms(ℓ,S)
    nNs = []
    ks = (1:ℓ)
    for k in ks
        nE, nϕ, nQ, nx = abs(X("E",(k,0),S)), norm(X("ϕ",(k,0),S)), norm(X("Q",(k,0),S)), X("x",k,S)
        push!(nNs,(nE,nϕ,nQ,nx))
        px("k=",k," |E^k_0|=",nE," ||ϕ||=",nϕ," ||Q||=",nQ," x=",nx," <ϕ1,ϕk> ",abs(X("ϕ",(1,0),S)⋅X("ϕ",(k,0),S)))
    end
    ℓ_start = Int(floor(ℓ/2))
    # a_of(q) = exp(log_slope(nNs[ℓ][q],nNs[ℓ_start][q],ℓ,ℓ_start))
    p = 3/2
    # a_of(q) = (nNs[ℓ][q]/nNs[ℓ-1][q]) * (1+ 1/(ℓ-1))^p # p known
    a_of(q) = (nNs[ℓ][q]/nNs[ℓ-1][q]) # p not known
    p_of(q) = -log(nNs[ℓ_start][q]/(nNs[ℓ_start-1][q]*a_of(q)))/log(1+1/ℓ)
    c_of(q) = nNs[ℓ_start][q]*(ℓ_start^p_of(q))/((a_of(q))^ℓ_start)
    px("Quantities = c a^n / n^p")
    px("a = for --- E ",a_of(1)," ϕ ",a_of(2)," Q ",a_of(3)," x ",a_of(4))
    px("c = for --- E ",c_of(1)," ϕ ",c_of(2)," Q ",c_of(3)," x ",c_of(4))
    px("p = for --- E ",p_of(1)," ϕ ",p_of(2)," Q ",p_of(3)," x ",p_of(4))

    a = a_of(2)
    c = c_of(2)
    p = p_of(2)
    Sn(n) = c*(a^n)/(n^p)
    model = [Sn(n) for n in ks]

    f = Figure()
    ax = Axis(f[1, 1], yscale=Makie.log10)#xscale=Makie.log10, 
    nqs(j) = [nNs[k][j] for k in ks]
    lines!(ax, ks, nqs(1), color="black")
    lines!(ax, ks, nqs(2), color="red")
    lines!(ax, ks, nqs(3), color="blue")
    lines!(ax, ks, model, color="green")
    return f
end

############# Test
herm(M) = (M+M')/2

function random_Hn(N)
    Hs = [copy(herm(randn(N,N))) for k=0:2]
    # px("NORM ",norm(Hs[1].-Hs[2]))
    Hs
end

function get_λs(n_start,n_end)
    resolution = 1
    λs = logrange(10^float(n_end),10^float(n_start),resolution)
    λs
end

Hλ(λ,Hs) = sum((λ^k)*Hs[k+1] for k=0:length(Hs)-1)

function get_derivatives_approx(ϕ0,dλ,Hs)
    Hε = Hλ(dλ,Hs)
    Hε2 = Hλ(-dλ,Hs)
    Eεs, ϕεs = eigenmodes(Hε)
    Eεs2, ϕεs2 = eigenmodes(Hε2)
    ϕ1_approx = (ϕεs[1] - ϕ0)/dλ
    ϕ2_approx = (ϕεs[1] - 2*ϕ0 + ϕεs2[1])/(2*dλ^2)
    ϕ1_approx, ϕ2_approx
end

dist(a,b) = norm(a-b)/norm(a)

function test_derivatives()
    μ = 0
    Hs = random_Hn(5)
    S = stock_at_H(Hs)

    dλ = 1e-4
    ϕ0 = X("ϕ",(0,μ),S)
    ϕ1_approx, ϕ2_approx = get_derivatives_approx(ϕ0,dλ,Hs)
    px("here")

    ϕ1_exact = X("ϕ",(1,μ),S)
    ϕ2_exact = X("ϕ",(2,μ),S)

    px("Compare ",dist(ϕ1_approx,ϕ1_exact)," masses ",norm(ϕ1_approx)," and ",norm(ϕ1_exact))
    px("Compare ",dist(ϕ2_approx,ϕ2_exact)," masses ",norm(ϕ2_approx)," and ",norm(ϕ2_exact))
end

# test_derivatives()
