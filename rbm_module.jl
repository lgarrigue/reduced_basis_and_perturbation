include("modules/spectral_stuff.jl")
using Random

mutable struct EigenRBM
    V # list of vectors being in the variational space. It's an orthonormal basis
    ℓ
    ν
    dim
    generalized # true iff we use generalized eigenmode problems
    function EigenRBM(ℓ,ν;gen=false) # gen = true does not work
        R = new()
        R.V = []
        R.ℓ, R.ν = ℓ, ν
        R.dim = -1
        R.generalized = gen
        R
    end
end

function gram_schmidt_one_vector(v,V)
    if length(V)==0
        return v/norm(v)
    end
    u = v/norm(v)
    u_perp = u
    u_perp -= sum((vk⋅u)*vk for vk in V) 
    # px("norm of added vector ",norm(u_perp))
    u_perp /= norm(u_perp)
    u_perp
end

function add_vector(v,R)
    if !R.generalized
        if length(R.V)==0
            R.dim = length(v)
        end
        @assert length(v) == R.dim
        u_perp = gram_schmidt_one_vector(v,R.V)
        push!(R.V,u_perp)
    else
        push!(R.V,v)
    end
end

# do a Gram-Schmidt to help orthogonalize
function do_Gram_Schmidt(R;n=1) 
    for j=1:n
        Rs = []
        s = shuffle((1:nV(R)))
        nR = [R.V[j] for j in s]
        for i=1:nV(R)
            u_perp = gram_schmidt_one_vector(nR[i],Rs)
            push!(Rs,u_perp)
        end
        R.V = Rs
    end
end

nV(R) = length(R.V) # number of vectors in V

function add_vectors(V,R) # adds all the vectors of V
    for j=1:length(V)
        add_vector(V[j],R)
    end
end

function rbm_approx_eigenmodes(H,R) # H is a Hamiltonian, gives the approximated eigenfunctions, in the basis R.V
    N = nV(R)
    Hrbm = [R.V[i]⋅(H*R.V[j]) for i=1:N, j=1:N]
    B = [R.V[i]⋅R.V[j] for i=1:N, j=1:N]
    # Hrbm = zeros(ComplexF64,(N,N))
    # for i=1:N, j=1:N
        # Hrbm[i,j] = R.V[i]⋅(H*R.V[j])
    # end
    # Hrbm = Hermitian(Hrbm)
    # px("Complex part of Hrbm ",norm(imag.(Hrbm)))
    E, C = R.generalized ? gen_eigenmodes(Hrbm,B) : eigenmodes(Hrbm)
    Ψ = []
    for j=1:N
        φ = sum(C[j][k]*R.V[k] for k=1:N)
        if !R.generalized
            @assert abs(norm(φ)-1)<1e-6
        end
        push!(Ψ,φ)
    end
    E, Ψ
end

function add_random_vectors(n,R) # adds n random vectors
    for k=1:n
        v = copy(randn(R.dim))
        add_vector(v,R)
    end
end

function init_R_level_μ(R,μ,S)
    for k=0:R.ℓ
        ϕk = X("ϕ",(k,μ),S)
        # if α==3
            # v = copy(randn(R.dim))
            # add_vector(v,R)
        # else
        # if α in [4,5,6] || k==0
        add_vector(ϕk,R)
        # else
            # for j=1:10
                # v = copy(randn(length(ϕk)))
                # add_vector(v,R)
            # end
        # end
    end
end

function init_R(R,S)
    for α=0:R.ν
        init_R_level_μ(R,α,S)
    end
end

function test_R(S,R)
    test_orthonormal_basis(R)
    @assert S.p["dim"] > nV(R)+5
    # px("R has ",nV(R)," vectors")
end

### Test

# Tests that R.V is an orthonormal basis

function test_orthonormal_basis(R)
    if !R.generalized
        N = nV(R)
        S = [R.V[i]⋅R.V[j] for i=1:N, j=1:N]
        err = norm(S-I)
        # px("test orthonormal basis ",err," N=",nV(R))
        @assert err<1e-10
    end
end

# produces the projector over Span(R.V)
create_full_projector(R) = sum(R.V[k]*R.V[k]' for k=1:length(R.V))
create_projector(R) = [R.V[j][i] for i=1:length(R.V[1]), j=1:length(R.V)]
