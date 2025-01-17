# Implementation of low-energy states, density matrices, pseudo-inverses, resolvents

using LinearAlgebra, Optim
using MKL
px = println

################################## Low-energy states of a matrix

function eigenmodes(H)
    tol = 1e-5
    assert_antiherm(H,tol)
    # @assert H isa Hermitian
    E, ψ = eigen(Hermitian(H))
    N = length(E)
    Ψ = []
    for j=1:N
        φ = ψ[:,j]
        # φ /= φ[1] # Fixes the gauge ! In case H is complex and (hence) φ is complex. Don't do that, if φ[1] is small then φ is Nan
        φ /= norm(φ)
        # Tests
        @assert abs(norm(φ)-1)<tol
        res = norm(H*φ .- E[j]*φ)
        # px("residue ",j," ",res)
        @assert res<tol
        # px("IMAG ψ ",norm(imag.(φ)))
        push!(Ψ,φ)
    end
    # px("ENERGIES ",E)
    E, Ψ # BEFORE
end

function assert_isPositiveDefinite(A)
    E = eigvals(A)
    tol = 1e-13
    if E[1] < tol
        px("Problem positive def ",E[1])
        @assert false
    end
end

function gen_eigenmodes(H,B;tol=1e-10,check_tests=true) # eigenmodes, H ϕ = λ B ϕ
    if check_tests
        assert_antiherm(H,tol)
        assert_antiherm(B,tol)
        assert_isPositiveDefinite(B)
    end
    Hh = Hermitian((H+H')/2)
    Bh = Hermitian((B+B')/2)
    E, ψ = eigen(Hh,Bh)
    N = length(E)
    Ψ = []
    for j=1:N
        φ = ψ[:,j]
        # φ /= φ[1] # Fixes the gauge ! In case H is complex and (hence) φ is complex. Don't do that, if φ[1] is small then φ is Nan
        # φ /= norm(φ) # the generalized problem is not normed ! see test_sqrt_matrix_eigen
        # Tests
        # @assert abs(norm(φ)-1)<tol
        res = norm(Hh*φ .- E[j]*Bh*φ)/norm(φ)
        @assert res<tol
        push!(Ψ,φ)
    end
    # px("ENERGIES ",E)
    E, Ψ # BEFORE
end

################################## Pseudo-inverse K

# K = (E0-H)^{-1}⟂ when the ground energy is E0 and is non degenerate
# Es, ψs = eigenmodes(H)
function pseudo_inverse(Es,ψs) 
    # println("size ",size(ψs[1]))
    N = size(ψs[1])[1] # size of Hilbert space
    Ps = [ψs[j]*ψs[j]' for j=1:N] # projections on the j^th excited state
    E0 = Es[1] # ground energy
    K = sum((1/(E0-Es[j]))*Ps[j] for j=2:N)
    @assert norm(K*ψs[1])<1e-5
    K
end

function all_pseudo_inverses(Es,ψs) 
    N = size(ψs[1])[1] # size of Hilbert space
    Ps = [ψs[j]*ψs[j]' for j=1:N]
    Ks = []
    # px("Computes pseudo inverses N=",N," dim= ",length(ψs[1]))
    for k=1:N
        ks = [j for j=1:N]
        deleteat!(ks,k)
        # println("ks ",ks)
        K = sum((1/(Es[k]-Es[j]))*Ps[j] for j in ks)
        # px("K ",K)
        # px("ψ ",ψs[k])
        # px("norm Kψ ",norm(K*ψs[k]))
        # @assert norm(K*ψs[k])<1e-5
        push!(Ks,K)
    end
    Ks
end

################################## Low level for contour integrals

# Integration on complex plane
circ(z0,r,θ) = z0 + r*cis(2π*θ)
circle_path(z0,r,res) = circ.(z0,r,(0:1/res:1-1/res)) # array of coords of circle around z0
∇array(a,dx) = [a[mod1(i+1,length(a))]-a[mod1(i-1,length(a))] for i=1:length(a)]/(2*dx)

function test_circle_path()
    len = 30
    cp = circle_path(im+2,0.1,len)
    @assert len==length(cp)
    test_circle_path(cp)
end

function test_circle_path(cp)
    p = Plots.scatter(real(cp),imag(cp))
    display(p)
end

# A = z ↦ A(z) is an operator valued function, the integration contour is ∂B_r(z0), res is the number of points on the circle around, returns 1/(2πi) ∫ A(z) dz
# contour_integral(A,z0,r,res) = sum(A.(circle_path(z0,r,res)))*r/(res*im)

function contour_integral(A,z0,resolution,r=-1)
    res = Int(resolution)
    cp = circle_path(z0,abs(r),res)
    ∇cp = ∇array(cp,2π/res)
    l = [A(cp[i]) for i=1:res]
    sum(l.*∇cp)/(res*im)
end

# Evaluates (z-H)^(-1)
function resolvent(H,z,resolution,tests=false)
    Es,ψs = eigen(H)
    E = Es[1]; e = Es[2]
    P = ψs[:,1]*ψs[:,1]'
    @assert abs(1-norm(P)) + norm(P^2 - P) + norm(P' - P) < 1e-10
    res(z) = inv(z*I-H)
    P_integrated = contour_integral(res,E,resolution,(e-E)/2)
    if tests
        px("norm P_int ",norm(P_integrated),", P_integrated = P ",dist(P_integrated,P)," norm E ",norm(res(E)))
    end
    P_integrated
end

function test_contour_integral()
    N = 20
    H = rand(N,N)
    H = H+H'
    Es,ψs = eigen(H)
    E = Es[1]
    resolvent(H,E,1000,true)
end

################################## Misc

function normy(f,g,dd,normalize)
    d = dd
    if normalize
        nor = (norm(f) + norm(g))/2
        if abs(nor) < 1e-15
            px("Division by zero in distance")
            return 0
        end
        d /= nor
    end
    regularizes(d)
end

# relative distance between f and g, whatever they are
function dist(f,g;normalize=true)
    d = norm(f.-g)
    normy(f,g,d,normalize)
end

function regularizes(x) # regularizes
    lim = 1e-15
    max(lim,x)
end

# relative distance between f and g, minimized over the phasis
# z := <u,v>
# f(θ) = ||u-exp(iθ)v||^2 = ||u||^2 + ||v||^2 - 2 Re (z exp(iθ))
# 0.5 f'(θ) = (sin θ) Re z + (cos θ) Im z
# f'(θ) = 0 iff tan θ = - (Im z)/(Re z) iff θ = -arctan((Im z)/(Re z)) [π]
function dist_quo(f,g;normalize=true) # also works when f and g are real, then it is min(f-g,f+g)
    n(θ) = norm(f - cis(θ)*g)
    z = f⋅g
    Θ = -atan((imag(z))/(real(z)))
    d = minimum(n.(Θ .+ π*(-1:1)))
    normy(f,g,d,normalize)
    # d
end

function antiherm(M) # antihermitian part of M
    n = size(M,1)
    @assert size(M) == (n,n)
    s = sum(abs.(M))
    if s<1e-10
        return 0
    end
    sum(abs.((M'.-M)/2))/s
end

function assert_antiherm(M,tol)
    h = antiherm(M) # antihermitian part of M
    if h>tol
        px("Antiherm problem ",h)
        @assert h<tol
    end
end

function test_hermitianity(M,name) 
    s = sum(abs.(M))
    if s<1e-10
        return 0
    end
    px("Hermitianity ",name," : ",antiherm(M))
end

function hermitianize(h)
    assert_antiherm(h,1e-10)
    Hermitian((h'+h)/2)
end

herm(h) = Hermitian((h'+h)/2)

function gauge_away(ψ,ϕ,tol=1e-3) # tests if ψ = α ϕ, returns α
    αvec = ψ./ϕ
    if dist(αvec,fill(αvec[1],length(αvec))) < tol
        px("ψ = αϕ ")
    else
        px("ψ ≂̸ αϕ ")
    end
    αvec[1]
end

function test_projector(A,name) # test whether A is a projector
    x = dist(A^2,A) + dist(A',A)
    px("Test if ",name," is a projector : ",x)
end

# Init matrices
zero(M) = zeros(size(M))
id(S) = zero(S) + I
# A is a matrix
function id(A) 
    N = size(A,1)
    Matrix(1.0I, N,N)
end

function rand_herm(N) # random hermitian matrix
    M = rand(N,N)
    M .+= M'
    Hermitian(M)
end
