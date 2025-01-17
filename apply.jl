using LinearAlgebra
using MKL
using CairoMakie 
using LaTeXStrings

include("modules/schrodinger.jl")
include("modules/spectral_stuff.jl")
include("derivatives_stock.jl")
include("rbm_module.jl")
px = println

PATH_PLOTS = "pics"
if !isdir(PATH_PLOTS)
    mkdir(PATH_PLOTS)
end

sca(a,b) = a⋅b # scalar product

logrange(start,end_,len) = start.*((end_/start)^(1/(len-1))).^(0:len-1)

function reg(x) # regularizes
    lim = 1e-15
    if x>lim
        return x
    else
        return lim
    end
end

function schro_Hn(N)
    exact = true
    (A,H0,G0,G1,G2) = create_schrodinger_potentials(N;plot=false)
    for M in [H0,G0,G1,G2]
        test_hermitianity(M,"M")
    end
    Hs = [H0 .+ G0,G1,G2]
    A, Hs
end

function random_Hn(N)
    Hs = [copy(herm(randn(N,N))) for k=0:2]
    # px("NORM ",norm(Hs[1].-Hs[2]))
    I, Hs
end

log_slope(y2,y1,x2,x1) = (log(y2) - log(y1))/(x2 - x1)
log_log_slope(y2,y1,x2,x1) = (log(y2) - log(y1))/(log(x2) - log(x1))

function perturbation(q,λ,ℓ,μ,S;λ0=0,unit_norm=true)
    v = sum((λ-λ0)^k * X(q,(k,μ),S) for k=0:ℓ)
    if q in ["ϕ","Φ"] && !unit_norm
        v = sum((λ-λ0)^k*X(q,(k,μ),S) for k=0:ℓ)
    end
    if unit_norm
        v /= norm(v)
    end
    v
end

Hλ(λ,Hs;λ0=0) = sum((λ-λ0)^k*Hs[k+1] for k=0:length(Hs)-1)
rp = ["rbm","pert"]
dist_vects(f,g,A) = dist_quo(A*f,A*g;normalize=false)
dist_q = Dict("E" => (f,g,A) -> dist(f,g;normalize=false), "ϕ" => dist_vects)



function fill_results(r,λ,Hs,S,R,A;λ0=0,s=1,unit_norm=true)
    H = Hλ(λ,Hs;λ0=λ0)
    # Solves exact
    E_exact, ϕ_exact = eigenmodes(H)
    # Solves RBM
    E_rbm, ϕ_rbm = rbm_approx_eigenmodes(H,R)
    # Fills results
    for μ=0:length(E_rbm)-1
        # Solves perturbative
        E_pert, ϕ_pert = compute_pert_approx(λ,R.ℓ,μ,S,Hs;unit_norm=unit_norm)
        q_pert =  Dict("E" => E_pert, "ϕ" => ϕ_pert)

        Eexct = E_exact[μ+1]
        ϕexct = ϕ_exact[μ+1]
        Erbm = E_rbm[μ+1]
        ϕrbm = ϕ_rbm[μ+1]

        # Reference
        if !unit_norm
            ϕexct /= sca(ϕ0,ϕexct)
            ϕrbm /=  sca(ϕ0,ϕrbm)
        end

        q_exact = Dict("E" => Eexct, "ϕ" => ϕexct)
        q_rbm =   Dict("E" => Erbm,  "ϕ" => ϕrbm)
        c = (R.ℓ,μ,R.ν,s)
        # if (c[1],c[2],c[3]) == (0,0,0) && λ < 0.3
            # px("Fills ",c," ",λ)
        # end
        for q in ["E","ϕ"]
            r[(λ,q,c,"pert")] = q_pert[q]
            r[(λ,q,c,"exact")] = q_exact[q]
            r[(λ,q,c,"rbm")] = q_rbm[q]
            dis = dist_q[q]
            x = q=="ϕ" ? norm(A*X("ϕ",(0,μ),S)) : abs(X("E",(0,μ),S))
            r[(λ,q,c,"pert")] = dis(r[(λ,q,c,"exact")], r[(λ,q,c,"pert")], A)/x
            r[(λ,q,c,"rbm")]  = dis(r[(λ,q,c,"exact")], r[(λ,q,c,"rbm")],  A)/x
            # px("D_pert ",r[(λ,q,c,"D_pert")]," D_rbm ",r[(λ,q,c,"D_rbm")])
        end
    end
end

function compute_pert_approx(λ,ℓ,μ,S,Hs;unit_norm=true)
    H = Hλ(λ,Hs;λ0=0)
    ϕ_pert = perturbation("ϕ",λ,ℓ,μ,S;λ0=0,unit_norm=unit_norm)
    E_pert = ϕ_pert⋅(H*ϕ_pert)/norm(ϕ_pert)^2
    E_pert, ϕ_pert 
end


# Computes factor ξ
function compute_ξ_from_r(r,λs,R)
    λ = minimum(λs)
    for μ=0:R.ν
        c = (R.ℓ,μ,R.ν,1)
        ξ = r[(λ,"ϕ",c,"pert")]/r[(λ,"ϕ",c,"rbm")]
        px("μ=",μ," ν=",R.ν," ℓ=",R.ℓ," => ξ=",ξ)
    end
end

function test_PKP_equals_Q()
    ## Tests whether PKP = Q
    Pnq = create_projector(R) # P is size d × length(R.V)
    test_hermitianity(Pnq,"P")
    SP = stock_at_H([Pnq'*h*Pnq for h in Hs])
    K0 = X("K",0,S); Q0 = X("K",0,SP)
    px("PKP=Q ? ",dist(Pnq'*K0*Pnq, Q0))
    px("norm(K)=",norm(K0)," norm(Q)=",norm(Q0))
    ## End tests whether PKP = Q
end

function slopes()
    λ_end = minimum(λs)
    λ_pe = minimum(λs[λs .!= λ_end])
    for qu in ["E","ϕ"], ap in rp, μ=0:ν
        slope = log_log_slope(r[(λ_end,qu,μ,ap)],r[(λ_pe,qu,μ,ap)],λ_end,λ_pe)
        px("Slope of ",qu," for μ=",μ," and ",ap," is ",slope)
    end
end

function get_λs(n_start,n_end)
    resolution = 150
    λs = logrange(10^float(n_end),10^float(n_start),resolution)
    λs
end

# Plot functions
δ(ℓ,ν) = "{("*string(ℓ,",",ν)*")}"
ev_labels(ℓ,ν) = Dict("E" => Dict("exact"=>"E", "pert"=>"e^"*string(ℓ), "rbm"=>"\\bscrE^"*string(ℓ)),
                      "ϕ" => Dict("exact"=>"\\phi", "pert"=>"\\varphi^"*string(ℓ), "rbm"=>"\\psi^"*string(ℓ)))
ev_labels_ind(ℓ,ν) = Dict("E" => Dict("exact"=>"E", "pert"=>"e_"*string(ν), "rbm"=>"\\bscrE_"*string(ν)),
                      "ϕ" => Dict("exact"=>"\\phi", "pert"=>"\\varphi_"*string(ν), "rbm"=>"\\psi_"*string(ν)))
Ddist = Dict("E" => "", "ϕ"=> "_e")

style = Dict("pert" => :dash, "rbm" => nothing)
width = Dict("pert" => 1, "rbm" => 2)

pref_norm = Dict("E" => "|", "ϕ" => "\\Vert ")
suff_norm = Dict("E" => "|", "ϕ" => "\\Vert ")

label(Eϕ,ℓ,μ,ν,pert_rbm) = latexstring(pref_norm[Eϕ]*ev_labels(ℓ,ν)[Eϕ]["exact"]*"_"*string(μ)*" (\\lambda)  -  "*ev_labels(ℓ,ν)[Eϕ][pert_rbm]*"_"*string(μ)*"(\\lambda)"*suff_norm[Eϕ]*Ddist[Eϕ]*"/"*pref_norm[Eϕ]*ev_labels(ℓ,ν)[Eϕ]["exact"]*"_"*string(μ)*" (0) "*suff_norm[Eϕ]*Ddist[Eϕ])


label_expo(Eϕ,ℓ,μ,ν,pert_rbm) = latexstring(pref_norm[Eϕ]*ev_labels(ℓ,ν)[Eϕ]["exact"]*" (\\lambda)  -  "*ev_labels(ℓ,ν)[Eϕ][pert_rbm]*"(\\lambda)"*suff_norm[Eϕ]*Ddist[Eϕ]*"/"*pref_norm[Eϕ]*ev_labels(ℓ,ν)[Eϕ]["exact"]*" (0) "*suff_norm[Eϕ]*Ddist[Eϕ])

label_ind(Eϕ,ℓ,μ,ν,pert_rbm) = latexstring(pref_norm[Eϕ]*ev_labels_ind(ℓ,ν)[Eϕ]["exact"]*" (\\lambda)  -  "*ev_labels_ind(ℓ,ν)[Eϕ][pert_rbm]*"(\\lambda)"*suff_norm[Eϕ]*Ddist[Eϕ]*"/"*pref_norm[Eϕ]*ev_labels_ind(ℓ,ν)[Eϕ]["exact"]*" (0) "*suff_norm[Eϕ]*Ddist[Eϕ])

which_str(Eϕ) = Eϕ=="E" ? "E" : "phi"
colors = ["red",RGBf(0.5, 0.2, 0.8),"green","blue","black","brown","orange","pink","cyan","grey",RGBf(0.1, 0.9, 0.4),RGBf(0.4, 0.2, 0.5),RGBf(0.7, 0.1, 0.3),RGBf(0.7, 0.54, 0.9)]

function vrbs(var,v,ℓ,μ,ν)
    if var=="ℓ"
        return (v,μ,ν,1)
    elseif var=="μ"
        return (ℓ,v,ν,1)
    elseif var=="ν"
        return (ℓ,μ,v,1)
    elseif var=="s"
        return (ℓ,μ,ν,v)
    end
end

text_var = Dict("ν"=>"nu","μ"=>"mu","ℓ"=>"ell","s"=>"N")

function vrbs_str(var,ℓ,μ,ν)
    ell, mu, nu = string.((ℓ,μ,ν))
    if var=="ℓ"
        return ["ℓ",mu,nu]
    elseif var=="μ"
        return [ell,"μ",nu]
    elseif var=="ν"
        return [ell,mu,"ν"]
    elseif var=="s"
        return [ell,mu,nu]
    end
end

function get_first_exc_state(H)
    E, Ψ = eigenmodes(H)
    Ψ[2]
end

# to find the eigenmode of R which is modeled by R, corresponding to the mode μ of H^0
function find_which_mode_modeled(μ,S,R)
    H0 = X("H",0,S)
    E = X("E",(0,μ),S)
    E_rbm, ϕ_rbm = rbm_approx_eigenmodes(H0,R)
    μ_rbm = argmin(E_rbm)-1
    px("Dist μ,μ_rbm ",dist(μ,μ_rbm))
    μ_rbm
end

function varies_both_ℓ_and_ν()
    unit_norm = true
    rg = (0:3:21) # ℓ and ν will vary here

    # Parameters
    N = 80
    A, Hs = schro_Hn(N)
    # A, Hs = random_Hn(N)
    S = stock_at_H(Hs)
    estimate_norms(10,S)

    R = Dict()
    ℓν = ["ℓ","ν"]
    function give_ℓν(v,vl)
        ell = vl == "ℓ" ? v : 0
        nu = vl == "ν" ? v : 0
        ell, nu
    end
    for vl in ℓν
        for v in rg
            x = (vl,v)
            ℓ, ν = give_ℓν(v,vl)
            R[x] = EigenRBM(ℓ,ν)
            # px(R[x].ℓ," ",R[x].ν)
            init_R(R[x],S)
            do_Gram_Schmidt(R[x];n=5)
            test_R(S,R[x])
        end
    end

    λs = get_λs(2,-0.2)
    r = Dict() # results
    
    for λ in λs
        for vl in ℓν, v in rg
            x = (vl,v)
            fill_results(r,λ,Hs,S,R[x],A,unit_norm=unit_norm)
        end
    end

    # Fills a plot
    plots = Dict()
    for qu in ["E","ϕ"], ap in rp, vl in ℓν, v in rg
        ℓ, ν = give_ℓν(v,vl)
        plots[(qu,ap,vl,v)] = [r[(λ,qu,(ℓ,0,ν,1),ap)] for λ in λs]
    end
    
    #### Builds a plot
    size_txt = 30
    ls = Dict(); labels = Dict()
    x_sizes = Dict("ν"=>500, "μ"=>500, "ℓ"=>600)
    function build_plot_ℓ_ν(Eϕ) # Eϕ ∈ ["E","ϕ"]
        ls[Eϕ] = []
        labels[Eϕ] = []
        # Structure
        f = Figure(resolution=(600, x_sizes["ℓ"]), fontsize=size_txt)
        xlabel = L"\lambda"
        # ylabel = label(Eϕ,ℓ_str,μ_str,ν_str,"rbm")
        ax = Axis(f[1, 1], xscale=Makie.log10, yscale=Makie.log10, xlabel=xlabel, yminorticks = [10^(float(n)) for n=-15:15], yminorticksvisible = true, yminorgridvisible = true)#, xticks = [10^(-4)])ylabel=ylabel, 
        xlims!(ax, minimum(λs), maximum(λs))
        ylims!(ax, 1e-6, maximum(λs))
        # hidedecorations!(ax, grid = false) # hides x ticks

        # Fills with lines
        this_style = Dict("ν" => :dash, "ℓ" => nothing)
        this_width = Dict("ν" => 1, "ℓ" => 2)
        for ka in ["rbm"]
            i = 1
            for v in rg
                for vl in ℓν
                    line = plots[(Eϕ,ka,vl,v)]
                    l = lines!(ax, λs, line, color=colors[i], linestyle=this_style[vl], linewidth=this_width[vl]) # line
                    push!(ls[Eϕ],l)
                    if vl == "ℓ"
                        push!(labels[Eϕ],string(v))
                    end
                end
                i += 1
            end
        end

        fictitious_lines = [lines!(ax, λs, [1 for λ in λs], color="black", linestyle=this_style[vl], linewidth=this_width[vl]) for vl in ℓν]

        ℓ_str(vl) = vl=="ℓ" ? "β" : "0"
        ν_str(vl) = vl=="ν" ? "β" : "0"
        lab(vl) = vl == "ℓ" ? label_expo : label_ind

        axislegend(ax, fictitious_lines, [lab(vl)(Eϕ,ℓ_str(vl),"0",ν_str(vl),"rbm") for vl in ℓν], position = :rb, orientation = :vertical)
        # for l in fictitious_lines
            # delete!(ax.scene, l)
        # end

        # Saves
        namefile = PATH_PLOTS*"/both_ell_mu_varies_"*which_str(Eϕ)*".pdf"
        # px(namefile)
        save(namefile,f)
    end
    for Eϕ in ["E","ϕ"]
        build_plot_ℓ_ν(Eϕ) # Eϕ ∈ ["E","ϕ"]
    end

    # Legend
    fig_leg = Figure(resolution=(1200, 80), fontsize=20)
    ls_f = ls["ϕ"][1:2:2*length(rg)]
    labs_f = labels["ϕ"][1:length(rg)]
    Legend(fig_leg[1,2], ls_f, labs_f, "β = ", halign = :left, valign = :center, orientation = :horizontal)
    namefile_leg = PATH_PLOTS*"/legend_both_ell_mu_varies.pdf"
    save(namefile_leg,fig_leg)
    # axislegend(ax, ls, labels, position = :rb, orientation = :horizontal)
end

# var ∈ ["ν","μ","ℓ"] is the variable which varies
# rg is the range in which the variable varies

function extract_ξs(r,λs,rg)
    ξs = []
    λ = maximum(λs)
    for x in λs
        if x < λ && x > 10^(-0.4)
            λ = x
        end
    end
    for n in rg
        q = "ϕ"
        c = (n,0,0,1)
        # Gives the ξ's
        ξ = r[(λ,q,c,"pert")]/r[(λ,q,c,"rbm")]
        ξr = round(ξ, digits=1)
        print(floor(Int,ξr)," &")
        # push!(ξs,ξ)
    end
end

function varying_one_variable(var,rg;ℓ=0,μ=0,ν=0,unit_norm=true)
    px("VARIES ",var)
    # Parameters
    N = 40
    A, Hs = schro_Hn(N)
    # A, Hs = random_Hn(N)
    S = stock_at_H(Hs)
    estimate_norms(10,S)

    R = Dict()
    for v in rg
        (ell,mu,nu) = vrbs(var,v,ℓ,μ,ν)
        R[v] = EigenRBM(ell,nu)
        init_R(R[v],S)
        Ψ1 = get_first_exc_state(Hλ(-1,Hs))
        Ψ2 = get_first_exc_state(Hλ(-2,Hs))
        add_vectors([Ψ1,Ψ2],R[v])
        do_Gram_Schmidt(R[v];n=5)
        test_R(S,R[v])
    end

    λs = get_λs(1,-0.5)
    r = Dict() # results
    
    for v in rg
        for λ in λs
            fill_results(r,λ,Hs,S,R[v],A,unit_norm=unit_norm)
        end
        compute_ξ_from_r(r,λs,R[v])
    end

    extract_ξs(r,λs,rg)

    # Fills a plot
    plots = Dict()
    for qu in ["E","ϕ"], ap in rp, v in rg
        plots[(qu,ap,v)] = [r[(λ,qu,vrbs(var,v,ℓ,μ,ν),ap)] for λ in λs]
    end
    
    #### Builds a plot
    size_txt = 30
    ls = Dict(); labels = Dict()
    x_sizes = Dict("ν"=>500, "μ"=>500, "ℓ"=>600)
    function build_plot2(Eϕ) # Eϕ ∈ ["E","ϕ"]
        (ℓ_str,μ_str,ν_str) = vrbs_str(var,ℓ,μ,ν)
        ls[Eϕ] = []
        labels[Eϕ] = []
        # Structure
        f = Figure(resolution=(600, x_sizes[var]), fontsize=size_txt)
        xlabel = L"\lambda"
        # ylabel = label(Eϕ,ℓ_str,μ_str,ν_str,"rbm")
        ax = Axis(f[1, 1], xscale=Makie.log10, yscale=Makie.log10, xlabel=xlabel, yminorticks = [10^(float(n)) for n=-15:15], yminorticksvisible = true, yminorgridvisible = true)#, xticks = [10^(-4)])ylabel=ylabel, 
        xlims!(ax, minimum(λs), maximum(λs))
        # hidedecorations!(ax, grid = false) # hides x ticks

        # Fills with lines
        miny = 1
        maxy = 0
        for ka in rp
            i = 1
            for v in rg
                line = plots[(Eϕ,ka,v)]
                miny = min(miny,minimum(line))
                maxy = max(maxy,maximum(line))
                # wrk(x) = x <1e-14 ? 1e200 : x
                # line = [wrk(line[j]) for j=1:length(line)]
                l = lines!(ax, λs, line, color=colors[i], linestyle=style[ka], linewidth=width[ka]) # line
                push!(ls[Eϕ],l); push!(labels[Eϕ],string(v))
                i += 1
            end
        end
        ylims!(ax, 2e-14, maxy)


        fictitious_lines = [lines!(ax, λs, [1 for λ in λs], color="black", linestyle=style[k], linewidth=width[k]) for k in rp]
        axislegend(ax, fictitious_lines, [label_expo(Eϕ,ℓ_str,μ_str,ν_str,k) for k in rp], position = :rb, orientation = :vertical)
        for l in fictitious_lines
            delete!(ax.scene, l)
        end

        # Saves
        namefile = PATH_PLOTS*"/"*text_var[var]*"_varies_"*which_str(Eϕ)*".pdf"
        # px(namefile)
        save(namefile,f)
    end
    for Eϕ in ["E","ϕ"]
        build_plot2(Eϕ) # Eϕ ∈ ["E","ϕ"]
    end

    # Legend
    fig_leg = Figure(resolution=(1500, 80), fontsize=20)
    ls_f = ls["E"][1:length(rg)]
    labs_f = labels["E"][1:length(rg)]
    Legend(fig_leg[1,2], ls_f, labs_f, var*" = ", halign = :left, valign = :center, orientation = :horizontal)
    namefile_leg = PATH_PLOTS*"/legend_"*text_var[var]*"_varies.pdf"
    save(namefile_leg,fig_leg)
    # axislegend(ax, ls, labels, position = :rb, orientation = :horizontal)
end

function see_if_rbm_error_is_better(rg,r,λ,S)
    ϕ1 = X("ϕ",(1,0),S)
    h1_ϕ1 = X("h",(1,0),S)*ϕ1
    β = sca(ϕ1 , h1_ϕ1 ) / sca( X("ϕ",(0,0),S) , h1_ϕ1 )
    pos = 2β*sca(ϕ1,X("Φ",(2,0),S)) - β^2*norm(ϕ1)^2
    # px("β = ",β," pos ? ",pos)

    for k in rg
        c = (k,0,0,1)
        di = (r[(λ,"ϕ",c,"pert")] - r[(λ,"ϕ",c,"rbm")])/λ^2
        # if di < 0
        if k==1 && (di < 0 || pos < 0)
            # if k==1 
            px("λ^(-ℓ) (|ϕ(λ)-φ(λ)| - |ϕ(λ)-ψ(λ)|) for ℓ=",k," : ",di," pos ",pos/abs(pos))
        end
    end
end

function study_fine_error()
    μ=0; ν=0
    unit_norm=true
    var = "ℓ"
    rg = (0:4)
    # Parameters
    N = 20
    n_trials = 1000
    for i=1:n_trials
        Hs = random_Hn(N)
        S = stock_at_H(Hs)

        R = Dict()
        for v in rg
            R[v] = EigenRBM(v,ν)
            init_R(R[v],S)
            # add_random_vectors(3,R[ν])
            do_Gram_Schmidt(R[v];n=5)
            test_R(S,R[v])
        end

        λ = 1e-6
        r = Dict() # results
        
        for v in rg
            fill_results(r,λ,Hs,S,R[v],A,unit_norm=unit_norm)
        end

        see_if_rbm_error_is_better(rg,r,λ,S)
    end
end

function vary_variables()
    unit_norm = true
    var = "ℓ"; μ = 0; ν = 0; rg = (0:2:12)
    varying_one_variable(var,rg;μ=μ,ν=ν,unit_norm=unit_norm)

    # var = "μ"; ℓ = 2; ν = 7; rg = (0:ν+1)
    # varying_one_variable(var,rg;ℓ=ℓ,ν=ν,unit_norm=unit_norm)

    # var = "ν"; ℓ = 2; μ = 0; rg = (0:8)
    # varying_one_variable(var,rg;ℓ=ℓ,μ=μ,unit_norm=unit_norm)
end

function study_preconstant()
    L = 10
    ℓs = (0:L)
    # Parameters
    N = 40
    A, Hs = schro_Hn(N)
    # A, Hs = random_Hn(N)
    S = stock_at_H(Hs)
    # estimate_norms(10,S)

    R = Dict()
    for ℓ in ℓs
        (ell,mu,nu) = vrbs(var,v,ℓ,μ,ν)
        R[v] = EigenRBM(ell,nu)
        init_R(R[v],S)
        do_Gram_Schmidt(R[v];n=5)
        test_R(S,R[v])
    end

    λs = get_λs(1,-0.5)
    r = Dict() # results
    
    for v in rg
        for λ in λs
            fill_results(r,λ,Hs,S,R[v],A,unit_norm=unit_norm)
        end
        compute_ξ_from_r(r,λs,R[v])
    end

    extract_ξs(r,λs,rg)

    # Fills a plot
    plots = Dict()
    for qu in ["E","ϕ"], ap in rp, v in rg
        plots[(qu,ap,v)] = [r[(λ,qu,vrbs(var,v,ℓ,μ,ν),ap)] for λ in λs]
    end
end

function test_positivity()
    unit_norm = true
    var = "ℓ"; μ = 0; ν = 0; rg = (0:2)
    varying_one_variable(var,rg;μ=μ,ν=ν,unit_norm=unit_norm)
end

function plot_schrodinger()
    path = PATH_PLOTS*"/pot_sol.pdf"
    create_schrodinger_potentials(150;plot=true,path=path)
end

plot_schrodinger()
varies_both_ℓ_and_ν()
vary_variables()
