#=
Author: Olivia
Date: October 2024 
=#
@with_kw struct Primitives
    N::Int64 = 66 #number of periods 
    n::Float64 = 0.011 #population growth rate//model period length 
    a_1::Float64 = 0.01 #initial assets at age 1 
    a_max::Float64 = 100.0 #max assets
    nA::Int64 = 501 #number of asset grid points
    A_grid::Array{Float64,1} = collect(range(a_1, length = nA, stop = a_max)) #grid for assets
    σ::Float64 = 2.0 #coefficient of relative risk aversion
    η::Array{Float64,1} = map(x->parse(Float64,x), readlines("PS5/ef.txt")) # deterministic age-efficiency profile
    π_hh::Float64 = 0.9261 #transition probability from high to high productivity
    π_ll::Float64 = 0.9811 #transition probability from low to low productivity
    π::Array{Float64,2} = [π_hh 1-π_hh; 1-π_ll π_ll] #transition matrix
    π_0::Array{Float64, 1} = [0.2037, 0.7963]  # Erodic distribution of Π
    α::Float64 = 0.36 #capital share of output
    δ::Float64 = 0.06 #depreciation rate
    β::Float64 = 0.97 #discount rate
end

#structure that holds model results
@with_kw mutable struct Results
    val_function::Array{Float64, 3} #firm value function W
    val_w::Array{Float64, 3} #value function for workers over assets and productivity
    val_r::Array{Float64, 3} #value function for retired just over assets 
    pol_func::Array{Float64, 3} #policy function (choice of assets/savings function)
    e::Array{Float64, 2} # productivity levels
    mu_norm::Array{Float64, 1} #relative size of each cohort
    F::Array{Float64, 3} #Steady state distribution over age, productivity, and assets
    K::Float64 #capital supply
    L::Float64 #labor supply
    labor::Array{Float64, 3} #labor supply over assets and productivity
    w::Float64 #wage
    r::Float64 #interest rate
    b::Float64 #pension
    θ::Float64 #proportional labor income tax
    γ::Float64 #weight on consumption
    z::Array{Float64,1} #productivity states
    g_0::Array{Float64, 4} #policy function for transition path
    V_transition::Array{Float64, 4} #value function for transition path
    labor_transition::Array{Float64, 4} #labor supply for transition path
    w_transition::Array{Float64, 1} #wage for transition path
    r_transition::Array{Float64, 1} #interest rate for transition path
    b_transition::Array{Float64, 1} #pension for transition path
end

#function for initializing model primitives and results
function Initialize(θ::Float64, γ::Float64, z::Array{Float64,1},N_t::Int64)
    prim = Primitives() #initialize primtiives
    val_function = zeros(prim.N,prim.nA, 2) #initial value function guess
    val_w = zeros(prim.N,prim.nA, 2) #initial value function guess
    val_r = zeros(prim.N,prim.nA, 2) #initial value function guess
    pol_func = zeros(prim.N,prim.nA, 2) #initial policy function guess
    #e = prim.η.*prim.z'  # productivity levels
    e = zeros(prim.N, 2)
    mu_norm = zeros(prim.N) #normalized relative size of each cohort
    F = zeros(prim.N,prim.nA,2)
    K = 0 #initial capital supply guess
    L = 0 #initial labor supply guess
    labor = zeros(prim.N, prim.nA, 2) #initial labor supply guess
    w = 1.05 
    r = 0.05 
    b = 0.2 
    g_0 = zeros(prim.N, prim.nA, 2, N_t)
    V_transition = zeros(prim.N, prim.nA, 2, N_t)
    labor_transition = zeros(prim.N, prim.nA, 2, N_t)
    w_transition = zeros(N_t)
    r_transition = zeros(N_t)
    b_transition = zeros(N_t)
    res=Results(val_function, val_w, val_r, pol_func, e, mu_norm, F,K,L, labor, w, r, b, θ, γ, z,g_0, V_transition, labor_transition, w_transition, r_transition, b_transition) #initialize vector results struct
    prim, res #return deliverables
end

function utility_r(prim, a, ap, γ)
   @unpack σ = prim
    c = (1+res.r)*a + res.b - ap
    if c<=0 
        #return -Inf
        return -1/eps()
    else 
        return (c^((1-σ)*γ))/(1-σ)  
    end 
end


function labor(a, ap,E, θ, γ)
    num = γ*(1-θ)*E*res.w - (1-γ)*((1+res.r)*a-ap) 
    denom = (1-θ)*res.w*E
    interior_solution = num/denom
    return min(1, max(0, interior_solution))
end 


function utility_w(prim, a, ap, E, θ, γ)
    @unpack σ = prim
    l = labor(a, ap, E,θ, γ)
    c = res.w*(1-θ)*E*l + (1+res.r)*a - ap 
    if (c>0 && l>=0 && l<=1 )
        return (((c^γ)*(1-l)^(1-γ))^(1-σ))/(1-σ)
    else
        #return -Inf
        return -1/eps()
    end 
end 

function Bellman(prim, res, θ, z, γ)
    @unpack N, n, A_grid, π, β = prim
    @unpack val_function, val_w, val_r, pol_func, e = res
    res.e = prim.η.*z'
    val_r_next = zeros(N, prim.nA, 2) .- Inf
    val_w_next = zeros(N, prim.nA, 2) .- Inf
    val_w = zeros(N, prim.nA, 2) .- Inf
    val_r = zeros(N, prim.nA, 2) .- Inf
    for age in N:-1:1 
        for j in 1:2 
            for i in 1:prim.nA
                a = A_grid[i]
                if age==66
                    ap=0
                    val_r[age, i, j] = utility_r(prim, a, ap, γ)
                    res.val_function[age, i, j] = val_r[age, i, j] 
                    res.pol_func[age, i, j] = 0
                    #println("age: ", age)
                elseif 45<age<66 
                    choice_lower = 1 
                    #=if θ==0 && i==1
                        val_r[age, i, j] = utility_r(prim, a, 0, γ) + β*res.val_function[age+1, 1, j]
                        res.pol_func[age, i, j] = 0.0=#
                    #else
                        for k in 1:prim.nA 
                            ap = A_grid[k]
                            val_r_next[age, i, j] = utility_r(prim, a, ap, γ) + β*res.val_function[age+1, k, j]
                            if val_r_next[age, i, j]>val_r[age, i, j] 
                                val_r[age, i, j] = val_r_next[age, i, j]
                                res.pol_func[age, i, j] = ap
                                choice_lower = k
                            end 
                            #val_r = val_r_next
                        end 
                    #end 
                    res.val_function[age, i, j] = val_r[age, i, j] 
                    #println("age: ", age)
                else
                    E = res.e[age, j]
                    choice_lower = 1
                    for k in 1:prim.nA 
                        ap = A_grid[k]
                        val_w_next[age, i, j]  = utility_w(prim, a, ap, E, θ, γ) + β*(π[j,1]*res.val_function[age+1, k, 1] + π[j,2]*res.val_function[age+1, k, 2])
                        if val_w_next[age, i, j]>val_w[age, i, j]
                            val_w[age, i, j] = val_w_next[age, i, j]
                            res.pol_func[age, i, j] = ap
                            res.labor[age, i, j] = labor(a,ap, E, θ, γ)
                            choice_lower = k
                        end 
                        #val_w = val_w_next
                    end 
                    res.val_function[age, i, j] = val_w[age, i, j]
                    #println("age: ", age)
                end 
            end 
        end 
    end 
    return res.val_function, res.pol_func
end 

function Tstar(prim::Primitives,res::Results; tol::Float64 = 1e-10, max_iter::Int64 = 1000)
    @unpack A_grid, nA, π, N, π_0, n = prim #unpack model primitives 
    @unpack val_function, F = res #unpack value
    trans_mat = zeros(N,nA*2,nA*2) #initialize transition matrix
    for age in 1:N 
        for j in 1:2
            for i in 1:nA
                ap = res.pol_func[age,i,j]  # get next period's asset from policy function
                ap_id = argmin(abs.(ap .- A_grid))

                for jp in 1:2
                    trans_mat[age,i + nA*(j -1), ap_id + nA*(jp - 1)] =  π[j, jp]
                end
            end
        end 
    end
    
    μ_dist = zeros(N,nA*2)
    μ_dist[1,1] = π_0[1]
    μ_dist[1,nA+1] = π_0[2]

    for age in 2:N
        μ_dist[age,:] = trans_mat[age-1,:,:]' * μ_dist[age-1,:]
    end

    mu = zeros(N)
    mu[1] = 1
    for age in 1:N-1 
        mu[age+1] = mu[age]/(1+n)
    end
    res.mu_norm = mu./sum(mu)

    μ_dist = res.mu_norm.*μ_dist
    res.F = reshape(μ_dist, (N,nA, 2))
end 


#compute aggregate labor supply 
function agg_labor(prim,res)
    
    res.L = 0 
    for age in 1:45
        for i in 1:prim.nA
            for j in 1:2
                res.L += res.F[age, i, j]*res.e[age, j]*res.labor[age, i, j]
            end 
        end 
    end 
    return res.L
end 

#compute aggregate assets 
function agg_assets(prim,res)
    
    res.K =0 
    for age in 1:prim.N
        for i in 1:prim.nA
            for j in 1:2
                res.K += res.F[age, i, j]*prim.A_grid[i]
            end 
        end 
    end
    return res.K
end 


#solve for prices 
function price_solve(prim, res, K,L, θ)
    @unpack α= prim
    mu = zeros(prim.N)
    mu[1] = 1
    for age in 1:prim.N-1 
        mu[age+1] = mu[age]/(1+prim.n)
    end
    mu_norm = mu./sum(mu)
    res.w = (1-α)*(K^α)*(L^(-α)) #MPL 
    res.r = α*(K^(α-1))*(L^(1-α)) - prim.δ #MPK 
    res.b = (θ*res.w*L)/(sum(mu_norm[46:prim.N,:,:]))#pension 
end 

#solve model 
function model_solve(prim, res; θ::Float64, z::Array{Float64,1}, γ::Float64,eps::Float64, λ::Float64, K::Float64, L::Float64) 
    
    K_0 = K #initial guess for capital supply
    L_0 = L #initial guess for labor supply
    price_solve(prim, res, K_0, L_0, θ) #solve for prices

    max_iter = 100 #maximum number of iterations
    iter = 0 #initialize counter
    conv = 0 
    while iter<max_iter && conv==0
        iter += 1
        println("Iteration: ", iter)
        Bellman(prim, res, θ, z, γ) #solve the HH problem 
        Tstar(prim, res)
        
        @show K_next = agg_assets(prim, res) #compute aggregate assets
        @show L_next = agg_labor(prim, res) #compute aggregate labor supply

        @show err = abs(K_next-K_0) + abs(L_next-L_0) #compute absolute error
        println("K1: ",K_next," K0: ",K_0, " L1: ", L_next, " L0: ",L_0, " Error: ", err)
        if err>eps
            K_0 = (1-λ)*K_0 + λ*K_next #update capital supply
            L_0 = (1-λ)*L_0 + λ*L_next #update labor supply
            price_solve(prim,res, K_0, L_0, θ)
        else #converged 
            println("Model converged in ", iter, " iterations.")
            res.K = K_next
            res.L = L_next
            conv = 1 
        end 
    end  
    return res
end 

## Make tables ##
process_results = function(res::Results)
    # calculate total welfare
    welfare = res.val_function .* res.F
    welfare = sum(welfare[isfinite.(welfare)])

    # calculate coefficient of variation of wealth
    @unpack A_grid, N, nA = prim
    a_grid_3d = permutedims(reshape(repeat(A_grid, N * 2), nA, N, 2), (2, 1, 3))
    wealth_mean = sum(res.F .* a_grid_3d)
    wealth_second_moment = sum(res.F .* a_grid_3d .^ 2)
    wealth_second_central_moment = wealth_second_moment - wealth_mean^2
    cv = wealth_mean / sqrt(wealth_second_central_moment)

    # create vector of summary statistics
    [res.θ, res.γ, res.z[1], res.K, res.L,
     res.w, res.r, res.b, welfare, cv]
end

function create_table(results_vector::Array{Results})
    table = DataFrames.DataFrame(Tables.table(reduce(hcat,process_results.(results_vector))'))
    rename!(table, [:theta, :gamma, :z_H, :k, :l, :w, :r, :b, :welfare, :cv])
end

#solve for prices 
function price_solve_transition(prim, res, K, L, θ, N_t)
    @unpack α= prim
    mu = zeros(prim.N)
    mu[1] = 1
    for age in 1:prim.N-1 
        mu[age+1] = mu[age]/(1+prim.n)
    end
    res.mu_norm = mu./sum(mu)
    for it in 1:N_t
        res.w_transition[it] = (1-α)*(K[it]^α)*(L[it]^(-α)) #MPL 
        res.r_transition[it] = α*(K[it]^(α-1))*(L[it]^(1-α)) - prim.δ #MPK 
        res.b_transition[it] = (θ[it]*res.w_transition[it]*L[it])/(sum(res.mu_norm[46:prim.N,:,:]))#pension 
    end 
end 

function utility_r_t(prim, a, ap, γ, it)
    @unpack σ = prim
     c = (1+res.r_transition[it])*a + res.b_transition[it] - ap
     if c<=0 
         #return -Inf
         return -1/eps()
     else 
         return (c^((1-σ)*γ))/(1-σ)  
     end 
 end
 
 
 function labor_t(a, ap,E, θ, γ, it)
     num = γ*(1-θ[it])*E*res.w_transition[it] - (1-γ)*((1+res.r_transition[it])*a-ap) 
     denom = (1-θ[it])*res.w_transition[it]*E
     interior_solution = num/denom
     return min(1, max(0, interior_solution))
 end 
 
 
 function utility_w_t(prim, a, ap, E, θ, γ, it)
     @unpack σ = prim
     l = labor_t(a, ap, E,θ, γ,it)
     c = res.w_transition[it]*(1-θ[it])*E*l + (1+res.r_transition[it])*a - ap 
     if (c>0 && l>=0 && l<=1 )
         return (((c^γ)*(1-l)^(1-γ))^(1-σ))/(1-σ)
     else
         #return -Inf
         return -1/eps()
     end 
 end 
 

function bellman_transition(prim, res; V_N_SS, g_N_SS, N_t::Int64, K_t::Array{Float64,1}, L_t::Array{Float64,1}, θ_grid::Array{Float64,1})
    @unpack N, n, A_grid, π, β = prim
    @unpack val_function, val_w, val_r, pol_func, V_transition, e = res
    z=[3.0,0.5]
    γ=0.42
    res.e = prim.η.*z'
    println("HH problem in ", N_t)
    #res.V_transition=zeros(N, prim.nA, 2, N_t).-Inf
    #res.g_0=zeros(N, prim.nA, 2, N_t).-Inf
    #res.V_transition=zeros(N, prim.nA, 2, N_t).-Inf
    res.g_0[:,:,:,N_t] = g_N_SS
    #res.g_0[46:prim.N,1,:,N_t] .= 0.0
    res.V_transition[:,:,:,N_t] = V_N_SS
    for it=(N_t-1):-1:1
        println("HH problem in ", it)
        #k = K_t[it]
        #l = L_t[it]
        #theta = θ_grid[it]
        #price_solve(prim, res, k, l, theta)
        #val_r_next = zeros(N, prim.nA, 2) .- Inf
        #val_w_next = zeros(N, prim.nA, 2) .- Inf
        #val_w = zeros(N, prim.nA, 2) .- Inf
        #val_r = zeros(N, prim.nA, 2) .- Inf
        for j in 1:2 
            for age in N:-1:1 
                for i in 1:prim.nA
                    a = A_grid[i]
                    val_prev = -Inf 
                    if age==66
                        #val_r[age, i, j] = utility_r(prim, a, 0, γ)
                        #V_store[age, i, j] = val_r[age, i, j]
                        res.V_transition[age, i, j, it] = utility_r_t(prim, a, 0, γ,it)
                        #println("age 66 calculation utility:", utility_r_t(prim, a, 0, γ,it))
                        res.g_0[age, i, j, it] = 0
                        #println("age: ", age)
                    elseif 45<age<66
                        #choice_lower = 1 
                        #=if i==1
                            #println("filling in ","age: ", age, " i: ", i, " j: ", j, " it: ", it)
                            res.V_transition[age, i, j, it] = utility_r_t(prim, a, 0, γ,it) + β*res.V_transition[age+1, 1, j, it+1]
                            res.g_0[age, i, j, it] = 0
                        else=#
                            for k in 1:prim.nA 
                                ap = A_grid[k]
                                #val_r_next[age, i, j] = utility_r(prim, a, ap, γ) + β*V_prior[age+1, k, j]
                                val_r_next = utility_r_t(prim, a, ap, γ,it) + β*res.V_transition[age+1, k, j, it+1]
                                #= if age==65 && it>28 && it<=29 && ap<=2.0
                                    println("age:",age," a:",a," ap:",ap, " val_r_next: ", val_r_next)
                                    println("utility:", utility_r_t(prim, a, ap, γ,it))
                                    println("V_transition:", res.V_transition[age+1, k, j, it+1])
                                end =#
                                #=if i>=1 && i<=2 && age==65 && it==29
                                    println("age: ", age, " i: ", i, " j: ", j, " it: ", it)
                                    println("utility:",utility_r_t(prim, a, ap, γ,it), " V_transition:", res.V_transition[age+1, k, j, it+1], "val_r_next: ", val_r_next)
                                    println("val prev: ", val_prev)
                                end =#
                                if val_r_next>val_prev #&& res.V_transition[age+1, k, j, it+1]!=0.0
                                    #val_r[age, i, j] = val_r_next[age, i, j]
                                    val_prev = val_r_next
                                    res.V_transition[age, i, j, it] = val_r_next
                                    res.g_0[age, i, j, it] = ap
                                    #choice_lower = k
                                end 
                                #end 
                                #val_r = val_r_next
                            end 
                        #println("policy replace with ap=", res.g_0[age, i, j, it]," age is ", age)
                        #V_store[age, i, j] = val_r[age, i, j] 
                        #println("age: ", age)
                    else
                        E = res.e[age, j]
                        #val_prev = -Inf
                        #choice_lower = 1
                        for k in 1:prim.nA 
                            ap = A_grid[k]
                            #val_w_next[age, i, j]  = utility_w(prim, a, ap, E, theta, γ) + β*(π[j,1]*V_prior[age+1, k, 1] + π[j,2]*V_prior[age+1, k, 2])
                            val_w_next = utility_w_t(prim, a, ap, E, θ_grid, γ,it) + β*(π[j,1]*res.V_transition[age+1, k, 1,it+1] + π[j,2]*res.V_transition[age+1, k, 2,it+1])
                            if val_w_next>val_prev
                                #val_w[age, i, j] = val_w_next[age, i, j]
                                val_prev = val_w_next
                                res.V_transition[age, i, j, it] = val_w_next
                                res.g_0[age, i, j,it] = ap
                                #res.labor[age, i, j] = labor(a,ap, E, theta, γ)
                                res.labor_transition[age, i, j, it] = labor_t(a,ap, E, θ_grid, γ,it)
                                #choice_lower = k
                            end 
                        end 
                        #println("policy replace with ap=",res.g_0[age,i,j,it]," age is ", age)
                        #V_store[age, i, j] = val_w[age, i, j]
                    end 
                end 
            end 
            #V_prior = V_store
        end 
    end 
    return res.g_0, res.V_transition 
end 

function Tstar_transition(prim::Primitives,res::Results; tol::Float64 = 1e-10, max_iter::Int64 = 1000, T::Int64)
    @unpack A_grid, nA, π, N, π_0, n = prim #unpack model primitives 
    @unpack val_function, F = res #unpack value
    trans_mat = zeros(N,nA*2,nA*2) #initialize transition matrix
        for age in 1:N 
            for j in 1:2
                for i in 1:nA
                    #ap = res.g_0[age,i,j,T+1]  # get next period's asset from policy function
                    ap = res.g_0[age,i,j,T]  # get next period's asset from policy function
                    ap_id = argmin(abs.(ap .- A_grid))

                    for jp in 1:2
                        trans_mat[age,i + nA*(j -1), ap_id + nA*(jp - 1)] =  π[j, jp]
                    end
                end 
            end
        end 

    μ_dist = zeros(N,nA*2)
    μ_dist[1,1] = π_0[1]
    μ_dist[1,nA+1] = π_0[2]

    for age in 2:N
        μ_dist[age,:] = trans_mat[age-1,:,:]' * μ_dist[age-1,:]
    end



    mu = zeros(N)
    mu[1] = 1
    for age in 1:N-1 
        mu[age+1] = mu[age]/(1+n)
    end
    res.mu_norm = mu./sum(mu)

    μ_dist = res.mu_norm.*μ_dist
    
    Gamma = reshape(μ_dist, (N,nA, 2))
    #println("sanity check: ", sum(Gamma))
    return Gamma 
end 



function update_capital_path(res,Gamma_0_SS, K_0_SS)
    K_t_new = zeros(N_t)
    println("initialize K ", K_t_new)
    #K_t_new[1] = K_0_SS
    for age in 1:prim.N
        for i in 1:prim.nA
            for j in 1:2 
                K_t_new[1] += prim.A_grid[i]*Gamma_0_SS[age, i, j]
            end 
        end 
    end 
    println("K new[1]: ", K_t_new[1])
    for it in 2:N_t
        Gamma_new = Tstar_transition(prim, res; tol=1e-10, max_iter=1000, T=it-1)
        for age in 1:prim.N
            for i in 1:prim.nA
                for j in 1:2
                    K_t_new[it] += prim.A_grid[i]*Gamma_new[age, i, j]
                end 
            end 
        end 
    end 
    return K_t_new
end 

function update_labor_path(prim,res, Gamma_0_SS, L_0_SS, L_N_SS)
    L_t_new = zeros(N_t)
    #L_t_new[1] = L_0_SS
    for age in 1:45 
        for i in 1:prim.nA
            for j in 1:2
                #E = res.e[age, j]
                #a = res.g_0[age, i, j, 1]
                #ap = res.g_0[age, i, j, 2]
                #=a = prim.A_grid[i]
                ap = res.g_0[age, i, j, 2]
                theta = θ_grid[2]
                l = labor(a, ap, E, theta, 0.42)
                L_t_new[2] += Gamma_0_SS[age, i, j]*E*l=#
                L_t_new[1] += Gamma_0_SS[age, i, j]*res.e[age,j]*res.labor_transition[age, i, j, 1]
            end 
        end 
    end 
    for it in 2:N_t-1
        Gamma_new = Tstar_transition(prim, res; tol=1e-10, max_iter=1000, T=it-1)
        for age in 1:45 
            for i in 1:prim.nA
                for j in 1:2
                   #= E = res.e[age, j]
                    a = res.g_0[age, i, j, n]
                    ap = res.g_0[age, i, j, n+1]
                    theta = θ_grid[n+1]
                    l = labor(a, ap, E, theta, 0.42)
                    L_t_new[n] += Gamma_new[age, i, j]*E*l =#
                    L_t_new[it] += Gamma_new[age, i, j]*res.e[age,j]*res.labor_transition[age, i, j, it]
                end 
            end 
        end
    end 
    L_t_new[N_t] = L_N_SS
    return L_t_new
end 


function transition_solve(prim, Gamma_0_SS, K_0_SS, L_0_SS, L_N_SS, V_N_SS, N_t, K_t, L_t, θ_grid; rho, eps)
    conv = 0 
    iter = 1 
    while conv==0 
        println("Iteration: ", iter)
        price_solve_transition(prim, res, K_t, L_t, θ_grid, N_t)
        println("K:", K_t)
        println("L:", L_t)
        println("wage:",res.w_transition)
        println("rate:",res.r_transition)
        println("pension",res.b_transition)
        bellman_transition(prim, res; V_N_SS, g_N_SS, N_t, K_t, L_t, θ_grid)
        K_t_new = update_capital_path(res,Gamma_0_SS, K_0_SS)
        L_t_new = update_labor_path(prim,res, Gamma_0_SS, L_0_SS, L_N_SS)
        ksum = sum(abs.(K_t_new - K_t))
        lsum = sum(abs.(L_t_new-L_t))
        if max(ksum,lsum)<eps 
            println("Transition path converged.")
            conv=1 
            K_t = K_t_new
            L_t = L_t_new
        else 
            println("Capital Error: ", abs.(K_t_new - K_t), "sum", ksum)
            println("Labor Error: ", abs.(L_t_new-L_t), "sum", lsum)
            K_t = rho.*K_t + (1-rho).*K_t_new
            L_t = rho.*L_t + (1-rho).*L_t_new
        end 
        iter+=1
    end 
    return K_t, L_t
end 