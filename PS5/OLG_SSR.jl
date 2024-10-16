#=
Author: Olivia
Date: October 2024 
=#
@with_kw struct Primitives
    N::Int64 = 66 #number of periods 
    n::Float64 = 0.011 #population growth rate//model period length 
    a_1::Float64 = 0.0 #initial assets at age 1 
    a_max::Float64 = 100.0 #max assets
    nA::Int64 = 2501 #number of asset grid points
    A_grid::Array{Float64,1} = collect(range(0.0, length = nA, stop = a_max)) #grid for assets
    #θ::Float64 = 0.11 #proportional labor income tax 
    #γ::Float64 = 0.42 #weight on consumption 
    σ::Float64 = 2.0 #coefficient of relative risk aversion
    #z::Array{Float64,1} = [3.0,0.5] #productivity states
    η::Array{Float64,1} = map(x->parse(Float64,x), readlines("PS5/ef.txt")) # deterministic age-efficiency profile
    π_hh::Float64 = 0.9261 #transition probability from high to high productivity
    π_ll::Float64 = 0.9811 #transition probability from low to low productivity
    π::Array{Float64,2} = [π_hh 1-π_hh; 1-π_ll π_ll] #transition matrix
    π_0::Array{Float64, 1} = [0.2037, 0.7963]  # Erodic distribution of Π
    α::Float64 = 0.36 #capital share of output
    δ::Float64 = 0.06 #depreciation rate
    #b::Float64 = 0.2 #pension 
    #w::Float64 = 1.05 #wage 
    #r::Float64 = 0.05 #interest rate
    β::Float64 = 0.97 #discount rate
end

#structure that holds model results
mutable struct Results
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
end

#function for initializing model primitives and results
function Initialize(θ::Float64, γ::Float64, z::Array{Float64,1})
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
    res=Results(val_function, val_w, val_r, pol_func, e, mu_norm, F,K,L, labor, w, r, b, θ, γ, z) #initialize vector results struct
    prim, res #return deliverables
end

function utility_r(prim, a, ap, γ)
   # @unpack σ, γ, r, b = prim
   @unpack σ = prim
    c = (1+res.r)*a + res.b - ap
    if c<0 
        return -Inf
    else 
        return (c^((1-σ)*γ))/(1-σ)  
    end 
end


function labor(a, ap,E, θ, γ)
    #@unpack w, θ, γ, r = prim
    num = γ*(1-θ)*E*res.w - (1-γ)*((1+res.r)*a-ap) 
    denom = (1-θ)*res.w*E
    interior_solution = num/denom
    return min(1, max(0, interior_solution))
end 


function utility_w(prim, a, ap, E, θ, γ)
    #@unpack σ, γ, w, θ, r = prim
    @unpack σ = prim
    l = labor(a, ap, E,θ, γ)
    c = res.w*(1-θ)*E*l + (1+res.r)*a - ap 
    if (c>0 && l>=0 && l<=1 )
        return (((c^γ)*(1-l)^(1-γ))^(1-σ))/(1-σ)
    else
        return -Inf
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
                    val_r[age, i, j] = utility_r(prim, a, 0, γ)
                    res.val_function[age, i, j] = val_r[age, i, j]
                    res.pol_func[age, i, j] = 0
                    #println("age: ", age)
                elseif 45<age<66 
                    choice_lower = 1 
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
function price_solve(res, K,L, θ)
    @unpack α= prim
    res.w = (1-α)*(K^α)*(L^(-α)) #MPL 
    res.r = α*(K^(α-1))*(L^(1-α)) - prim.δ #MPK 
    res.b = (θ*res.w*L)/(sum(res.mu_norm[46:prim.N,:,:]))#pension 
end 

#solve model 
function model_solve(prim, res; θ::Float64, z::Array{Float64,1}, γ::Float64,eps::Float64, λ::Float64, K::Float64, L::Float64) 
    
    K_0 = K #initial guess for capital supply
    L_0 = L #initial guess for labor supply
    price_solve(res, K_0, L_0, θ) #solve for prices

    max_iter = 100 #maximum number of iterations
    iter = 0 #initialize counter

    while iter<max_iter 
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
            price_solve(res, K_0, L_0, θ)
        else #converged 
            println("Model converged in ", iter, " iterations.")
            break 
        end 
    end 

    res.K = K_0 
    res.L = L_0    
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
