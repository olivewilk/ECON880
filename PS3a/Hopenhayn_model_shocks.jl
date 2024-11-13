#=
Author: Olivia
Date: September 2024 
=#

@with_kw struct Primitives
    eulergamma::Float64 = 0.577216 #euler constant
    α::Float64 = 1.0 #variance of the distribution 
    tol::Float64 = 1e-6 #tolerance level for convergence
    tol_p::Float64 = 1e-4 #tolerance level for price convergence
    err::Float64 = 100.0 #initialize error level
    c_f::Float64 = 10 #fixed cost of running a business
    c_e::Float64 = 5 #cost of entry
    s_grid::Array{Float64,1} = [3.98e-4, 3.58, 6.82, 12.18, 18.79] #productivity of firms
    ns::Int64 = length(s_grid) #number of productivity states
    v_entrant::Array{Float64,1} = [0.37,0.4631,0.1102,0.0504, 0.0063] #entrant distribution 
    β::Float64 = 0.8 #discount rate
    θ::Float64 = 0.64 #probability of staying with current productivity
    η::Float64 = 0.5 #probability that entrant firm is high productivity
    F::Matrix{Float64} = [0.6598 0.2600 0.0416 0.0331 0.0055; 
                           0.1997 0.7201 0.0420 0.0326 0.0056;
                           0.2000 0.2000 0.5555 0.0344 0.0101; 
                           0.2000 0.2000 0.2502 0.3397 0.0101; 
                           0.2000 0.2000 0.2500 0.3400 0.0100] #markov process for firm size/productivity 
    A::Float64 = 1/200 #productivity 
end

#structure that holds model results
mutable struct Results_v
    W::Array{Float64, 1} #firm value function W
    #val_stay::Array{Float64, 2} #value function for staying in business
    #val_exit::Array{Float64, 2} #value function for exiting business
    pol_func::Array{Float64, 1} #policy function (choice of x which is the minimum productivity level to stay in business)
    mu_dist::Array{Float64,1} #cross sectional firm distribution
    π::Array{Float64,1} # individual profit
    N::Array{Float64,1} #labor demand
end

mutable struct Results_a
    p::Float64 #Price 
    M::Float64 #mass of entrants 
    #market clearing stuff 
    Ld::Float64 #demand for labor (aggregate)
    Ls::Float64 #supply of labor (aggregate)
    Π::Float64 #aggregate profits
end

#function for initializing model primitives and results
function Initialize()
    prim = Primitives() #initialize primtiives
    W = zeros(prim.ns) .+2 #initial value function guess
    pol_func = zeros(prim.ns) #initial policy function guess 
    mu_dist = ones(prim.ns) #initial distribution guess 
    π = zeros(prim.ns) #initial profit guess
    N = zeros(prim.ns) #initial labor demand guess
    p = 1.0 
    Ld = 0.0 
    Ls = 0.0
    Π = 0.0
    M = 0.0
    resvectors = Results_v(W, pol_func, mu_dist, π, N) #initialize vector results struct
    resfloats = Results_a(p, M, Ld, Ls, Π) #initialize non-vector results struct
    prim, resvectors, resfloats #return deliverables
end

function firm_labor_demand(prim, resvectors, resfloats, s)
    @unpack p = resfloats 
    @unpack c_f, θ = prim 
    labordemand = (θ*p*s)^((1/(1-θ))) 
end 

function profit(prim,  resvectors, resfloats, s, i)
    @unpack p = resfloats 
    @unpack N = resvectors
    @unpack c_f, θ = prim  
    n=N[i]
    profit=p*s*n^θ -n -p*c_f
end 

function Bellman(prim,  resvectors, resfloats)
    @unpack p = resfloats 
    @unpack c_f, θ, α, eulergamma = prim 

    #pol_func_next = zeros(prim.ns)
    W_next = zeros(prim.ns)

    for i in 1:prim.ns
        s = prim.s_grid[i]
        resvectors.N[i] = firm_labor_demand(prim,  resvectors, resfloats, s)
        resvectors.π[i] = profit(prim,  resvectors, resfloats, s, i)

        W_exit = resvectors.π[i] # value of exiting is just profit 
        W_stay = resvectors.π[i] 
        for j = 1:prim.ns 
            W_stay += prim.β*sum(prim.F[i,j].*resvectors.W[j]) #value of staying is profit + continuation value 
        end 

        # Using log-sum-exp trick
        choice = max(α * W_stay, α * W_exit)

        W_next[i] = (eulergamma / α) + (1/α) * (choice + log(exp(α * W_stay - choice) + exp(α * W_exit - choice)))

        resvectors.pol_func[i] = exp(α * W_exit - choice) / (exp.(α * W_stay - choice) + exp(α * W_exit - choice))
    end 
    return W_next
end 

function W_iterate(prim,  resvectors, resfloats)
    @unpack tol, err = prim
    n = 0 #counter
    while err>tol #begin iteration
        #println("price is: ", resfloats.p)
        W_next = Bellman(prim,  resvectors, resfloats) 
        err = maximum(abs.(W_next.-resvectors.W))/abs(W_next[prim.ns])  #reset error level
        resvectors.W .= W_next
        n += 1
    end
    #println("Value function converged in ", n, " iterations.")
end 

#entry depends on price, need positive entry 
function Entry_decision(prim, resvectors, resfloats)
    @unpack c_e, v_entrant = prim

    return sum(resvectors.W .* v_entrant)/resfloats.p - c_e
end     

function Price_solve(prim, resvectors, resfloats)
    @unpack tol, tol_p, err = prim

    #solve for price using bisection 

    p_low = 0.0
    p_high = 10.0 
    p_mid = (p_low + p_high)/2

    entry = 100 
    n =1 

    while abs(entry) > tol_p && n < 10000
        #println("Entry is: ", entry)
        W_iterate(prim, resvectors, resfloats) #solve incumbent firm problem 
        entry = Entry_decision(prim, resvectors, resfloats) #solve entry decision

        if entry < 0 #entry should be positive 
            p_low = p_mid
        else
            p_high = p_mid
        end 

        p_mid = (p_low + p_high)/2 #recompute midpoint
        resfloats.p = p_mid #replace p with midpoint value 
        n += 1
    end 

    resfloats.p = p_mid #replace p with midpoint value
    println("Price converged in ", n, " iterations. It is equal to ", resfloats.p)
end 

function Tstar(prim,resvectors, resfloats)
    @unpack ns, F = prim #unpack model primitives 
    @unpack W, pol_func, mu_dist = resvectors #unpack results 
    #new mu distribution should equal old firms still in plus new entrants 
    
    mu_next = zeros(ns)
    for i in 1:ns
        for j in 1:ns
            #sum over current period i 
            mu_next[j] += (1 - pol_func[i])*F[i,j]*mu_dist[i] + (1 - pol_func[i])*F[i,j]*resfloats.M*prim.v_entrant[i]
        end 
    end 
    return mu_next
end 

function Find_mu(prim, resvectors, resfloats)
    @unpack ns, F, tol, err = prim #unpack model primitives 
    @unpack W, pol_func, mu_dist = resvectors #unpack results 
    n=1 
    max_iter = 1000
    while err>tol && n<max_iter
        mu_next = Tstar(prim, resvectors, resfloats) #update distribution
        err = maximum(abs.(mu_next.-mu_dist))
        resvectors.mu_dist = mu_next
        n+=1
    end 
    return resvectors.mu_dist
end 

function Market_clearing(prim, resvectors, resfloats) 
    @unpack v_entrant, A = prim
    #@unpack mu_dist = resvectors 
    @unpack M = resfloats

    resvectors.mu_dist = Find_mu(prim, resvectors, resfloats) #update distribution

    #labor demand 
    resfloats.Ld = sum(resvectors.N .* resvectors.mu_dist) + M * sum(resvectors.N .* v_entrant) # labor demand is sum of labor demand of incumbents and entrants
    #profits 
    resfloats.Π = sum(resvectors.π .* resvectors.mu_dist) + M * sum(resvectors.π .* v_entrant) # profits is sum of profits of incumbents and entrants
    #labor supply 
    resfloats.Ls = 1/A - resfloats.Π #labor supply is 1/A - profits

   return resfloats.Ld - resfloats.Ls #market clearing condition
    
end 

function New_entrants(prim, resvectors, resfloats)
    @unpack tol = prim 
    # solve for M using bisection
    M_low = 0.0
    M_high = 20.0 
    resfloats.M = (M_low + M_high)/2

    clearing_condition = 100 
    n = 1 

    while abs(clearing_condition)>tol && n < 10000

        clearing_condition = Market_clearing(prim, resvectors, resfloats) #check market clearing condition
        #println("Clearing condition is: ", clearing_condition)
        if clearing_condition < 0 #raise lower bound on entrants 
            M_low = resfloats.M
        else #lower upper bound on entrants 
            M_high = resfloats.M 
        end
        resfloats.M = (M_low + M_high)/2 #recompute midpoint
        n+=1 
        #println("Mass of entrants converged in ", n, " iterations. It is equal to ", resfloats.M)
    end 
    return resfloats.M 
end 

function Solve_model(prim, resvectors, resfloats)
    Price_solve(prim, resvectors, resfloats) #solve for price
    New_entrants(prim, resvectors, resfloats) #solve for entrants
end
