#=
This file contains skeleton code for solving the Krusell-Smith model.

Table of Contents:
1. Setup model
    - 1.1. Primitives struct
    - 1.2. Results struct
2. Generate shocks
    - 2.1. Shocks struct
    - 2.2. Simulations struct
    - 2.3. functions to generate shocks
3. Solve HH problem
    - 3.1 utility function
    - 3.2 Bellman operator
    - 3.3 VFI algorithm
4. Solve model
    - 4.1 Simulate capital path
    - 4.2 Estimate regression
    - 4.3 Solve model
=#

using Parameters, LinearAlgebra, Random, Interpolations, Optim
using GLM, DataFrames, Plots


######################### Part 1 - setup model #########################


@with_kw struct Primitives
    β::Float64 = 0.99           # discount factor
    α::Float64 = 0.36           # capital share
    δ::Float64 = 0.025          # depreciation rate
    ē::Float64 = 0.3271         # labor productivity

    z_grid::Vector{Float64} = [1.01, .99]      # grid for TFP shocks
    z_g::Float64 = z_grid[1]
    z_b::Float64 = z_grid[2]
    nz::Int64 = length(z_grid)

    ϵ_grid::Vector{Float64} = [1, 0]           # grid for employment shocks
    nϵ::Int64 = length(ϵ_grid)

    nk::Int64 = 51
    k_min::Float64 = 0.001
    k_max::Float64 = 15.0
    k_grid::Vector{Float64} = range(k_min, length=nk,stop=k_max) # grid for capital, start coarse

    nK::Int64 = 17
    K_min::Float64 = 11.0
    K_max::Float64 = 15.0
    K_grid::Vector{Float64} = range(K_min, length=nK, stop=K_max) # grid for aggregate capital, start coarse

end

@with_kw mutable struct Results
    Z::Vector{Float64}                      # aggregate shocks
    E::Matrix{Float64}                      # employment shocks

    V::Array{Float64, 4}                    # value function, dims (k, ϵ, K, z)
    k_policy::Array{Float64, 4}             # capital policy, similar to V

    a₀::Float64                             # constant for capital LOM, good times
    a₁::Float64                             # coefficient for capital LOM, good times
    b₀::Float64                             # constant for capital LOM, bad times
    b₁::Float64                             # coefficient for capital LOM, bad times
    R²_a::Float64                             # R² for capital LOM - good times 
    R²_b::Float64                             # R² for capital LOM - bad times 

    V_data::Array{Float64, 2}               # matrix of model decisions (policy data)
    K_estimate::Vector{Float64}             # estimated path of aggregate capital (data)
    K_path::Vector{Float64}                 # simulated path of aggregate capital
    policy_path::Array{Float64, 2}          # policy simulation
end



######################### Part 2 - generate shocks #########################


@with_kw struct Shocks
    #parameters of transition matrix:
    d_ug::Float64 = 1.5 # Unemp Duration (Good Times)
    u_g::Float64 = 0.04 # Fraction Unemp (Good Times)
    d_g::Float64 = 8.0  # Duration (Good Times)
    u_b::Float64 = 0.1  # Fraction Unemp (Bad Times)
    d_b::Float64 = 8.0  # Duration (Bad Times)
    d_ub::Float64 = 2.5 # Unemp Duration (Bad Times)

    #transition probabilities for aggregate states
    pgg::Float64 = (d_g-1.0)/d_g
    pgb::Float64 = 1.0 - (d_g-1.0)/d_g
    pbg::Float64 = 1.0 - (d_b-1.0)/d_b
    pbb::Float64 = (d_b-1.0)/d_b

    #transition probabilities for aggregate states and staying unemployed
    pgg00::Float64 = (d_ug-1.0)/d_ug
    pbb00::Float64 = (d_ub-1.0)/d_ub
    pbg00::Float64 = 0.75*pgg00
    pgb00::Float64 = 1.25*pbb00

    #transition probabilities for aggregate states and becoming employed
    #I think this is 1-staying unemployed 
    pgg01::Float64 = 1 - pgg00 
    pbb01::Float64 = 1 - pbb00 
    pbg01::Float64 = 1 - pbg00 
    pgb01::Float64 = 1 - pgb00 

    #transition probabilities for aggregate states and becoming unemployed
    pgg10::Float64 = (1/(1-u_g))*(u_g - u_g*pgg00)
    pbb10::Float64 = (1/(1-u_b))*(u_b - u_b*pbb00)
    pbg10::Float64 = (1/(1-u_b))*(u_g - u_b*pbg00)
    pgb10::Float64 = (1/(1-u_g))*(u_b - u_g*pgb00)

    #transition probabilities for aggregate states and staying employed
    pgg11::Float64 = 1.0 - pgg10
    pbb11::Float64 = 1.0 - pbb10
    pbg11::Float64 = 1.0 - pbg10 
    pgb11::Float64 = 1.0 - pgb10


    # Markov Transition Matrix
    Mgg::Array{Float64,2} = [pgg11 pgg10
                            pgg01 pgg00]

    Mgb::Array{Float64,2} = [pgb11 pgb10
                            pgb01 pgb00]

    Mbg::Array{Float64,2} = [pbg11 pbg10
                            pbg01 pbg00]

    Mbb ::Array{Float64,2} = [pbb11 pbb10
                             pbb01 pbb00]

    M::Array{Float64,2} = [pgg*Mgg pgb*Mgb
                          pbg*Mbg pbb*Mbb]  #Markov transition matrix
    
    Pi::Array{Float64,2} = [pgg*pgg11 pgb*pgb11 pgg*pgg10 pgb*pgb10
                            pbg*pbg11 pbb*pbb11 pbg*pbg10 pbb*pbb10
                            pgg*pgg01 pgb*pgb01 pgg*pgg00 pgb*pgb00
                            pbg*pbg01 pbb*pbb01 pbg*pbg00 pbb*pbb00]
                            

    # aggregate transition matrix
    Mzz::Array{Float64,2} = [pgg pgb
                            pbg pbb]
    

end


@with_kw struct Simulations
    T::Int64 = 11_000           # number of periods to simulate
    #T::Int64 = 2000           # number of periods to simulate
    N::Int64 = 5_000            # number of agents to simulate
    seed::Int64 = 1234          # seed for random number generator

    V_tol::Float64 = 1e-9       # tolerance for value function iteration
    V_max_iter::Int64 = 10_000  # maximum number of iterations for value function

    burn::Int64 = 1_000         # number of periods to burn for regression
    reg_tol::Float64 = 1e-6     # tolerance for regression coefficients
    reg_max_iter::Int64 = 10_000 # maximum number of iterations for regression
    λ::Float64 = 0.5            # update parameter for regression coefficients

    K_initial::Float64 = 11.55   # initial aggregate capital

end


function sim_Markov(current_index::Int64, Π::Matrix{Float64})
    #=
    Simulate the next state index given the current state index and Markov transition matrix

    Args
    current_index (Int): index current state
    Π (Matrix): Markov transition matrix, rows must sum to 1
    
    Returns
    next_index (Int): next state index
    =#
    
    # Generate a random number between 0 and 1
    rand_num = rand()

    # Get the cumulative sum of the probabilities in the current row
    cumulative_sum = cumsum(Π[current_index, :])
    #println(cumulative_sum)
    # Find the next state index based on the random number
    next_index = searchsortedfirst(cumulative_sum, rand_num)

    return next_index
end


function DrawShocks(prim::Primitives, sho::Shocks, sim::Simulations)
    #=
    Generate a sequence of aggregate shocks

    Args
    prim (Primitives): model parameters
    sho (Shocks): shock parameters
    sim (Simulations): simulation parameters

    Returns
    Z (Vector): matrix of aggregate shocks, length T
    E (Matrix): matrix of employment shocks, size N x T
    =#

    Z = zeros(sim.T)
    E = zeros(sim.N, sim.T)

    Z[1] = prim.z_g # start in good times 
    for t in 2:sim.T
        if Z[t-1] == prim.z_g
            current_index = 1
        else 
            current_index = 2
        end
        next = sim_Markov(current_index , sho.Mzz)
        Z[t] = prim.z_grid[next]
    end 
    E[:,1] .= prim.ϵ_grid[1]
    for t in 2:sim.T
        for n in 1:sim.N
            if Z[t-1] == prim.z_g && Z[t] == prim.z_g && E[n,t-1] == prim.ϵ_grid[1]
                next = sim_Markov(1, sho.Mgg)
            elseif Z[t-1] == prim.z_g && Z[t] == prim.z_g && E[n,t-1] == prim.ϵ_grid[2]
                next = sim_Markov(2, sho.Mgg)
            elseif Z[t-1] == prim.z_g && Z[t] == prim.z_b && E[n,t-1] == prim.ϵ_grid[1]
                next = sim_Markov(1, sho.Mgb)
            elseif Z[t-1] == prim.z_g && Z[t] == prim.z_b && E[n,t-1] == prim.ϵ_grid[2]
                next = sim_Markov(2, sho.Mgb)
            elseif Z[t-1] == prim.z_b && Z[t] == prim.z_g && E[n,t-1] == prim.ϵ_grid[1]
                next = sim_Markov(1, sho.Mbg)
            elseif Z[t-1] == prim.z_b && Z[t] == prim.z_g && E[n,t-1] == prim.ϵ_grid[2]
                next = sim_Markov(2, sho.Mbg)
            elseif Z[t-1] == prim.z_b && Z[t] == prim.z_b && E[n,t-1] == prim.ϵ_grid[1]
                next = sim_Markov(1, sho.Mbb)
            elseif Z[t-1] == prim.z_b && Z[t] == prim.z_b && E[n,t-1] == prim.ϵ_grid[2]
                next = sim_Markov(2, sho.Mbb)
            end 

            if next == 1
                E[n,t] = prim.ϵ_grid[1]
            else 
                E[n,t] = prim.ϵ_grid[2]
            end
        end 
    end 
    return Z, E
end


function Initialize()
    prim = Primitives()
    sho = Shocks()
    sim = Simulations()
    Random.seed!(sim.seed)
    Z, E = DrawShocks(prim, sho, sim)

    V = zeros(prim.nk, prim.nϵ, prim.nK, prim.nz)
    k_policy = zeros(prim.nk, prim.nϵ, prim.nK, prim.nz)

    a₀ = 0.095
    a₁ = 0.999
    b₀ = 0.085
    b₁ = 0.999
    R²_a = 0.0
    R²_b = 0.0

    V_data= zeros(sim.N,sim.T-sim.burn) # matrix of model decisions (policy data)
    K_estimate = zeros(sim.T-sim.burn) # estimated path of aggregate capital (data)
    policy_path = zeros(sim.T)
    K_path = zeros(sim.N, sim.T)
    
    res = Results(Z, E, V, k_policy, a₀, a₁, b₀, b₁, R²_a, R²_b, V_data, K_estimate, policy_path, K_path)
    return prim, res, sim, sho
end




######################### Part 3 - HH Problem #########################

function u(c::Float64; ε::Float64 = 1e-16)
    #=
    Define the utility function, with stitching function for numerical optimization

    Args
    c (Float): consumption
    ε (Float): small number for numerical stability (optional)

    Returns
    u (Float): utility value
    =#
    if c > ε
        return log(c)
    else # a linear approximation for stitching function
        # ensures smoothness for numerical optimization
        return log(ε) - (ε - c) / ε
    end
end


function Bellman(prim::Primitives, res::Results, sho::Shocks)
    #= 
    Solve the Bellman equation for the household problem

    Args
    prim (Primitives): model parameters
    res (Results): results struct
    sho (Shocks): shock parameters

    Returns
    V_next (Array): updated value function
    k_next (Array): updated capital policy function
    =#

    @unpack_Primitives prim
    @unpack_Results res

    V_next = zeros(nk, nϵ, nK, nz)
    k_next = zeros(nk, nϵ, nK, nz)

    # bilinear interpolation 
    interpg0 = interpolate(res.V[:,2,:,1], BSpline(Linear())) # baseline interpolation of value function
    extrapg0 = extrapolate(interpg0, Interpolations.Flat()) # gives linear extrapolation off grid 
    V_interpg0 = scale(extrapg0, range(k_min, length=nk, k_max), range(K_min, length=nK, K_max)) # has to be scaled on increasing range object

    interpg1 = interpolate(res.V[:,1,:,1], BSpline(Linear())) # baseline interpolation of value function
    extrapg1 = extrapolate(interpg1, Interpolations.Flat()) # gives linear extrapolation off grid
    V_interpg1 = scale(extrapg1, range(k_min, length=nk, k_max), range(K_min, length=nK, K_max)) # has to be scaled on increasing range object

    interpb0 = interpolate(res.V[:,2,:,2], BSpline(Linear())) # baseline interpolation of value function
    extrapb0 = extrapolate(interpb0, Interpolations.Flat()) # gives linear extrapolation off grid
    V_interpb0 = scale(extrapb0, range(k_min, length=nk, k_max), range(K_min, length=nK, K_max)) # has to be scaled on increasing range object

    interpb1 = interpolate(res.V[:,1,:,2], BSpline(Linear())) # baseline interpolation of value function
    extrapb1 = extrapolate(interpb1, Interpolations.Flat()) # gives linear extrapolation off grid
    V_interpb1 = scale(extrapb1, range(k_min, length=nk, k_max), range(K_min, length=nK, K_max)) # has to be scaled on increasing range object
    
    
    for (z_index, z) in enumerate(prim.z_grid)
        if z_index==1
            L = prim.ē * (1 - sho.u_g)
        else 
            L= prim.ē * (1 - sho.u_b)
        end  
        for (ϵ_index, ϵ) in enumerate(ϵ_grid)
            p = sho.M[ϵ_index + 2*z_index - 2 , :]
            for (K_index, K) in enumerate(K_grid)

                r = α * z * (K / L) ^ (α - 1) 
                w = (1 - α) * z * (K / L) ^ α

                if z_index == 1
                    Kp = exp(a₀ + a₁ * log(K)) #find K prime value 
                else 
                    Kp = exp(b₀ + b₁ * log(K)) #find K prime value 
                end   

                for (k_index, k) in enumerate(k_grid)


                    budget =  w * ϵ + (1 + r - δ) * k

                    obj(k_prime) = -u(budget - k_prime) - β * (p[1] * V_interpg1(k_prime,Kp) + p[2] * V_interpg0(k_prime, Kp) + p[3] * V_interpb1(k_prime, Kp) + p[4] * V_interpb0(k_prime,Kp))
                    res = optimize(obj, 0.0, budget)

                    if res.converged
                        V_next[k_index, ϵ_index, K_index, z_index] = -res.minimum
                        k_next[k_index, ϵ_index, K_index, z_index] = res.minimizer
                    else
                        error("Optimization did not converge")
                    end
                end
            end
        end 
    end 

    return V_next , k_next

end


function VFI(prim::Primitives, res::Results, sim::Simulations, sho::Shocks; tol=1e-9, max_iter=10000)
    #=
    Iterate on the value function until convergence

    Args
    prim (Primitives): model parameters
    res (Results): results struct
    sim (Simulations): simulation parameters
    =#
        
    error = 100 * tol
    iter = 0

    while error > tol && iter < max_iter
        V_next, k_next = Bellman(prim, res, sho)
        error = maximum(abs.(V_next - res.V))
        println("error: ",error)
        res.V = V_next
        res.k_policy = k_next
        iter += 1
    end

    if iter == max_iter
        println("Maximum iterations reached in VFI")
    elseif error < tol
        println("Converged in VFI after $iter iterations")
    end
end







########################### Part 4 - Solve model ###########################


function SimulateCapitalPath(prim::Primitives, res::Results, sim::Simulations, sho::Shocks)
    #=
    Simulate the path of K

    Args
    prim (Primitives): model parameters
    res (Results): results struct
    sim (Simulations): simulation parameters

    Returns
    K_path (Vector): path of capital
    =#
    
    @unpack_Primitives prim
    @unpack_Results res
    @unpack_Simulations sim
    @unpack_Shocks sho

    K_path_new = zeros(sim.T)
    K_path_new[1] = sim.K_initial #initial aggregate capital stock
    println("a0: ", res.a₀, " a1: ", res.a₁, " b0: ", res.b₀, " b1: ", res.b₁)
    policy_path_new = zeros(sim.N, sim.T)
    policy_path_new[:,1] .= sim.K_initial #everyone starts with the initial aggregate capital stock


    #interpolate the policy grid from the bellman 
    ig0 = interpolate(res.k_policy[:,2,:,1], BSpline(Linear())) # baseline interpolation of policy function
    eg0 = extrapolate(ig0,  Interpolations.Flat()) # gives linear extrapolation off grid
    K_g0 = scale(eg0, range(k_min, length=nk, k_max), range(K_min, length=nK, K_max)) # has to be scaled on increasing range object

    ig1 = interpolate(res.k_policy[:,1,:,1], BSpline(Linear())) # baseline interpolation of policy function
    eg1 = extrapolate(ig1,  Interpolations.Flat()) # gives linear extrapolation off grid
    K_g1 = scale(eg1, range(k_min, length=nk, k_max), range(K_min, length=nK, K_max)) # has to be scaled on increasing range object

    ib0 = interpolate(res.k_policy[:,2,:,2], BSpline(Linear())) # baseline interpolation of policy function
    eb0 = extrapolate(ib0,  Interpolations.Flat()) # gives linear extrapolation off grid
    K_b0 = scale(eb0, range(k_min, length=nk, k_max), range(K_min, length=nK, K_max)) # has to be scaled on increasing range object

    ib1 = interpolate(res.k_policy[:,1,:,2], BSpline(Linear())) # baseline interpolation of policy function
    eb1 = extrapolate(ib1,  Interpolations.Flat()) # gives linear extrapolation off grid
    K_b1 = scale(eb1, range(k_min, length=nk, k_max), range(K_min, length=nK, K_max)) # has to be scaled on increasing range object

    for t in 2:T 
        for n in 1:N 
            if E[n,t-1] == 1 && Z[t-1] == z_g
                policy_path_new[n,t] = K_g1(policy_path_new[n,t-1], K_path_new[t-1])
            elseif E[n,t-1] == 0 && Z[t-1] == z_g
                policy_path_new[n,t] = K_g0(policy_path_new[n,t-1], K_path_new[t-1])
            elseif E[n,t-1] == 1 && Z[t-1] == z_b
                policy_path_new[n,t] = K_b1(policy_path_new[n,t-1], K_path_new[t-1])
            else 
                policy_path_new[n,t] = K_b0(policy_path_new[n,t-1], K_path_new[t-1])
            end 
        end 
        K_path_new[t] = sum(policy_path_new[:,t])/N
    end 

    V_data_new = zeros(sim.N, sim.T-sim.burn)
    V_data_new[:,:] = policy_path_new[:, sim.burn+1:end]
    K_estimate_new = zeros(sim.T-sim.burn)
    K_estimate_new[:] = sum(V_data_new, dims=1)/sim.N
    return V_data_new, K_estimate_new, K_path_new
 
end


function EstimateRegression(prim::Primitives, res::Results, sim::Simulations; tol=.01)
    #=
    Estimate the law of motion for capital with log-log regression

    Args
    prim (Primitives): model parameters
    res (Results): results struct
    sim (Simulations): simulation parameters

    Returns
    a₀ (Float): constant for capital LOM, good times
    a₁ (Float): coefficient for capital LOM, good times
    b₀ (Float): constant for capital LOM, bad times
    b₁ (Float): coefficient for capital LOM, bad times
    R² (Float): R² for capital LOM
    =#

    @unpack_Primitives prim
    @unpack_Results res 
    @unpack_Simulations sim 

    err = 100 
    iter = 1 
    max_iter = 100

    while err>tol && iter<max_iter
        println("iteration: ", iter)

        VFI(prim, res, sim, sho; tol=1e-4, max_iter=15000)

        V_data_new, K_new, K_path_new= SimulateCapitalPath(prim, res, sim, sho)

        res.V_data = V_data_new 
        res.K_estimate = K_new

        Plots.plot(sim.burn+1:sim.T, res.K_estimate, title="Simulated Capital Stock over time", color="cadetblue", linewidth=:2.0)
        
        #regression for good times 
        indicator_g = zeros(sim.T-sim.burn)
        indicator_b = zeros(sim.T-sim.burn)
        for t in 1 : sim.T-sim.burn
            if res.Z[t]==prim.z_g
                indicator_g[t] = 1 
                indicator_b[t] = 0
            else 
                indicator_g[t] = 0 
                indicator_b[t] = 1
            end 
        end 
    
        X = log.(res.K_estimate[1:end-1])
     
        y = log.(res.K_estimate[2:end])
        data = DataFrame(x=X, y=y, zb_d =indicator_b[1:end-1], zg_d=indicator_g[1:end-1])
        filtered_data_g = data[data.zg_d .== 1, :]
        filtered_data_b = data[data.zb_d .== 1, :]
        #regression 
        model_g = lm(@formula(y ~ x), filtered_data_g)
        a = coef(model_g)
        a0 = a[1]
        a1 = a[2]
        r_sqr_g = r2(model_g)
        model_b = lm(@formula(y ~ x), filtered_data_b)
        b= coef(model_b)
        b0 = b[1]
        b1 = b[2]
        r_sqr_b = r2(model_b)
        
        #distance 
        errors = [abs(res.a₀-a0), abs(res.a₁-a1), abs(res.b₀-b0),abs(res.b₁-b1)]
        println("a0 old: ", res.a₀, " a0 new: ", a0," difference: ",abs(res.a₀-a0), " a1 old: ", res.a₁, " a1 new: ", a1, " difference: ", abs(res.a₁-a1))
        println("b0 old: ", res.b₀, " b0 new: ", b0," difference: ",abs(res.b₀-b0), " b1 old: ", res.b₁, " b1 new: ", b1, " difference: ", abs(res.b₁-b1))
        err = maximum(errors)
        r2_min = minimum([r_sqr_b, r_sqr_g])
        println("r2g: ", r_sqr_g, "r2b: ", r_sqr_b)
        if r2_min<0.9 
            print("not a good fit, r2=", r2_min)
            err+=1 #dont  let loop break
        end 
        if iter>max_iter
            print("max iterations reached")
            break 
        end
        res.a₀=0.5*a0 + 0.5*res.a₀
        res.a₁=0.5*a1 + 0.5*res.a₁
        res.b₀=0.5*b0 + 0.5*res.b₀
        res.b₁=0.5*b1 + 0.5*res.b₁
        res.R²_a = r_sqr_g
        res.R²_b = r_sqr_b
        iter+=1
    end 
end
