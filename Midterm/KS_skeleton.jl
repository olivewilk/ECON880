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

    nk::Int64 = 31
    k_min::Float64 = 0.001
    k_max::Float64 = 15.0
    k_grid::Vector{Float64} = range(k_min, length=nk, stop=k_max) # grid for capital, start coarse

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
    R²::Float64                             # R² for capital LOM

    K_path::Vector{Float64}                 # path of capital

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
    pgb::Float64 = 1.0 - (d_b-1.0)/d_b
    pbg::Float64 = 1.0 - (d_g-1.0)/d_g
    pbb::Float64 = (d_b-1.0)/d_b

    #transition probabilities for aggregate states and staying unemployed
    pgg00::Float64 = (d_ug-1.0)/d_ug
    pbb00::Float64 = (d_ub-1.0)/d_ub
    pbg00::Float64 = 1.25*pbb00
    pgb00::Float64 = 0.75*pgg00

    #transition probabilities for aggregate states and becoming employed
    pgg01::Float64 = (u_g - u_g*pgg00)/(1.0-u_g)
    pbb01::Float64 = (u_b - u_b*pbb00)/(1.0-u_b)
    pbg01::Float64 = (u_b - u_g*pbg00)/(1.0-u_g)
    pgb01::Float64 = (u_g - u_b*pgb00)/(1.0-u_b)

    #transition probabilities for aggregate states and becoming unemployed
    pgg10::Float64 = 1.0 - (d_ug-1.0)/d_ug
    pbb10::Float64 = 1.0 - (d_ub-1.0)/d_ub
    pbg10::Float64 = 1.0 - 1.25*pbb00
    pgb10::Float64 = 1.0 - 0.75*pgg00

    #transition probabilities for aggregate states and staying employed
    pgg11::Float64 = 1.0 - (u_g - u_g*pgg00)/(1.0-u_g)
    pbb11::Float64 = 1.0 - (u_b - u_b*pbb00)/(1.0-u_b)
    pbg11::Float64 = 1.0 - (u_b - u_g*pbg00)/(1.0-u_g)
    pgb11::Float64 = 1.0 - (u_g - u_b*pgb00)/(1.0-u_b)

    # Markov Transition Matrix
    Mgg::Array{Float64,2} = [pgg11 pgg01
                            pgg10 pgg00]

    Mbg::Array{Float64,2} = [pgb11 pgb01
                            pgb10 pgb00]

    Mgb::Array{Float64,2} = [pbg11 pbg01
                            pbg10 pbg00]

    Mbb ::Array{Float64,2} = [pbb11 pbb01
                             pbb10 pbb00]

    M::Array{Float64,2} = [pgg*Mgg pgb*Mgb
                          pbg*Mbg pbb*Mbb]

    # aggregate transition matrix
    Mzz::Array{Float64,2} = [pgg pbg
                            pgb pbb]
end


@with_kw struct Simulations
    T::Int64 = 11_000           # number of periods to simulate
    N::Int64 = 5_000            # number of agents to simulate
    seed::Int64 = 1234          # seed for random number generator

    V_tol::Float64 = 1e-9       # tolerance for value function iteration
    V_max_iter::Int64 = 10_000  # maximum number of iterations for value function

    burn::Int64 = 1_000         # number of periods to burn for regression
    reg_tol::Float64 = 1e-6     # tolerance for regression coefficients
    reg_max_iter::Int64 = 10_000 # maximum number of iterations for regression
    λ::Float64 = 0.5            # update parameter for regression coefficients

    K_initial::Float64 = 12.5   # initial aggregate capital
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

end


function Initialize()
    prim = Primitives()
    sho = Shocks()
    sim = Simulations()
    Z, E = DrawShocks(prim, sho, sim)

    V = zeros(prim.nk, prim.nϵ, prim.nK, prim.nz)
    k_policy = zeros(prim.nk, prim.nϵ, prim.nK, prim.nz)

    a₀ = 0.095
    a₁ = 0.085
    b₀ = 0.999
    b₁ = 0.999
    R² = 0.0

    K_path = zeros(sim.T)

    res = Results(Z, E, V, k_policy, a₀, a₁, b₀, b₁, R², K_path)
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

end


function VFI(prim::Primitives, res::Results, sim::Simulations)
    #=
    Iterate on the value function until convergence

    Args
    prim (Primitives): model parameters
    res (Results): results struct
    sim (Simulations): simulation parameters
    =#


end







########################### Part 4 - Solve model ###########################


function SimulateCapitalPath(prim::Primitives, res::Results, sim::Simulations)
    #=
    Simulate the path of K

    Args
    prim (Primitives): model parameters
    res (Results): results struct
    sim (Simulations): simulation parameters

    Returns
    K_path (Vector): path of capital
    =#


end


function EstimateRegression(prim::Primitives, res::Results, sim::Simulations)
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

end


function SolveModel()
    #=
    Solve the Krusell-Smith model

    Returns
    res (Results): results struct
    =#

end
