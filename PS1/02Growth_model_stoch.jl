#loop over productivity and capital grids
#instead of vector value function and policy function, have matrix values nk x nz 
#compute expected value function in each step rather than deterministic 
#parallelize over z 

#keyword-enabled structure to hold model primitives
@with_kw struct Primitives
    β::Float64 = 0.99 #discount rate
    δ::Float64 = 0.025 #depreciation rate
    α::Float64 = 0.36 #capital share
    k_min::Float64 = 0.01 #capital lower bound
    k_max::Float64 = 90.0 #capital upper bound
    nk::Int64 = 1000 #number of capital grid points
    k_grid::Array{Float64,1} = collect(range(k_min, length = nk, stop = k_max)) #capital grid
    z_min::Float64 = 0.2 #productivity low
    z_max::Float64 = 1.25 #productivity high
    nz::Int64 = 2 #number of production grid points
    z_grid::Array{Float64,1} = collect(range(z_max, length = nz, stop = z_min)) #productivity grid
    Γ::Array{Float64,2} = [0.977 0.023; 0.074 0.926]
end

#structure that holds model results
mutable struct Results
    val_func::Array{Float64, 2} #value function
    pol_func::Array{Float64, 2} #policy function
end

#function for initializing model primitives and results
function Initialize()
    prim = Primitives() #initialize primtiives
    val_func = zeros(prim.nk, prim.nz) #initial value function guess
    pol_func = zeros(prim.nk, prim.nz ) #initial policy function guess
    res = Results(val_func, pol_func) #initialize results struct
    prim, res #return deliverables
end

#Bellman Operator
function Bellman(prim::Primitives,res::Results)
    @unpack val_func = res #unpack value function
    @unpack z_grid, k_grid, β, δ, α, nk, nz, Γ = prim #unpack model primitives
    v_next = zeros(nk, nz) #next guess of value function to fill
    for z_index = 1:nz 
        choice_lower = 1 #for exploiting monotonicity of policy function
        z = z_grid[z_index]
        for k_index = 1:nk
            k = k_grid[k_index] #value of k
            candidate_max = -Inf #bad candidate max
            budget = z*k^α + (1-δ)*k #budget

            for kp_index in choice_lower:nk #loop over possible selections of k', exploiting monotonicity of policy function
                c = budget - k_grid[kp_index] #consumption given k' selection
                if c>0 #check for positivity
                    val = log(c) + β*(Γ[z_index,1]*val_func[kp_index,1] + Γ[z_index,2]*val_func[kp_index,2]) #compute value
                    if val>candidate_max #check for new max value
                        candidate_max = val #update max value
                        res.pol_func[k_index, z_index] = k_grid[kp_index] #update policy function
                        choice_lower = kp_index #update lowest possible choice
                    end
                end
            end
            v_next[k_index,z_index] = candidate_max #update value function
        end
    end 
    v_next #return next guess of value function
end

#Value function iteration
function V_iterate(prim::Primitives, res::Results; tol::Float64 = 1e-4, err::Float64 = 100.0)
    n = 0 #counter

    while err>tol #begin iteration
        v_next = Bellman(prim, res) #spit out new vectors
        err = abs.(maximum(v_next.-res.val_func))/abs(v_next[prim.nk, 1]) #reset error level
        res.val_func = v_next #update value function
        n+=1
    end
    println("Stochastic value function converged in ", n, " iterations.")
end

#solve the model
function Solve_model(prim::Primitives, res::Results)
    V_iterate(prim, res) #in this case, all we have to do is the value function iteration!
end
##############################################################################
