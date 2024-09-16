
@with_kw struct Primitives
    β::Float64 = 0.9932 #discount rate
    α::Float64 = 1.5 #coefficient of relative risk aversion
    e::Float64 = 1.0 #normalized earning when employed 
    u::Float64 = 0.5 #earnings when unemployed
    ny::Int64 = 2 #number of earnings states
    y_grid::Array{Float64,1} = [1.0, 0.5] #vector of earnings
    q_min::Float64 = 0.0 #minimum bond price
    q_max::Float64 = 1.0 #maximum bond price  
    A_min::Float64 = -2.0 #minimum asset
    A_max::Float64 = 5.0 #maximum asset
    nA::Int64 = 500 #number of asset grid points
    A_grid::Array{Float64,1} = collect(range(A_min, length = nA, stop = A_max))
    Π::Array{Float64,2} = [0.97 0.03; 0.5 0.5] #markov process for earnings 
    q_initial=1.392 #initial bond price
    μ_dist_initial= vcat(((A_grid.-A_min)./(A_max-A_min)), ((A_grid.-A_min)./(A_max-A_min)))
end

#structure that holds model results
mutable struct Results
    val_func::Array{Float64, 2} #value function
    pol_func::Array{Float64, 2} #policy function (choice of a')
end

#function for initializing model primitives and results
function Initialize()
    prim = Primitives() #initialize primtiives
    val_func = zeros(prim.nA, prim.ny) #initial value function guess
    pol_func = zeros(prim.nA, prim.ny ) #initial policy function guess
    res = Results(val_func, pol_func) #initialize results struct
    prim, res #return deliverables
end

#loop through current asset grid and future 
    
#Bellman Operator
function Bellman(prim::Primitives,res::Results, q::Float64)
    @unpack val_func = res #unpack value function
    @unpack y_grid, A_grid, A_min, A_max, β, α, nA, ny, Π, q_initial = prim #unpack model primitives
    v_next = zeros(nA, ny) #next guess of value function to fill
    for y_index = 1:ny 
        y = y_grid[y_index]
        for a_index = 1:nA
            a = A_grid[a_index] #value of assets
            candidate_max = -Inf #bad candidate max
            budget = y + a #budget
            for ap_index in 1:nA
                c = budget - q*A_grid[ap_index] #consumption given a' selection
                if c>0 #check for positivity
                    val = (c^(1-α)-1)/(1-α) + β*(Π[y_index,1]*val_func[ap_index,1] + Π[y_index,2]*val_func[ap_index,2]) #compute value
                    if val>candidate_max #check for new max value
                        candidate_max = val #update max value
                        res.pol_func[a_index, y_index] = A_grid[ap_index] #update policy function
                    end
                end
            end
            v_next[a_index, y_index] = candidate_max #update value function
        end
    end 
    v_next #return next guess of value function 
end

#Value function iteration
function V_iterate(prim::Primitives, res::Results, q::Float64; tol::Float64 = 1e-4, err::Float64 = 100.0)
    n = 0 #counter
    while err>tol #begin iteration
        v_next = Bellman(prim, res, q)
        err = maximum(abs.(v_next .- res.val_func))
        #err = abs.(maximum(v_next.-res.val_func))/abs(v_next[prim.nA, 1]) #reset error level
        res.val_func .= v_next
        n += 1
        #println("Iteration #", n, "error:", err)
    end
    println("Value function converged in ", n, " iterations.")
end

function Tstar(prim::Primitives,res::Results, μ_dist::Array{Float64,1})
    @unpack y_grid, A_grid, A_min, A_max, β, α, nA, ny, Π, q_initial = prim #unpack model primitives
    @unpack val_func = res #unpack value
    trans_mat = zeros(nA*ny,nA*ny) #initialize transition matrix
    for y_index in 1:ny
        for a_index in 1:nA
            ap = res.pol_func[a_index, y_index]  # get next period's asset from policy function
            ap_id = argmin(abs.(ap .- A_grid))

            for yp_index in 1:ny
                trans_mat[a_index + nA*(y_index -1), ap_id + nA*(yp_index - 1) ] =  Π[y_index, yp_index]
                #P[(a_i - 1) * ns + s_i, (a_prime_id - 1) * ns + s_prime] = Π[s_i, s_prime]
            end
        end
    end
    #println("Transition matrix: ", trans_mat)
    #μ_dist = TStar(prim, res, μ_dist) #compute stationary distribution
       ## Solve For Stationary Distribution
    # Apply T* operator. That is, iterate on the cross-sectional distribution until convergence.  
    # start iteration 
    #it = 1
    #converged = 0
    tol = 1e-5
    max_iter= 1000
    for iter in 1:max_iter
        μ_next = trans_mat' * μ_dist
        if norm(μ_next - μ_dist) < tol
            return μ_next / sum(μ_next)  # normalize distribution
        end
        μ_dist = μ_next
    end

    #=while (converged == 0 & it < maxit)
        μ_dist_up = trans_mat'*μ_dist 
        #μ_dist_up = μ_dist'*trans_mat
        
        # Calculate the Supnorm
        max_diff = sum(abs.(μ_dist_up - μ_dist))
        if max_diff < tol
            converged = 1
            μ_dist_out = μ_dist_up
        end
            
        it=it+1
       # println("Iteration #", it, "max_diff:", max_diff)
        
        # update cross sectional distribution
        μ_dist = μ_dist_up =#
    #end
    return μ_dist
end 

#Price Solve 
function Q_Solve(prim::Primitives, res::Results)
    @unpack val_func = res #unpack value function
    @unpack y_grid, A_grid, A_min, A_max, β, α, nA, ny, Π, q_initial, μ_dist_initial = prim #unpack model primitives
    q = q_initial #initial bond price guess
    μ_dist = μ_dist_initial #initial distribution guess
    #μ_dist = ones(nA*ny)
    mk=0 #market not cleared initialization 
    iterations= 0 
    while mk == 0 #market not cleared 
        println("Bond price:",q)
        iterations+=1

        V_iterate(prim, res, q) #iterate on value function
        
        Tstar(prim, res, μ_dist) #iterate on cross sectional distribution

       # println("mu: ",μ_dist)
        integral = sum(res.pol_func[:, 1]'*μ_dist[1:nA] + res.pol_func[:, 2]'*μ_dist[nA+1:end])
        #integral = sum(μ_dist[1:nA]*res.pol_func[:, 1] + μ_dist[nA+1:end]*res.pol_func[:, 2])
        println("summation:",integral)

        if abs(integral)<=.01 #check for market clearing
            mk = 1 #market cleared
        elseif integral<-0.01 #adjust bond price
            q = q-0.000001
        elseif integral>0.01 #adjust bond price
            q = q+0.000001
        end 
        println("Bond price:",q)
        if iterations>4000 #check for convergence
            println("Bond price did not converge")
            break
        end
    end 
    mk #return market clearing indicator
    println("Bond price converged in ", iterations, " iterations.")
end 


#solve the model
function Solve_model(prim::Primitives, res::Results)
    Q_Solve(prim, res) 
    #V_iterate(prim, res, q==.95)
end

