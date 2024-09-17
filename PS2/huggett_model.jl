
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
    nA::Int64 = 1001 #number of asset grid points
    A_grid::Array{Float64,1} = collect(range(A_min, length = nA, stop = A_max))
    Π::Array{Float64,2} = [0.97 0.03; 0.5 0.5] #markov process for earnings 
    q_initial=0.994 #initial bond price
    #μ_dist_initial= vcat(((A_grid.-A_min)./(A_max-A_min)), ((A_grid.-A_min)./(A_max-A_min)))
end

#structure that holds model results
mutable struct Results
    val_func::Array{Float64, 2} #value function
    pol_func::Array{Float64, 2} #policy function (choice of a')
    μ_dist::Array{Float64, 1} #cross sectional distribution
end

#function for initializing model primitives and results
function Initialize()
    prim = Primitives() #initialize primtiives
    val_func = zeros(prim.nA, prim.ny) #initial value function guess
    pol_func = zeros(prim.nA, prim.ny ) #initial policy function guess 
    μ_dist= vcat(((prim.A_grid.- prim.A_min)./(prim.A_max-prim.A_min)), ((prim.A_grid.- prim.A_min)./(prim.A_max-prim.A_min)))
    res = Results(val_func, pol_func, μ_dist) #initialize results struct
    prim, res #return deliverables
end

#continuation value function 
function cont_val(ap, y, a, q, v_e, v_u, p, prim)
    """ calculates V(a, s ; q) = u(c) + β*E[V(a',s';q)] """

    @unpack β, α = prim
    c = y + a - q*ap
    if c <= 0
        u = -Inf
    else 
        u = ( c^(1 - α) - 1 )/ (1 - α)
    end 
    Ev = p[1]*v_e(ap) + p[2]*v_u(ap)
    v = -1*(u + β*Ev)
    return v
end 

#Bellman Operator
function Bellman(prim::Primitives,res::Results, q::Float64)
    @unpack val_func = res #unpack value function
    @unpack y_grid, A_grid, A_min, A_max, β, α, nA, ny, Π, q_initial = prim #unpack model primitives
    v_next = zeros(nA, ny) #next guess of value function to fill

    # Interpolate the value function 
    #v_e = linear_interpolation(A_grid, res.val_func[:,1])
    v_e = LinearInterpolation(A_grid, res.val_func[:,1])
    #v_u = linear_interpolation(A_grid, res.val_func[:,2])
    v_u = LinearInterpolation(A_grid, res.val_func[:,2])

    for y_index = 1:ny
        y = y_grid[y_index]
        p = Π[y_index, :]
        for a_index = 1:nA
            a = A_grid[a_index]
            a_hat = min((y + a)/q, A_max) # c ≥ 0 constraint

            optim_results = optimize(ap -> cont_val(ap, y, a, q, v_e, v_u, p, prim), A_min, a_hat)
            a_star = optim_results.minimizer
            v_star = -1*optim_results.minimum

            v_next[a_index, y_index] = v_star
            res.pol_func[a_index, y_index] = a_star
        end
    end
    return v_next
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

function Tstar(prim::Primitives,res::Results)
    @unpack y_grid, A_grid, A_min, A_max, β, α, nA, ny, Π, q_initial = prim #unpack model primitives
    @unpack val_func, μ_dist = res #unpack value
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
    tol = 1e-10
    max_iter= 1000
    for iter in 1:max_iter
        μ_next = trans_mat' * μ_dist
        if norm(μ_next - μ_dist) < tol
            return μ_next / sum(μ_next)  # normalize distribution
        end
        μ_dist = μ_next
    end
    return μ_dist
end 

#Price Solve 
function Q_Solve(prim::Primitives, res::Results)
    @unpack val_func, μ_dist = res #unpack value function
    @unpack y_grid, A_grid, A_min, A_max, β, α, nA, ny, Π, q_initial = prim #unpack model primitives
    q = q_initial #initial bond price guess
    #μ_dist = μ_dist_initial #initial distribution guess
    #μ_dist = ones(nA*ny)
    mk=0 #market not cleared initialization 
    iterations= 0 
    while mk == 0 #market not cleared 
        println("Bond price:",q)
        iterations+=1

        V_iterate(prim, res, q) #iterate on value function
        
        res.μ_dist=Tstar(prim, res) #iterate on cross sectional distribution
        #μ = reshape(res.μ_dist, (prim.nA, prim.ny))
        #println("mu: ",μ_dist)
        integral = sum(res.pol_func[:, 1]'*res.μ_dist[1:nA] + res.pol_func[:, 2]'*res.μ_dist[nA+1:end])
        #integral = sum(μ.*res.pol_func)
        println("summation:",integral)

        if abs(integral)<=.001 #check for market clearing
            mk = 1 #market cleared
        elseif integral<-0.001 #adjust bond price
            q = q-0.01*(1-q)/2
        elseif integral>0.001 #adjust bond price
            q = q+0.01*(1-q)/2
        end 
        println("Bond price:",q)
        if iterations>1000 #check for convergence
            println("Bond price did not converge")
            break
        end
    end 
    mk #return market clearing indicator
    println("Bond price converged in ", iterations, " iterations.")
    return res.μ_dist
end 


#solve the model
function Solve_model(prim::Primitives, res::Results)
    Q_Solve(prim, res) 
    #V_iterate(prim, res, q==.95)
end

