## Construct Big Transition Matrix (Pi in Dean's Slides)
#= The reason we do this first is because Pi does not depend at all on the
distribution. So we can construct this first and then repeatedly use it
when we iterate over the distribution. =#


#=intuitively, what is trans_mat?
trans_mat is a big transition matrix that tells you if you are in a
certain state today (some asset level and employment status), where are
you transitioning to tomorrow. The (row col) combo tells you where you are
today (row) and where you are going tomorrow (col) with some probability 
(the entry in trans_mat). that probability is determined by the markov 
transition matrix. =#

function TStar(prim::Primitives,res::Results, dist_mu::Array{Float64,1})

    @unpack y_grid, A_grid, Π, nA, ny = prim
    @unpack pol_func = res
    # Allocate space 
    trans_mat = zeros(nA*ny,nA*ny)

    for y_index = 1:ny
        y_today = y_grid[y_index]
        for a_index = 1:nA
            a_today = A_grid[a_index]
            
            #trans_mat is dimension (n_a*n_s x n_a*n_s). We need a way to map
            # counting up from from i_a = 1:n_a and i_s = 1:n_s into a single
            # number that runs from 1:n_a*n_s. 
            row = a_index + nA*(y_index - 1) # defines row position in trans_mat. Do you see how your state today is encoded in row?
            
            for yp_index = 1:ny
                for ap_index = 1:nA
                    a_tomorrow = pol_func[ap_index, a_index, y_index] # read off asset choice based on policy function
                    if a_tomorrow == A_grid[ap_index] # This is one way to construct the indicator function Dean references in his slides.
                        col = ap_index + nA*(yp_index - 1) # defines col position in trans_mat. Do you see how your state tomorrow is encoded in col?
                        trans_mat[row, col] = Π[y_index, yp_index] # fill in the transition probability
                    end
                end
            end
        end
    end

    ## Solve For Stationary Distribution
    # Apply T* operator. That is, iterate on the cross-sectional distribution until convergence.  
    # start iteration 
    it = 1
    converged = 0
    tol = 1e-5
    maxit = 1000

    while (converged == 0 & it < maxit)
        dist_mu_up = trans_mat'*dist_mu;
        
        # Calculate the Supnorm
        max_diff = sum(abs(dist_mu_up - dist_mu))
        if max_diff < tol
            converged = 1
            dist_mu_out = dist_mu_up
        end
            
        it=it+1
        println("Iteration #", it, "max_diff:", max_diff)
        
        # update cross sectional distribution
        dist_mu = dist_mu_up
    end

    return dist_mu 
end 



μ_dist = (A_grid.-A_min)./(A_max-A_min) 

using Parameters, Plots #import the libraries we want
include("huggett_model.jl") #import the functions that solve our growth model
prim, res = Initialize() #initialize primitive and results structs
function TStar(prim::Primitives,res::Results, μ_dist::Array{Float64,1})