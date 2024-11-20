## import packages
using StatFiles, DataFrames, LinearAlgebra, Plots, Optim

# Read the Stata files
df_characteristics = DataFrame(load("PS3b/inputs/Car_demand_characteristics_spec1.dta"))
df_iv  = DataFrame(load("PS3b/inputs/Car_demand_iv_spec1.dta"))
df_distribution = DataFrame(load("PS3b/inputs/Simulated_type_distribution.dta"))

# Sort datasets by year and model_id
sort!(df_characteristics, [:Year, :Model_id])
sort!(df_iv, [:Year, :Model_id])

#parameters 
lambda_p0 = 0.6
eps = 1e-12 
eps1 = 1

# invert demand for lambda_p = 0.6 using contraction mapping and contraction mapping with Newton 
function invert_demand(year, lambda_p0, eps, eps1)
    df = filter(row -> row[:Year] == year, df_characteristics)

    delta_iia = df.delta_iia 
    share = df.share
    price = df.price
    Y = Matrix(df_distribution)
    J = length(share) #number of products in the market (year)
    R = length(Y) #number of incomes 
    Id = I(J) #identity matrix

    err = 100 #starting error 
    errnorm = [] #store the norm across iterations
    iter = 1 
    maxiter = 1000 

    # compute idiosyncratic utility (function of consumers and products - no delta) 
    mu = lambda_p0 .* (price * Y')

    # initialize deltas 
    delta0 = copy(delta_iia)
    delta1 = copy(delta_iia)

    while err > eps && iter < maxiter
        # compute choice probability 
        #Lambda = exp.(delta0 .+ mu) 
        sigma_rj = (exp.(delta0 .+ mu)  ./ (1 .+ sum(exp.(delta0 .+ mu), dims = 1)))
        sigma_j = (1/R).*sum((exp.(delta0 .+ mu)  ./ (1 .+ sum(exp.(delta0 .+ mu), dims = 1))), dims =2)

        #update guess: when err1=err then only contraction mapping is used 
        if err > eps1  # if error is larger than 1 use contraction mapping
            delta1 = delta0 + log.(share) - log.(sigma_j)
        else # when error is small enough switch to newton 
            Delta = ((1 / R) .* Id .* (sigma_rj * (1 .- sigma_rj)')) - ((1 / R) .* (1 .- Id) .* (sigma_rj * sigma_rj'))
            delta1 = delta0 + (inv(Delta ./ sigma_j) * (log.(share) - log.(sigma_j)))
        end 

        # compute error norm 
        err = norm(delta1 - delta0)
        push!(errnorm, err) #store the error in the norm vector
 
        delta0 = copy(delta1)

        iter += 1
    end 
    return errnorm, delta0 
end 

# 1) plot the evolution of the norm between log predicted and observed shares across iterations 
q1resa = invert_demand(1985, lambda_p0, eps, eps1) #contraction mapping and Newton
q1resb = invert_demand(1985, lambda_p0, eps, eps) #only contraction mapping 
cn = q1resa[1]
cm = q1resb[1]
plot(cn[2:end], label="Contraction mapping and Newton", title="Convergence of the norm", color="cadetblue",xlabel="Iteration", ylabel="norm", lw=2)
plot!(cm[2:end], label="Contraction mapping only", color="palevioletred", lw=2)
Plots.savefig("PS3b/output/q1_evolution_of_norm.png")

# 2) Grid search over lambda_p in [0,1]
X = Matrix(df_characteristics[:, 6:end])
Z = hcat(Matrix(df_characteristics[:, 7:end]),Matrix(df_iv[:, 3:end]))

function beta_iv(X,Z,W,delta_grid)
    beta = inv((X'*Z)*W*(Z'*X))*(X'*Z)*W*Z'*delta_grid
    return beta 
end 
function rho(delta_grid, X, beta_iv)
    rho = delta_grid - X*beta_iv
end 

#function gmm_obj(lambda_p, eps, eps1, W, X, Z)
function GMM(lambda_p::Float64, W)

    delta_grid = []
    for y in 1985:2015
        delta_grid = vcat(delta_grid,invert_demand(y, lambda_p, eps, eps1)[2])
    end 
    
    beta = beta_iv(X,Z,W,delta_grid)

    r = rho(delta_grid, X, beta)
    
    ans = r'*Z*W*Z'*r
    ans = ans[1,1]
    return ans
end 

# grid search 
W1 = inv(Z'*Z)
lambda_grid = collect(range(0, length=11, stop=1))
q2res = zeros(length(lambda_grid))
for i in 1:length(lambda_grid)
    #answer = gmm_obj(lambda_grid[i], eps, eps1, W1, X, Z)[1]
    answer = GMM(lambda_grid[i], W1)
    q2res[i] = answer
end
# plot the GMM objective function for lambda 
plot(lambda_grid, q2res, label="GMM objective function", title="GMM objective function over lambda", color="cadetblue",xlabel="lambda_p", ylabel="GMM objective function", lw=2)
Plots.savefig("PS3b/output/q2_gmm_first.png")

# 3) Using the minimum from the grid search, estimate the parameter λp using 2-step GMM. 
lambda_guess = [0.6]
f(l) = GMM(l, W1)
gmm = optimize(f, lambda_guess, BFGS())
lambda_hat = Optim.minimizer(gmm)

# find rho(λ_hat)
delta_grid_hat = []
for y in 1985:2015
    delta_grid_hat = vcat(delta_grid_hat,invert_demand(y, lambda_hat, eps, eps1)[2])
end    
beta_hat = beta_iv(X,Z,W1,delta_grid_hat)
rho_hat = rho(delta_grid_hat, X, beta_hat)

W2 = inv((Z.*rho_hat)'*(Z.*rho_hat)) #weight matrix for 2-step 

# 2-step GMM

q3res = zeros(length(lambda_grid))
for i in 1:length(lambda_grid)
    #answer = gmm_obj(lambda_grid[i], eps, eps1, W1, X, Z)[1]
    answer = GMM(lambda_grid[i], W2)
    q3res[i] = answer
end

g(l) = GMM(l, W2)
gmm2 = optimize(g, lambda_hat, BFGS())
lambda_hat2 = Optim.minimizer(gmm2)

label1 = round(lambda_hat[1], digits=3)
label2 = round(lambda_hat2[1], digits=3)
plot(lambda_grid, q2res, label="1st stage", title="GMM objective function over lambda", color="cadetblue",xlabel="lambda_p", ylabel="GMM objective function", lw=2)
plot!(lambda_grid, q3res, label="2nd stage", color="palevioletred", lw=2)
scatter!([lambda_hat2], [g(lambda_hat2)], label="lambda = $label2", color="palevioletred", markersize=5)
scatter!([lambda_hat], [f(lambda_hat)], label="lambda = $label1", color="cadetblue", markersize=5)
Plots.savefig("PS3b/output/q3_gmm_second.png")

