using CSV, DataFrames, Parameters, Optim 

beta0 = -1

df = DataFrame(CSV.File("PS1b/Mortgage_performance_data.csv")) # Load the data
datamatrix = Matrix(df)

y =datamatrix[:,1] # i close first year 
N = length(y)
X = hcat(ones(N),datamatrix[:,2:17] ) 
K = length(X[1,:])
beta = zeros(K)
beta[1] = -1

# Question 1: log-likehood function, score of log-likehood function, and Hessian. Evaluate at beta0 = -1 and beta = 0 
function log_likelihood(beta, X, N)
    Lambda = zeros(N)
    log_likelihood = 0
    for i in 1:N
        Lambda[i] = exp(vec(beta)'*X[i,:])./(1 .+ exp(vec(beta)'*X[i,:]))
        log_likelihood += log((Lambda[i]^y[i])* ((1 - Lambda[i])^(1 - y[i])))
    end 
    
    return log_likelihood
end 


function score_log_likelihood(beta, y, X, N, K)
    score = zeros(N,K)
    for i in 1:N
        score[i,:] = (y[i] - exp(vec(beta)'*X[i,:])./(1 .+ exp(vec(beta)'*X[i,:]))).*X[i,:]
    end 
    g = sum(score, dims = 1)
    return g
end 


function Hessian(beta, y, X, N, K)
    hes = zeros(N,K,K)
    for i in 1:N
        hes[i,:,:] = exp(vec(beta)'*X[i,:])./(1 .+ exp(vec(beta)'*X[i,:])) * (1 - exp(vec(beta)'*X[i,:])./(1 .+ exp(vec(beta)'*X[i,:]))) * X[i,:] * X[i,:]'
    end
    H = -sum(hes, dims = 1)
    return H
end 

ll = log_likelihood(beta, X, N)
score = score_log_likelihood(beta, y, X, N, K)
H = Hessian(beta, y, X, N, K)[1,:,:]

# Numerical first and second derivative of the log-likelihood function

function numerical_first(beta, X, N, K) 
    #println("beta: ", beta)
    h = 1e-10
    g = zeros(K)
    for i in 1:K 
        beta_plus = copy(beta)
        beta_plus[i] = beta_plus[i] + h
        ll_plus = log_likelihood(beta_plus, X, N)
        #println("ll_plus: ", ll_plus)

        beta_minus = copy(beta)
        beta_minus[i] = beta_minus[i] - h
        ll_minus = log_likelihood(beta_minus, X, N)
        #println("ll_minus: ", ll_minus)
        g[i] = (ll_plus - ll_minus) / (2 * h) #central difference
    end 
    return g
end 

function numerical_second(beta, X, N, K)
    h = 1e-10
    H = zeros(K,K)
    for i in 1:K
        beta_plus = copy(beta)
        beta_plus[i] += h
        g_plus = numerical_first(beta_plus, X, N, K)

        beta_minus = copy(beta)
        beta_minus[i] -= h
        g_minus = numerical_first(beta_minus, X, N, K)

        H[i,:] = (g_plus .- g_minus) ./ (2 * h) #central difference
    end 
    return H 
end 

g_num = numerical_first(beta, X, N, K)
H_num = numerical_second(beta, X, N, K)


#=println("The analytical first derivative of the log-likelihood function is: ")
@show score
println("The numerical first derivative of the log-likelihood function is: ")
@show g_num
println("The analytical second derivative of the log-likelihood function is: ")
@show H
println("The numerical second derivative of the log-likelihood function is: ")
@show H_num
=#
gscore = DataFrame(g=vec(score), gn=vec(g_num))

# export to CSV 
CSV.write("PS1b/PS1b_output_score.csv", gscore)
CSV.write("PS1b/PS1b_output_hess1.csv", DataFrame(H, :auto), writeheader = false)
CSV.write("PS1b/PS1b_output_hess2.csv", DataFrame(H_num, :auto), writeheader = false)


## Question 3: solve the maximum likelihood problem using Newton algorithm 

function newton_solve(score_log_likelihood, Hessian, log_likelihood, beta_guess, X, y, N, K; tol = 1e-10, max_iter = 1000)
    error = 100 
    iter = 1 
    lambda = 1
    while error > tol && iter < max_iter
        g = score_log_likelihood(beta_guess, y, X, N, K)
        #println("g: ", g)
        H = Hessian(beta_guess, y, X, N, K)[1,:,:]        
        #println("H: ", H)

        next_guess = beta_guess .- inv(H) * g'

        error = maximum(abs.(next_guess .- beta_guess))

        beta_guess = lambda*next_guess + (1-lambda)*beta_guess
        iter += 1
        #print("Iteration: ", iter, " Error: ", error)
    end 
    betanew = vec(beta_guess)
    ll = log_likelihood(betanew, X, N)
    return beta_guess, ll
end 

newton_res = newton_solve(score_log_likelihood, Hessian, log_likelihood, beta, X, y, N, K) #sensitive to lambda 
betanewton = vec(newton_res[1])

## Question 4: Compare with BFGS and Simplex packages 
bfgs_res = optimize(b -> -log_likelihood(b, X, N), beta, BFGS()) # Broyden–Fletcher–Goldfarb–Shanno algorithm

simplex_res = optimize(b -> -log_likelihood(b, X, N), betanewton, NelderMead()) # Broyden–Fletcher–Goldfarb–Shanno algorithm

betas = DataFrame(newton = betanewton, bfgs=bfgs_res.minimizer, simplex = simplex_res.minimizer)

CSV.write("PS1b/PS1b_output_betas.csv", betas)