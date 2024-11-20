using CSV, DataFrames, Parameters, Optim, Random, Distributions

# import dimension 1 KPU 
KPU1 = DataFrame(CSV.File("PS2b/inputs/KPU_d1_l20.csv")) # Load the data
KPU1 = Matrix(KPU1)

# import dimension 2 KPU
KPU2 = DataFrame(CSV.File("PS2b/inputs/KPU_d2_l20.csv")) # Load the data
KPU2 = Matrix(KPU2)


# integration functions 
function integrate1(expression, lower, KPU1)
    points = -log.(1 .- KPU1[:,1]) .+ lower
    jacobian = 1 ./ (1 .- KPU1[:,1])

    ans = sum(KPU1[:,2] .* expression.(points) .* jacobian)
    return ans
end 

function integrate2(expression, lower1, lower2, KPU2)
    points1 = -log.(1 .- KPU2[:,1]) .+ lower1
    jacobian1 = 1 ./ (1 .- KPU2[:,1]) 

    points2 = -log.(1 .- KPU2[:,2]) .+ lower2
    jacobian2 = 1 ./ (1 .- KPU2[:,2])
    
    ans = sum(KPU2[:,3] .* expression.(points1, points2) .* jacobian1 .* jacobian2)
    return ans 
end 

# import the mortgage data 
df = DataFrame(CSV.File("PS2b/inputs/Mortgage_performance_data.csv")) # Load the data

x_data = Matrix(df[:,1:15])
z_data = Matrix(df[:,15:17])
t_data = Vector(df[:,18])

# parameters t=0,1,2
K = length(x_data[1,:]) #number of x variables 
N = length(x_data[:,1]) #number of observations
a1 = 0.0
a2 = -1.0
a3 = -1.0
beta = zeros(K)
gamma = 0.3 
rho = 0.5

#simulate epsilon 
function get_shocks(N, rho)
    Random.seed!(1234) # Setting the seed
    d1 = Normal(0,1)
    eta = rand(d1,100,3)
    epsilon = zeros(100,3)
    sigma1sqrd = 1/((1-rho)^2)
    d2 = Normal(0,sigma1sqrd)
    
    epsilon[:,1] = rand(d2,100)
    epsilon[:,2] = rho*epsilon[:,1]  + eta[:,2]
    epsilon[:,3] = rho*epsilon[:,2]  + eta[:,3]

    return epsilon, eta, sigma1sqrd 
end 

# Phi and phi functions

function Phi(x)
    ans = cdf(Normal(0,1), x)
    return ans
end 

function phi(x)
    ans = pdf(Normal(0,1), x)
    #ans = 1/sqrt(2 * π) * exp((-1/2)*x^2)
    return ans
end 



## log likelihood function using quadrature method 
theta = vcat(a1, a2, a3, beta, gamma, rho)

#function llquad(a1, a2, a3, beta, gamma, rho, x_data, z_data, t_data, KPU1, KPU2, N, K)
function llquad(theta, x_data, z_data, t_data, KPU1, KPU2, N, K)

    #update parameters 
    a1 = theta[1]
    a2 = theta[2]
    a3 = theta[3]
    beta = theta[4:(K+3)]
    gamma  = theta[K+4]
    rho  = theta[K+5]

    eps ,eta, sigma1sqrd = get_shocks(N, rho)
    Y = zeros(N) 
    sigma1 = sqrt(sigma1sqrd)

    #update Y
    for i in 1:N
        if t_data[i] == 1
            Y[i] =  Phi((-a1 - beta'*x_data[i,:] - gamma*z_data[i,1])/sigma1)
        elseif t_data[i] == 2
            expression1(eps1) = (Phi(-a2 - beta'*x_data[i,:] - gamma*z_data[i,2] - rho*eps1)) * (phi(eps1/sigma1)/sigma1)
            lower = -a1 - beta'*x_data[i,:] - gamma*z_data[i,1]
            Y[i] = integrate1(expression1, lower, KPU1) 
        elseif t_data[i] == 3
            expression2(eps1, eps2) = Phi(-a3 - beta'*x_data[i,:] - gamma*z_data[i,3] - rho*eps2) * phi(eps2 - rho*eps1) * (phi(eps1/sigma1)/sigma1)
            lower1 = -a1 - beta'*x_data[i,:] - gamma*z_data[i,1]
            lower2 = -a2 - beta'*x_data[i,:] - gamma*z_data[i,2]
            Y[i] = integrate2(expression2, lower1, lower2, KPU2)
        elseif t_data[i] == 4
            expression3(eps1, eps2) = (1- Phi(-a3 - beta'*x_data[i,:] - gamma*z_data[i,3] - rho*eps2)) * phi(eps2 - rho*eps1) * (phi(eps1/sigma1)/sigma1)
            lower1 = -a1 - beta'*x_data[i,:] - gamma*z_data[i,1]
            lower2 = -a2 - beta'*x_data[i,:] - gamma*z_data[i,2]
            Y[i] = integrate2(expression3, lower1, lower2, KPU2)
        end 
    end 

    #update log likelihood
    ll = sum(log.(Y))

    return ll
end 

#test = llquad(a1, a2, a3, beta, gamma, rho, x_data, z_data, t_data, KPU1, KPU2, N, K)
test = llquad(theta, x_data, z_data, t_data, KPU1, KPU2, N, K)
println("test: ", test)

# accept reject method 

function acceptreject(a1, a2, a3, beta, gamma, rho, x_data, z_data, t_data, KPU1, KPU2, N, K)
    epsilon ,eta, sigma1sqrd = get_shocks(N, rho)
    count = zeros(N)

    for i in 1:N 
        if t_data[i] == 1
            count[i] = sum(a1 + beta'*x_data[i,:] + gamma*z_data[i,1] .+ epsilon[:,1] .< 0)
        elseif t_data[i] == 2
            count[i] = sum((a1 + beta'*x_data[i,:] + gamma*z_data[i,1] .+ epsilon[:,1] .> 0).*(a2 + beta'*x_data[i,:] + gamma*z_data[i,2] .+ epsilon[:,2] .< 0))
        elseif t_data[i] == 3
            count[i] = sum((a1 + beta'*x_data[i,:] + gamma*z_data[i,1] .+ epsilon[:,1] .> 0).*(a2 + beta'*x_data[i,:] + gamma*z_data[i,2] .+ epsilon[:,2] .> 0).*(a3 + beta'*x_data[i,:] + gamma*z_data[i,3] .+ epsilon[:,3] .< 0))
        elseif t_data[i] == 4
            count[i] = sum((a1 + beta'*x_data[i,:] + gamma*z_data[i,1] .+ epsilon[:,1] .> 0).*(a2 + beta'*x_data[i,:] + gamma*z_data[i,2] .+ epsilon[:,2] .> 0).*(a3 + beta'*x_data[i,:] + gamma*z_data[i,3] .+ epsilon[:,3] .> 0))
        end
    end 

    ll = sum(log.(count ./ 100))
    return ll
end 


test2 = acceptreject(a1, a2, a3, beta, gamma, rho, x_data, z_data, t_data, KPU1, KPU2, N, K)

a1 = 5.0  
a2 = 2.5
a3 = 2.5
beta = zeros(K)
gamma = 0.0 
rho = 0.5 
q4_res = optimize(θ -> -llquad(θ, x_data, z_data, t_data, KPU1, KPU2, N, K), 
                                vcat(a1, a2, a3, beta, gamma, rho),
                                BFGS())
                                
println(q4_res.minimizer)