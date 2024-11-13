#=
Author: Olivia 
Date: October 2024 
=#
#this file takes about 24 hours to run with 1500 grid points and the small errors. Reduce grid size to 500 and allow for error of 0.02 for more reasonable run time
using Parameters, Plots, LinearAlgebra, Optim, Interpolations #import the libraries we want
using Trapz


###### First part of the question ######
using DataFrames, CSV #import the libraries we want
include("OLG_SSR_transition.jl") #import the functions that solve our growth model

# social security
N_t = 20
prim, res= Initialize(0.11, 0.42, [3.0,0.5],N_t) #initialize primitive and results structs

@elapsed @time benchmark= model_solve(prim, res; θ = 0.11, z=[3.0,0.5], γ=0.42,eps=0.01, λ=0.4, K=3.4, L=.34) 
table_a = create_table([benchmark])
CSV.write("PS6/Tables/table_a.csv", table_a) 
V_0_SS = benchmark.val_function
g_0_SS = benchmark.pol_func
Gamma_0_SS = benchmark.F
K_0_SS = benchmark.K
L_0_SS = benchmark.L
# no social security 
prim, res= Initialize(0.0, 0.42, [3.0,0.5],N_t) #initialize primitive and results structs
@elapsed @time no_ss= model_solve(prim, res; θ = 0.0, z=[3.0,0.5], γ=0.42,eps=0.01, λ=0.15, K=4.75, L=0.3788) 
table_b = create_table([no_ss])
CSV.write("PS6/Tables/table_b.csv", table_b) 
V_N_SS = no_ss.val_function
g_N_SS = no_ss.pol_func
Gamma_N_SS = no_ss.F
K_N_SS = no_ss.K
L_N_SS = no_ss.L

# Full table
table_1 = create_table([benchmark, no_ss])
CSV.write("PS6/Tables/table_1.csv", table_1 )


###### Compute transition ###### 
include("OLG_SSR_transition.jl") #import the functions that solve our growth model
N_t = 20
θ_grid= collect(range(0.11, length = N_t, stop = 0.0)) #grid for θ
prim, res= Initialize(0.0, 0.42, [3.0,0.5],N_t) #initialize primitive and results structs
#K_t = collect(range(K_0_SS, length = N_t, stop = K_N_SS)) #grid for K
K_t = log.(collect(range(exp(K_0_SS), length = N_t, stop = exp(K_N_SS)))) #grid for K
#K_t = 
L_t = collect(range(L_0_SS, length = N_t, stop = L_0_SS)) #grid for L
CSV.write("PS6/Tables/transition_inputs.csv", DataFrame(theta_grid=θ_grid, K_t=K_t, L_t=L_t))
#=
@elapsed @time transitionresults=bellman_transition(prim, res; V_N_SS, g_N_SS, N_t, K_t, L_t, θ_grid)
@elapsed @time K_t_new = update_capital_path(res,Gamma_0_SS, K_0_SS)
@elapsed @time L_t_new = update_labor_path(prim, res,Gamma_0_SS, L_0_SS)
=#
#price_solve_transition(prim, res, K_t, L_t, θ_grid, N_t)
#bellman_transition(prim, res; V_N_SS, g_N_SS, N_t, K_t, L_t, θ_grid)
@elapsed @time K_final, L_final=transition_solve(prim, Gamma_0_SS, K_0_SS,L_0_SS,L_N_SS, V_N_SS, N_t, K_t,L_t, θ_grid; rho=0.5, eps=0.05)


# Plot the transition paths
#capital
Plots.plot(1:N_t, K_final[1:N_t], title="Capital Path", color="cadetblue", label="capital path", xlabel="time", linewidth=:2.0)
#plot!(1:N_t, K_t, color="palevioletred", linewidth=:2.0, label="guessed path")
plot!(1:N_t, K_0_SS*ones(N_t), color="blue", linewidth=:2.0, label="initial capital")
plot!(1:N_t, K_N_SS*ones(N_t), color="black", linewidth=:2.0, label="final capital")
Plots.savefig("PS6/Graphs/capital_path.png")
#labor 
Plots.plot(1:N_t, L_final, title="Labor Path", color="cadetblue", label="labor path", xlabel="time", linewidth=:2.0)
plot!(1:N_t, L_t, color="palevioletred", linewidth=:2.0, label="guessed path")
plot!(1:N_t, L_0_SS*ones(N_t), color="blue", linewidth=:2.0, label="initial labor")
plot!(1:N_t, L_N_SS*ones(N_t), color="black", linewidth=:2.0, label="final labor")
Plots.savefig("PS6/Graphs/labor_path.png")
#wage 
#=
Plots.plot(1:N_t, res.V_transition[60,250,1,1:N_t],title="")
plot!(1:N_t, V_N_SS[60,250,1]*ones(N_t))
plot!(1:N_t, V_N_SS[60,250,2]*ones(N_t))

Plots.plot(1:N_t, res.g_0[5,4,1,1:N_t],title="")
plot!(1:N_t, res.g_0[65,2,1,1:N_t])
plot!(1:N_t, res.g_0[65,3,1,1:N_t])
plot!(1:N_t, res.g_0[35,50,1,1:N_t])
plot!(1:N_t, res.g_0[25,50,1,1:N_t])
plot!(1:N_t, res.g_0[15,50,1,1:N_t])
#plot!(1:N_t, res.g_0[55,1,1,1:N_t])

res.g_0[65,4,2,:]
g_0_SS[65,1,2]
g_N_SS[65,1,2]
Plots.plot(1:N_t, res.V_transition[45,50,1,1:N_t])
plot!(1:N_t, res.V_transition[55,50,1,1:N_t])
plot!(1:N_t, res.V_transition[65,50,1,1:N_t])



K_t
res.w_transition[1:N_t]
res.r_transition[1:N_t]
res.b_transition[1:N_t]

res.V_transition[2,:,:,29]
res.g_0[1,:,:,28]
V_N_SS[65,:,:]

g_N_SS[66,:,:]

utility_r(prim, 0.01, 0, 0.42)

utility_r_t(prim, a, ap, γ, it)=#