#=
Author: Olivia 
Date: October 2024 
=#
#this file takes about 24 hours to run with 1500 grid points and the small errors. Reduce grid size to 500 and allow for error of 0.02 for more reasonable run time
using Parameters, Plots, LinearAlgebra, Optim, Interpolations #import the libraries we want
using Trapz


###### First part of the question ######

include("OLG_SSR.jl") #import the functions that solve our growth model
prim, res= Initialize(0.11, 0.42, [3.0,0.5]) #initialize primitive and results structs 
println("Solving the model with θ = 0.11, z=[3.0,0.5], γ=0.42")
@elapsed @time Bellman(prim, res, 0.11,[3.0,0.5], 0.42)
@unpack val_function, pol_func = res
#plot η over age 
Plots.plot(1:45, prim.η, title="Age efficiency profile", color="cadetblue", ylabel="η", xlabel="age", linewidth=:2.0, legend=false)
Plots.savefig("PS5/Graphs/q1_η.png")

#plot the value function at age 50 
Plots.plot(prim.A_grid, val_function[50, :, 1], title="Value Function at Model Age 50", color="cadetblue", linewidth=:2.0, ylabel="Value V(a)", xlabel="assets")
Plots.savefig("PS5/Graphs/q1_valfunc_50.png")

#plot the value function at age 20
Plots.plot(prim.A_grid, pol_func[20, :, 1], title="Savings Choice at Model Age 20", label="high z", color="cadetblue", linewidth=:2.0, ylabel="a'", xlabel="assets")
plot!(prim.A_grid, pol_func[20, :, 2], label="low z", color="palevioletred", linewidth=:2.0)
Plots.savefig("PS5/Graphs/q1_polfunc_20.png")

#plot the savings rate at age 20
pol_func_δ_1 = copy(pol_func[20,:,1]).-prim.A_grid
pol_func_δ_2 = copy(pol_func[20,:,2]).-prim.A_grid
Plots.plot(prim.A_grid, pol_func_δ_1, title="Savings Rate Changes at Model Age 20", label="high z", color="cadetblue", linewidth=:2.0, ylabel="a'-", xlabel="assets")
plot!(prim.A_grid, pol_func_δ_2, label="low z", color="palevioletred", linewidth=:2.0)
Plots.savefig("PS5/Graphs/q1_polfunc_δ_20.png")

#=
#### Other Parts ####
using DataFrames, CSV #import the libraries we want
include("OLG_SSR.jl") #import the functions that solve our growth model

prim, res= Initialize(0.11, 0.42, [3.0,0.5]) #initialize primitive and results structs
@elapsed @time benchmark= model_solve(prim, res; θ = 0.11, z=[3.0,0.5], γ=0.42,eps=0.01, λ=0.4, K=3.4, L=.34) 
table_a = create_table([benchmark])
CSV.write("PS5/table_a.csv", table_a) 

prim, res= Initialize(0.0, 0.42, [3.0,0.5]) #initialize primitive and results structs
@elapsed @time no_ss= model_solve(prim, res; θ = 0.0, z=[3.0,0.5], γ=0.42,eps=0.001, λ=0.25, K=4.4, L=.35) 
table_b = create_table([no_ss])
CSV.write("PS5/table_b.csv", table_b) 

prim, res= Initialize(0.11, 0.42, [0.5,0.5]) #initialize primitive and results structs
@elapsed @time benchmark_norisk= model_solve(prim, res; θ = 0.11, z=[0.5,0.5], γ=0.42,eps=0.001, λ=0.15, K=1.6, L=.16) 
table_c = create_table([benchmark_norisk])
CSV.write("PS5/table_c.csv", table_c) 

prim, res= Initialize(0.0, 0.42, [0.5,0.5]) #initialize primitive and results structs
@elapsed @time norisk_no_ss= model_solve(prim, res; θ = 0.0, z=[0.5,0.5], γ=0.42,eps=0.001, λ=0.1, K=1.5, L=.16) 
table_d = create_table([norisk_no_ss])
CSV.write("PS5/table_d.csv", table_d)

prim, res= Initialize(0.11, 1.0, [3.0,0.5]) #initialize primitive and results structs
@elapsed @time inelastic_l_ss= model_solve(prim, res; θ = 0.11, z=[3.0,0.5], γ=1.0,eps=0.001, λ=0.1, K=7.5,L=0.75) 
table_e = create_table([inelastic_l_ss])
CSV.write("PS5/table_e.csv", table_e) 

prim, res= Initialize(0.0, 1.0, [3.0,0.5]) #initialize primitive and results structs
@elapsed @time inelastic_l_no_ss= model_solve(prim, res; θ = 0.0, z=[3.0,0.5], γ=1.0,eps=0.001, λ=0.1, K=10.0, L=0.75) 
table_f = create_table([inelastic_l_no_ss])
CSV.write("PS5/table_f.csv", table_f) 

# Full table
table_1 = create_table([benchmark, no_ss,
                        benchmark_norisk, norisk_no_ss,
                        inelastic_l_ss, inelastic_l_no_ss])
CSV.write("PS5/table_1.csv", table_1 )
=#