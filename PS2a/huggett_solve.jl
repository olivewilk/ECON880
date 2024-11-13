#=
Author: Olivia 
Date: September 2024 
=#

using Parameters, Plots, LinearAlgebra, Optim, Interpolations #import the libraries we want
using Trapz
include("huggett_model.jl") #import the functions that solve our growth model

prim, res = Initialize() #initialize primitive and results structs
@elapsed @time Solve_model(prim, res) #solve the model!
@unpack val_func, pol_func, μ_dist = res
@unpack A_grid, nA = prim

#a) Plot the policy function 
##############Make plots
#value function
Plots.plot(A_grid, val_func[:,1], title="Value Function", label="employed", color="cadetblue", linewidth=:2.0, ylabel="Value V(a)", xlabel="assets")
plot!(A_grid, val_func[:,2], label="unemployed", color="palevioletred", linewidth=:2.0)
Plots.savefig("PS2/Graphs/Value_Functions.png")

#policy functions
# when does policy cross 45 degree line?
intersection = findmin(abs.(pol_func[:,1] .- A_grid))[1]
a_hat = zeros(nA)
for i in 1:nA
    if abs.(pol_func[i,1] .- A_grid[i])==intersection
        a_hat[i] = A_grid[i]
    end 
end 
a_hat_ans = findmax(a_hat)
println("Policy function crosses 45 degree line at $a_hat_ans")
a_hat_bar = minimum(a_hat_ans)

Plots.plot(A_grid, pol_func[:,1], title="Policy Functions", label="employed", color="cadetblue", linewidth=:2.0, ylabel="Policy a'(a)", xlabel="assets")
plot!(A_grid, pol_func[:,2], label="unemployed", color="palevioletred", linewidth=:2.0)
plot!(A_grid,A_grid,label = "45 degree",color="gray40",linestyle=:dash)
vline!([a_hat_bar], color="gray40", label="\$\\hat{a}\$")
Plots.savefig("PS2/Graphs/Policy_Functions.png")


#b) plot the cross sectional distribution of wealth for employed and unemployed agents
Plots.bar(A_grid, μ_dist[1:nA], title="Cross Sectional Distribution of Wealth", label="employed", color="cadetblue",linecolor="cadetblue", ylabel="Fraction of population", xlabel="assets")
bar!(A_grid, μ_dist[nA+1:end], label="unemployed", color="palevioletred", linecolor="palevioletred")
Plots.savefig("PS2/Graphs/Wealth_crosssectional_dist.png")

#Lorenz curve 
#fraction of wealth = y and fraction of population = x 

total_fraction = zeros(nA)
for i in 1:nA
    local x = μ_dist[i]
    local y = μ_dist[i+nA]
    total_fraction[i] = x+y
end 
population_lorenz = cumsum(total_fraction)

earnings_e = A_grid.+prim.y_grid[1]
earnings_u = A_grid.+prim.y_grid[2]
earnings = zeros(nA)
for i in 1:nA
    local x = μ_dist[i]*earnings_e[i]
    local y = μ_dist[i+nA]*earnings_u[i]
    earnings[i]= x+y
end 
total_wealth = sum(earnings)
wealth_lorenz = cumsum(earnings)/total_wealth

Plots.plot(population_lorenz, wealth_lorenz, title="Lorenz Curve", label="Lorenz Curve", color="cadetblue", linewidth=:2.0, ylabel="Fraction of wealth", xlabel="Fraction of population")
plot!(population_lorenz, population_lorenz, label="45 degree", color="gray40", linewidth=:2.0, linestyle=:dash)
Plots.savefig("PS2/Graphs/Lorenz.png")

#Calculate Gini coefficient
area = trapz(population_lorenz,wealth_lorenz)
@show gini = 1 - 2 * area
println("Gini coefficient in US 2021 is .398 (FRED). Our model has a Gini coefficient of $gini")

## PART III Welfare Anaylsis
w_fb = (((.9715)^(1-prim.α)-1)/(1-prim.α))/(1-prim.β)
αβ_frac = 1/((1-prim.α)*(1-prim.β))
num= w_fb + αβ_frac
denom = val_func .+ αβ_frac
lambda = (num./denom).^(1/(1-prim.α)) .-1

#a) plot lamba 
Plots.plot(A_grid, lambda[:,1], title="Consumption Equivalent", label="λ(a,e) (employed)", color="cadetblue", linewidth=:2.0, ylabel="λ(a,s)", xlabel="assets")
plot!(A_grid, lambda[:,2], label="λ(a,u) (unemployed)", color="palevioletred", linewidth=:2.0)
vline!([a_hat_bar], color="gray40", label="\$\\hat{a}\$")
Plots.savefig("PS2/Graphs/Consumption_Equivalent.png")

#b) calculate the welfare first best and welfare incomplete markets and welfare gain 
μ = reshape(μ_dist, (prim.nA, prim.ny))
w_inc = sum(μ.*val_func)
w_gain = sum(μ.*lambda)
println("Welfare first best is $w_fb")
println("Welfare incomplete markets is $w_inc")
println("Welfare gain is $w_gain")

vote = zeros(prim.nA, prim.ny)
for i in 1:prim.nA
    for j in 1:prim.ny
        if lambda[i,j] > 0
            vote[i,j] = 1
        else
            vote[i,j] = 0
        end
    end
end 

#c) what fraction would vote for complete markets?
fraction_change = sum(μ.*vote)
println("Fraction of agents that would pay for complete markets is $fraction_change")


println("All done!")
################################