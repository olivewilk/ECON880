//delimit;
set more off
cd "C:\Users\olivi\Documents\Wisconsin\ECON880\ECON880\PS2b\inputs"

use Mortgage_performance_data, clear
sum

local varlist  i_large_loan i_medium_loan rate_spread i_refinance score_0 age_r  cltv dti cu first_mort_r    i_FHA  i_open_year2-i_open_year5
tabstat  i_open_0 i_open_1 i_open_2 `varlist', stat(n mean sd min max) columns(statistics)

desc
forvalues i=0/2 {
	gen change_score_`i'=score_`i'-score_0
	probit i_open_`i'  change_score_`i' `varlist', r 
}

order i_large_loan i_medium_loan rate_spread i_refinance  age_r  cltv dti cu first_mort_r  i_FHA  i_open_year2-i_open_year5 score_0 score_1 score_2 duration 
export delimited using Mortgage_performance_data.csv, replace 


