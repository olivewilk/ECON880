*delimit
set more off
cd "C:\Users\olivi\Documents\Wisconsin\ECON880\ECON880\PS1b"

use Mortgage_performance_data, clear

local varlist  i_large_loan i_medium_loan rate_spread i_refinance  age_r  cltv dti cu first_mort_r score_0 score_1   i_FHA  i_open_year2-i_open_year5
tabstat  i_close_first_year `varlist', stat(mean sd min max) columns(statistics)
desc
logit i_close_first_year `varlist', r 

order i_close_first_year /*y variable */ i_large_loan i_medium_loan rate_spread i_refinance  age_r  cltv dti cu first_mort_r score_0 score_1   i_FHA  i_open_year2-i_open_year5
export delimited using Mortgage_performance_data.csv, replace 

 