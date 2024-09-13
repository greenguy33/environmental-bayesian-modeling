*Stata file to produce regression results for Table 1 of Kotz, Wenz, Stechermesser, Kalkuhl, Levermann (20202)

sysuse 'T_econ.dta'

*declare panel data
xtset ID yearn

eststo clear

*Column 1 
eststo: reghdfe dlgdp_pc_usd T5_varm, absorb(ID yearn) vce(cluster ID)

*Column 2
eststo: reghdfe dlgdp_pc_usd T5_varm T5_mean c.T5_mean_m#c.T5_mean P5_totalpr, absorb(ID yearn) vce(cluster ID)

*Column 3
eststo: reghdfe dlgdp_pc_usd T5_varm D.T5_mean c.T5_mean_m#c.D.T5_mean L.D.T5_mean c.T5_mean_m#c.L.D.T5_mean P5_totalpr, absorb(ID yearn) vce(cluster ID)

*Column 4
eststo: reghdfe dlgdp_pc_usd T5_varm T5_mean c.T5_mean_m#c.T5_mean D.T5_mean c.T5_mean_m#c.D.T5_mean L.D.T5_mean c.T5_mean_m#c.L.D.T5_mean P5_totalpr, absorb(ID yearn) vce(cluster ID)

*Column 5
eststo: reghdfe dlgdp_pc_usd T5_varm c.T5_seas_diff_m#c.T5_varm, absorb(ID yearn) vce(cluster ID)

*Column 6
eststo: reghdfe dlgdp_pc_usd T5_varm c.T5_seas_diff_m#c.T5_varm T5_mean c.T5_mean_m#c.T5_mean P5_totalpr, absorb(ID yearn) vce(cluster ID)

*Column 7
eststo: reghdfe dlgdp_pc_usd T5_varm c.T5_seas_diff_m#c.T5_varm D.T5_mean c.T5_mean_m#c.D.T5_mean L.D.T5_mean c.T5_mean_m#c.L.D.T5_mean P5_totalpr, absorb(ID yearn) vce(cluster ID)

*Column 8
eststo: reghdfe dlgdp_pc_usd T5_varm c.T5_seas_diff_m#c.T5_varm T5_mean c.T5_mean_m#c.T5_mean D.T5_mean c.T5_mean_m#c.D.T5_mean L.D.T5_mean c.T5_mean_m#c.L.D.T5_mean P5_totalpr, absorb(ID yearn) vce(cluster ID)

esttab using Table_1.tex, b(%9.3g) se(%9.1g) ar2 bic drop(_cons) order(T5_varm c.T5_seas_diff_m#c.T5_varm T5_mean c.T5_mean_m#c.T5_mean D.T5_mean c.T5_mean_m#c.D.T5_mean L.D.T5_mean c.T5_mean_m#c.L.D.T5_mean P5_totalpr) nomtitles addnotes("Standard errors clustered at the regional level") 





