##################################################################################
#		
# Title:	Anthropogenic Climate Change Has Slowed Global Agricultural 
# 		Productivity Growth
# 				
# Authors: 	Ariel Ortiz-Bobea, Cornell University (ao332@cornell.edu)
# 		Toby R. Ault, Cornell University
#		Carlos M. Carrillo, Cornell University
#		Robert G. Chambers , University of Maryland, College Park
#		David B. Lobell, Stanford University
#
##################################################################################

##################################################################################
# A. General instructions 
##################################################################################

This project is designed to be executed in R Studio. Load the TFP_global_R.Rproj 
project file in the “TFP_global_R” folder to load the R Studio project. Execute 
script files in the order they are listed.

Data sources:

- International agricultural Total Factor Productivity (TFP) from USDA ERS 
https://www.ers.usda.gov/data-products/international-agricultural-productivity/

- Princeton's Global Meteorological Forcing Dataset for land surface modeling
http://hydrology.princeton.edu/data.pgf.php

- Normalized Difference Vegetation Index (NDVI) - 3rd generation: NASA/GFSC GIMMS
https://climatedataguide.ucar.edu/climate-data/ndvi-normalized-difference-vegetation-index-3rd-generation-nasagfsc-gimms
https://ecocast.arc.nasa.gov/data/pub/gimms/3g.v1/

Notes regarding reproducibility:

Script files starting with "1.x_" in their name are for data preparation and 
cleaning. Most of these script files will NOT run because we do not include the raw 
data files for Princeton's GMFD weather or CMIP6. Files are simply too large to 
conveniently share. For some datasets we include just one raw data file for 
illustration. Thus the following folders are empty:
	../data2/Princeton
	../data/CMIP6
	../data/Princeton/v3/0.25deg/monthly

We suggest you run script file "2_regression_analysis.R" which directly reproduces the results 
in the paper. Because every model is estimated with a bootstrap (B=500) and a number 
of those models also entail a permutation/robustness checks (R=10,000), this portion of 
the code can take some time depending on your computing resources (~6 hours on a 
2019 iMac, 3.6 GHz Intel Core i9, 64 GB and 8 cores engaged). 

If you have any questions or wish to express anything to the authors, please reach
out to Prof. Ariel Ortiz-Bobea at the email indicated above.


##################################################################################
# B. Folder structure 
##################################################################################

Folder path		Description

../TFP_global_R 	R script files
../data				Raw datasets
../data2			Intermediate data files
../figures/paper	Where figures in the main text will be saved by the script
../figures/SM		Where figures in the Supplementary Material  will be saved by the script
../tables 			Where tables in the Supplementary Material will be saved by the script


##################################################################################
# C. Description of script files
##################################################################################

Note: Script files that start with "1.x" refer to processes of data preparation and
cleaning. Script file "2" is the main analysis script file.

- 1.1_global_map.R
Creates global map of countries that matches the countries in the 
USDA-ERS TFP dataset 

- 1.2_monthly_gridded_obs_weather.R
Aggregagte daily Princeton weather dataset to monthly time scale.

- 1.3_aggregation_critera.R
Resample fine-scale land cover data into coarser grid of weather
data for the purpose of creating weights for spatially aggregating 
gridded climate data to the country level. Data source for cropland and 
pasture area in 2000: http://www.earthstat.org/cropland-pasture-area-2000 

- 1.4_grid_to_country_aggregation_matrices.R
Create transformation matrices to convert gridded data from
Princeton data to country level dataset based on various criteria (e.g.  
cropland, pasture or both). These matrices are used in subsequent files for 
extraction. This speeds up grid-to-country aggregation.

- 1.5_country-level_weather.R
Prepares country-level weather data based on the various grid
level weights (cropland, pasture, and both).

- 1.6_monthly_gridded_gcm.R
Prepares country-level GCM data based on the various grid
level weights (cropland, pasture, or both).

- 1.7_peak_bottom_ndvi.R
Find the greenest and least green months of the year for each country 
with TFP data based on NDVI data. Country-level aggregation done based 
on various grid level weights (cropland, pasture, or both).

- 1.8_global_ag_composition
Create figure of composition of Global Net Production Value.

- 2_regression_analysis.R
Performs regression analysis and computes impacts of anthropogenic
climate change.


##################################################################################
# D. List of figures and tables
##################################################################################

Here we provide a list of files generated the figure names and what script file was used
To generate the figure (number provided in brackets, e.g. [2] means
The figure is generated in script file 2_regression_analysis.R)

Figures in the paper (/figures/paper):
- Fig 1 [Script 2]: Fig1_sum_stats_tfp_variation.pdf
- Fig 2 [Script 2]: Fig2_response_tfp_fd_equal_tmean_prec_quad_green5_pooled_year.FAOregion_0_cropland_none_1962-2015_nboot500_nperm10000.pdf
- Fig 3 [Script 2]: Fig3_global_impact_tfp_fd_equal_tmean_prec_quad_green5_pooled_year.FAOregion_0_cropland_none_prec_nboot500_ngcms7_npairs2000.pdf
- Fig 4 [Script 2]: Fig4_global_impact_tfp_nboot500_ngcms7_npairs2000_nmodels96.pdf
- Fig 5 [Script 2]: Fig5_regional_impact_tfp_all_nboot500_ngcms7_npairs2000.pdf

Extended Data Figures (/figures/SM):
- Extended Data Fig 1  [Script 1.8]: global_ag_composition.pdf
- Extended Data Fig 2  [Script 2]: Fig_response_tfp_fd_equal_tmean_prec_quad_green5_pooled_year.FAOregion_0_cropland_cold10_1962-2015_nboot500_nperm10000.pdf
- Extended Data Fig 3  [Script 2]: Fig_response_tfp_fd_equal_tmean_prec_quad_green5_pooled_year.FAOregion_0_cropland_hot10_1962-2015_nboot500_nperm10000.pdf
- Extended Data Fig 4  [Script 2]: Fig_response_tfp_fd_equal_tmean_prec_quad_green5_pooled_year.FAOregion_0_cropland_none_1962-1988_nboot500_nperm10000.pdf
- Extended Data Fig 5  [Script 2]: Fig_response_tfp_fd_equal_tmean_prec_quad_green5_pooled_year.FAOregion_0_cropland_none_1989-2015_nboot500_nperm10000.pdf
- Extended Data Fig 6  [Script 2]: Fig_response_tfp_fd_equal_tmean_prec_quad_green5_pooled_year.FAOregion_0_cropland_cold10_1962-1988_nboot500_nperm10000.pdf
- Extended Data Fig 7  [Script 2]: Fig_response_tfp_fd_equal_tmean_prec_quad_green5_pooled_year.FAOregion_0_cropland_cold10_1989-2015_nboot500_nperm10000.pdf
- Extended Data Fig 8  [Script 2]: Fig_response_tfp_fd_equal_tmean_prec_quad_green5_pooled_year.FAOregion_0_cropland_hot10_1962-1988_nboot500_nperm10000.pdf
- Extended Data Fig 9  [Script 2]: Fig_response_tfp_fd_equal_tmean_prec_quad_green5_pooled_year.FAOregion_0_cropland_hot10_1989-2015_nboot500_nperm10000.pdf
- Extended Data Fig 10 [Script 2]: Fig_global_impact_tfp_nboot500_ngcms7_npairs2000_nmodels298.pdf

Supplementary Information (/figures/SM):
- Supplementary Fig 1 [Script 2]: Fig_response_output_fd_equal_tmean_prec_quad_green5_pooled_year.FAOregion_0_cropland_none_1962-2015_nboot500_nperm10000.pdf
- Supplementary Fig 2 [Script 2]: Fig_global_impact_output_fd_equal_tmean_prec_quad_green5_pooled_year.FAOregion_0_cropland_none_FALSE_nboot500_ngcms7_npairs2000.pdf
- Supplementary Fig 3 [Script 1.1]: FAO_regions.pdf
- Supplementary Fig 4 [Script 1.4]: climate_aggregation_criteria.pdf
- Supplementary Fig 5 [Script 1.7]: peak_season_maps_all.pdf
- Supplementary Tab 1 [Script 2]: reg_baseline_model.csv
- Supplementary Tab 2 [Script 2]: lag_tests.csv
- Supplementary Tab 3 [Script 2]: acc_impacts_regional_tfp_all_nboot500_ngcms7_npairs2000.csv
- Supplementary Tab 4 [Script 2]: gcms.csv 

A ComparisonOutputFiles folder is included in this package.  It contains a copy of the outputs produced by the 2_regression_analysis.R code for comparison purposes only.




##################################################################################
# Analytic Data Citation and License
##################################################################################   

Data Creator 1 Name: Ortiz-Bobea, Ariel  
Data Creator 1 ORCID: https://orcid.org/0000-0003-4482-6843   

Data Creator 2 Name: Carillo, Carlos M.  
Data Creator 2 ORCID: https://orcid.org/0000-0002-0045-1595    

Year of Publication: 2021 
Title of Reproducibility package:  "Reproduction Materials for: Anthropogenic Climate Change Has Slowed Global Agricultural Productivity Growth" (version 2) (Analytic Data}.    
Location and Distribution          Ithaca, NY:  Cornell Institute for Social and Economic Research (distributor)   
DOI: https://doi.org/10.6077/pfsd-0v93   

Data Restrictions:   

**Data License:**  

License:  CC-BY 4.0  https://creativecommons.org/licenses/by/4.0/    

##################################################################################
# Code Citation and License
################################################################################## 

Code Author 1 Name: Ortiz-Bobea, Ariel   
Code Author 1 ORCID: https://orcid.org/0000-0003-4482-6843    

Code Author 2 Name: Carillo, Carlos M . 
Code Author 2 ORCID: https://orcid.org/0000-0002-0045-1595       
 
Year of Publication: 2021  
Title of Reproducibility package:  "Reproduction Materials for: Anthropogenic Climate Change Has Slowed Global Agricultural Productivity Growth" (version 2) (Codes}.    
Location and Distribution          Ithaca, NY:  Cornell Institute for Social and Economic Research (distributor)   
DOI: https://doi.org/10.6077/pfsd-0v93  

Code last executed from start to finish: 22-Feb-2021    
Software 1 name and version: R 4.0.2
Software 2 name and version: R Studio 1.3.959
Character encoding:  UTF-8
Computer Operating System: Windows Server 2019 Datacenter    
Computer Processor: Intel(R) Xeon(R)CPU E5-4669 v3 @ 2.10GHz (2 processors)    
Computer Memory (RAM): 256 GB    
Computer System type : 64-bit OS

**Code License:**  

BSD-3-Clause   

Copyright 2021 Ariel Ortiz-Bobea and Carlos Carillo

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:  
1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.  
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer 
   in the documentation and/or other materials provided with the distribution.  
3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from  
   this software without specific prior written permission.  

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE 
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE 
GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT 
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


The End