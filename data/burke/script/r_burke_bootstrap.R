burke_data <- read.csv("data/input/GrowthClimateDataset_2_years_withheld.csv")
f <- "growthWDI ~ UDel_temp_popweight + UDel_temp_popweight_2 + UDel_precip_popweight + UDel_precip_popweight_2 | iso + year | 0 | iso"
reg <- felm(f, burke_data, Nboot=1000, keepCX=T, keepX=T)