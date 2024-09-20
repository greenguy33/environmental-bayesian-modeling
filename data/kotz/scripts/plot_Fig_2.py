import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import glob

#project the losses of dlgdp onto a world map

#climate and economic data table
data=pd.read_stata("T_econ.dta")

#load masks
#country level
maskc=gpd.read_file('gadm36_levels.gpkg')
#regional level
mask=gpd.read_file('gadm36_levels.gpkg',layer=1)

#lists of helpful identifiers
GID0list=pd.unique(mask.GID_0)
isolist=pd.unique(data.iso)

#results from table 1, Column 7 based on highest R sqr, lowest BIC score
alpha=-11.5
beta=0.192

#empty list to hold values
proj=[]

#calculate marginal effects to plot 
for i in range(len(GID0list)):
	iso=GID0list[i]
	no_regions=len(mask.loc[mask.GID_0==iso])
	if iso in isolist:
		wlrd1list=pd.unique(data.loc[data.iso==iso,'wrld1id_1'])
		for j in range(no_regions):
			reg_no=j+1
			if reg_no in wlrd1list:	
				T_diff=data.loc[(data.iso==iso) & (data.wrld1id_1==reg_no),'T5_seas_diff_m'].mean()
				proj.append(alpha + beta*T_diff)
			else:	
				proj.append(np.nan)
	else:
		for j in range(no_regions):
			proj.append(np.nan)
		
mask['T5_varm_marg_losses']=proj

#generate divergent colormap
colors1 = np.flip(plt.cm.Blues_r(np.linspace(0., 1, 100)),0)
colors3 = np.flip(plt.cm.Reds(np.linspace(0, 1, 100)),0)
colors2 = np.array([1,1,1,1])

vmax=12
vmin=-12
colors = np.vstack((colors3, colors2, colors1))
my_cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)
#my_cmap = shiftedColorMap(my_cmap, start=vmin, stop = vmax, midpoint=0, name='my_cmap')
divnorm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

degree_sign= u'\N{DEGREE SIGN}'
C=degree_sign + 'C'

#simplify shapefile geometry for plotting
mask['geometry_simpl']=mask.geometry.simplify(tolerance=720/43200,preserve_topology=True)
mask_simp=mask.copy()
mask_simp['geometry']=mask_simp.geometry.simplify(tolerance=720/43200,preserve_topology=True)

degree_sign= u'\N{DEGREE SIGN}'
C=degree_sign + 'C'
fs=7
cm=1/2.54
vmin=-12
vmax=0
i=5
plt.close()
fig, ax = plt.subplots(1, 1,figsize=(18*cm,(9+0.5)*cm))
#maskc.plot(ax=ax,edgecolor='black',linewidth=0.1,color='grey')
mask_simp[~mask_simp.T5_varm_marg_losses.isnull()].plot(ax=ax,column='T5_varm_marg_losses',cmap='Reds_r',vmin=vmin,vmax=vmax)
mask_simp[mask_simp.T5_varm_marg_losses.isnull()].plot(ax=ax,color='grey')
ax.set_ylim([-60,85])
ax.set_xlim([-180,180])
ax.tick_params(labelsize='large')
sm = plt.cm.ScalarMappable(cmap='Reds_r', norm=plt.Normalize(vmin=vmin,vmax=vmax))
sm._A = []
ax.tick_params(bottom=False,left=False,labelbottom=False,labelleft=False)
cbar = fig.colorbar(sm,ax = ax,orientation='horizontal',fraction=0.046, pad=0.05)
cbar.set_label('Change in growth rates per extra degree of day-to-day temperature variability (%-points)',rotation=0, labelpad = 12, fontsize=fs)
cbar.ax.tick_params(labelsize=fs)
plt.savefig('Fig_2.pdf',dpi=600,bbox_inches='tight',pad_inches=0)
plt.close()


