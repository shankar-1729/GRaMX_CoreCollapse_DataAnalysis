import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from pylab import cm


# Edit the font, font size, and axes width
mpl.rcParams['font.family'] = 'DejaVu Serif'
mpl.rcParams['font.size'] = 26
#plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.linewidth'] = 2

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
#mpl.rcParams['font.family'] = 'Arial'

# Generate 2 colors from the 'tab10' colormap
#colors = cm.get_cmap('tab10', 2)
c1 = (0.1, 0.2, 0.5)
c2 = (0.5, 0.2, 0.1)

c1='#00f9bb'
c2='#00bff9'
c3='#0c00f9'
c4='#6c00f9'


import matplotlib.font_manager as fm
# Collect all the font names available to matplotlib
font_names = [f.name for f in fm.fontManager.ttflist]
#print(font_names)
#exit(1)

# Create figure object and store it in a variable called 'fig'
#fig = plt.figure(figsize=(6, 6))
fig = plt.figure()
# Add axes object to our figure that takes up entire figure
ax = fig.add_axes([0, 0, 1, 1])

# Edit the major and minor ticks of the x and y axes
ax.xaxis.set_tick_params(which='major', size=12, width=2, direction='in', top=True, pad=8)
ax.xaxis.set_tick_params(which='minor', size=6, width=2, direction='in', top=True)
ax.yaxis.set_tick_params(which='major', size=12, width=2, direction='in', right=True, pad=8)
ax.yaxis.set_tick_params(which='minor', size=6, width=2, direction='in', right=True)


# Edit the major and minor tick locations
'''
ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(40))
ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(20))
ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(2))
ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(1))
'''


# Set the axis limits
#ax.set_xlim(50, 240)
#ax.set_ylim(-0.2, 3.0)


#load data
output, it, t_pb_ms, level, ejecta_mass_M, ejecta_KE_erg, ejecta_IE_erg, ejecta_ME_erg, ejecta_TE_erg, energy_kuroda_method = np.loadtxt("ejecta_mass_energy_bernoulli_gt0.999.txt", unpack=True)

# Plot and show data
#ax.plot(t_pb_ms, ejecta_TE_erg/1.0e50, linewidth=4,  color="orchid", marker="", markersize=12,  label="Explosion energy")
ax.plot(t_pb_ms, ejecta_mass_M*1e3, linewidth=4,  color="tab:blue",  label="Ejecta mass")


# Add the x and y-axis labels
ax.set_xlabel(r't - $\mathregular{t_b}$ [ms]', labelpad=15)
ax.set_ylabel(r'Mass [$\times 10^{-3} M_{\odot}$]', labelpad=15)

# Add legend to plot
ax.legend(bbox_to_anchor=(0, 1), loc=2, frameon=False, fontsize=20)


plt.savefig('ejecta_mass_vs_time_bernoulli_gt0.999.png', dpi=300, transparent=False, bbox_inches='tight')
plt.savefig('ejecta_mass_vs_time_bernoulli_gt0.999.pdf', transparent=False, bbox_inches='tight', format="pdf")
#plt.show()

