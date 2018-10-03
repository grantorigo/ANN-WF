from __future__ import print_function
import numpy as np
import scipy as sp
import scipy.sparse as sparse
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as colors
import matplotlib.pyplot as pyplot
from matplotlib.ticker import MaxNLocator
import itertools
from numpy import linalg as la
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

'''Reading the configuration file'''
import json
with open('Plot.json') as f:
    file = json.load(f)

npzfile = np.load(file["File name"])

N = 8
nstates = 2**N
ham = 'pm'
model = 'TDT-lrf'
bias = 'y'
lrcurve = 3
dict = {'n':'#1f77b4', 'y':'#d62728'}
dict2 = {3:'Best', 4:'Worst'}

hist_data = np.absolute(npzfile['arr_1'][-2])
hist_data = hist_data[~np.isnan(hist_data)]

tickfont = 13
labelfont = 14

bins = 10.**np.arange(-16.,2.,1.)
fig, ax = plt.subplots()
#fig.suptitle('Model = "'+model+'"; AFH-sign-negative; J = ' + str(J) + '; N = '+str(N)+'; #epochs = $10^{6}$', fontsize=11)
pyplot.hist(hist_data, bins,color = dict[bias],rwidth=0.9 , log = False)#, label='bias = False')
pyplot.xscale('log')
pyplot.legend(loc='upper left', fontsize = tickfont)
pyplot.yticks(fontsize = tickfont)
pyplot.xticks(fontsize = tickfont)
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
ax.set_ylabel("Counts", fontsize = labelfont)
ax.set_xlabel("Relative error ($\\Delta E$)", fontsize = labelfont)
fig.savefig(model+'_AFH-'+ham+'_N'+str(N)+'_bm_Plot.pdf')