# -*- coding: utf-8 -*-
"""
Created on Fri May  8 18:21:07 2020

@author: alessandro
"""

from matplotlib.patches import Circle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import math

def rand_cmap(nlabels, type='bright', first_color_black=True, last_color_black=False, verbose=True):
    """
    Creates a random colormap to be used together with matplotlib. Useful for segmentation tasks
    :param nlabels: Number of labels (size of colormap)
    :param type: 'bright' for strong colors, 'soft' for pastel colors
    :param first_color_black: Option to use first color as black, True or False
    :param last_color_black: Option to use last color as black, True or False
    :param verbose: Prints the number of labels and shows the colormap. True or False
    :return: colormap for matplotlib
    """
    from matplotlib.colors import LinearSegmentedColormap
    import colorsys
    import numpy as np


    if type not in ('bright', 'soft'):
        print ('Please choose "bright" or "soft" for type')
        return

    if verbose:
        print('Number of labels: ' + str(nlabels))

    # Generate color map for bright colors, based on hsv
    if type == 'bright':
        randHSVcolors = [(np.random.uniform(low=0.0, high=1),
                          np.random.uniform(low=0.2, high=1),
                          np.random.uniform(low=0.9, high=1)) for i in xrange(nlabels)]

        # Convert HSV list to RGB
        randRGBcolors = []
        for HSVcolor in randHSVcolors:
            randRGBcolors.append(colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2]))

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]

        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Generate soft pastel colors, by limiting the RGB spectrum
    if type == 'soft':
        low = 0.6
        high = 0.95
        randRGBcolors = [(np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high)) for i in xrange(nlabels)]

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]
        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Display colorbar
    if verbose:
        from matplotlib import colors, colorbar
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(15, 0.5))

        bounds = np.linspace(0, nlabels, nlabels + 1)
        norm = colors.BoundaryNorm(bounds, nlabels)

        cb = colorbar.ColorbarBase(ax, cmap=random_colormap, norm=norm, spacing='proportional', ticks=None,
                                   boundaries=bounds, format='%1i', orientation=u'horizontal')

    return random_colormap
#%%
m=101

A = np.zeros((m,m))
B = np.zeros((m,m))

S = np.zeros(m)
R = np.zeros(m)

new_cmap = rand_cmap(m+1, type='bright', first_color_black=True, last_color_black=False, verbose=False)

#print("There are "+str(math.factorial(m))+" latin squares")
#print("There are "+str(math.factorial(m+1)-math.factorial(m))+" graeco latin squares")
#%%
p = np.random.randint(m) + 1

S = np.arange(m)
np.random.shuffle(S)

R = (p-S)%m

#%%

for k in range(m):
   for l in range(m):
       A[k,l] = ( S[l] + k - 1 )% m    
       
for k in range(m):
   for l in range(m):
       B[k,l] = ( p - S[l] + k - 1 )% m  
#%%
patches=[]

for k in range(m):
   for l in range(m):
       patches.append(Circle((l, k), radius=0.25,color=new_cmap(B[k,l]/float(m-1))))#cm.magma(B[k,l]/float(m-1))))

fig, ax = plt.subplots(1,figsize=(32,32))

ax.imshow(A,cmap=new_cmap)

for pat in patches:
    ax.add_patch(pat)


plt.savefig("gl-square.jpg",bbox_inches='tight')    
plt.show(fig)