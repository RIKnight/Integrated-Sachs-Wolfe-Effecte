#! /usr/bin/env python
"""
Name:
  plot2Ddist
Purpose:
  The plot2Ddist function plots the joint distribution of 2 variables, 
    with estimated density contours and marginal histograms.
  Designed to plot parameter distributions for MCMC samples.
Uses:
  pylab
  numpy
  #pymc
  matplotlib.patches
  mpl_toolkits.axes_grid1
  scipy.stats
Inputs:

Outputs:

Modification History:
  Downloaded from https://gist.github.com/roban/665605 on 2017.02.22
    Presumably written by Roban Hultman Kramer
  Excised all pymc references, since it won't install on my computers;
    Updated obsolete matplotlib.nxutils reference; 
    reversed order of levels in contour_enclosing to fix error; 
    added bw_method to gaussian_kde call; Z Knight, 2017.02.23
  Added bw_method, axSize to plot2Ddist parameter list; ZK, 2017.02.24
  Added nstart to plot2Ddist parameter list; Added returnfigure; 
    ZK, 2017.02.27
  Added myfs for fontsize control; ZK, 2017.04.30

"""

import pylab
import numpy
#import pymc
import matplotlib.patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.stats

def frac_inside_poly(x,y,polyxy):
    """Calculate the fraction of points x,y inside polygon polyxy.
    
    polyxy -- list of x,y coordinates of vertices.

    """
    xy = numpy.vstack([x,y]).transpose()
    #return float(sum(matplotlib.nxutils.points_inside_poly(xy, polyxy)))/len(x)

    myPath = matplotlib.path.Path(polyxy)
    pointsTF = myPath.contains_points(xy)
    return float(numpy.sum(pointsTF))/len(x)

def fracs_inside_contours(x, y, contours):
    """Calculate the fraction of points x,y inside each contour level.

    contours -- a matplotlib.contour.QuadContourSet
    """
    fracs = []
    for (icollection, collection) in enumerate(contours.collections):
        path = collection.get_paths()[0]
        pathxy = path.vertices
        frac = frac_inside_poly(x,y,pathxy)
        fracs.append(frac)
    return fracs

def frac_label_contours(x, y, contours, format='%.3f'):
    """Label contours according to the fraction of points x,y inside.
    """
    fracs = fracs_inside_contours(x,y,contours)
    levels = contours.levels
    labels = {}
    for (level, frac) in zip(levels, fracs):
        labels[level] = format % frac
    contours.clabel(fmt=labels)

def contour_enclosing(x, y, fractions, xgrid, ygrid, zvals, 
                      axes, nstart = 200, 
                      *args, **kwargs):
    """Plot contours encompassing specified fractions of points x,y.
    """

    # Generate a large set of contours initially.
    contours = axes.contour(xgrid, ygrid, zvals, nstart, 
                            extend='both')
    # Set up fracs and levs for interpolation.
    levs = contours.levels
    fracs = numpy.array(fracs_inside_contours(x,y,contours))
    sortinds = numpy.argsort(fracs)
    levs = levs[sortinds]
    fracs = fracs[sortinds]
    # Find the levels that give the specified fractions.
    levels = scipy.interp(fractions, fracs, levs)

    # Remove the old contours from the graph.
    for coll in contours.collections:
        coll.remove()
    # Reset the contours #kludge: reverse order of levels to fix problem
    contours.__init__(axes, xgrid, ygrid, zvals, levels[::-1], *args, **kwargs)
    return contours

def plot2Ddist(variables, axeslist=None, truevalues=None, 
               trimto=None, thin=1, histbinslist=[100, 100],
               labels=None, scaleview=True,
               plotscatter=True, plothists=True, plotcontours=True,
               contourKDEthin=1, contourNGrid=100, 
               contourFractions=[0.6827, 0.9545, 0.9973],
               labelcontours=True, returncontours=False, returnfigure=False,
               bw_method='scott', axSize="50%", nstart=200, myfs = 16,
               scatterstyle={}, histstyle={}, contourstyle={}, **styleArgs):
    """Plot joint distribution of two variables, with marginal histograms.

    The resulting graphic includes (at your discretion):

    * a scatter plot of the 2D distribution of the two variables

    * estimated density contours for the distribution

    * marginal histograms for each variable

    See plot2Ddist_example.py for an example:

    > plot2Ddist([a, b], truevalues=[intercept, slope], **styleargs)

    Notes
    -----

    The contour plotting can be quite slow for large samples because
    of the gaussian kernel density estimation. Try passing a larger
    value for contourKDEthin to speed it up.
    
    Inputs
    ------

    variables -- list-like of length 2
        a list of two array-like or pymc.Variable objects. The lengths
        of the arrays or variable traces should be equal.
        EDIT 2017.02.23 : no more pymc option; must by array-like.

    axeslist -- list-like of length 3
       a list of three Matplotlib Axes for: the joint plot, marginal
       x histogram, and marginal y histogram, respectively.

    truevalues -- list-like of length 2
       a list of the true values for each variable

    trimto -- int
        plot only the last trimto elements of each variable

    thin -- int
        plot only every thin-th element of each variable

    histbinlist -- list-like of length 2
        specify the bins (number or limits) for x and y marginal histograms.

    labels -- list-like of two strings
        the x and y axis labels

    scaleview -- bool
        whether to set the axes limits according to the plotted data

    plotscatter, plothists, plotcontours -- bool
        whether to plot the scatter, marginal histograms, and contours

    scatterstyle, histstyle, contourstyle -- dict-like
        additional keyword arguments for the plot, hist, or contour commands
        
    contourKDEthin -- int
        factor by which to thin the samples before calculating the
        gaussian kernel density estimate for contouring

    contourNGrid -- int
        size of the grid to use (in each dimension) for the contour plotting

    contourFractions -- list-like
        countours are chosen to include the fractions of points specified here

    labelcontours -- bool
        whether to label the contours with the fraction of points enclosed

    returncontours -- bool

    returnfigure -- bool

    bw_method -- string, scalar, or callable
        to be passed to scipy.stats.gaussian_kde for bandwidth selection

    axSize -- float
        controls relative size of histograms compared to 2d plot

    nstart -- float
        controls number of contours generated prior to interpolation

    myfs -- int
        fontsize for plot axis labels
        Default: 16
 
    styleArgs --
        leftover arguments are passed to both the plot and hist commands


    Returns
    -------
    figure, axeslist, contours, depending on parameters

    """
    
    ### Set up figures and axes. ###
    if axeslist is None:
        fig1 = pylab.figure(figsize=(6,6))
        fig1.set_label('traces')
        ax1 = pylab.gca()

        divider = make_axes_locatable(ax1)
        ax2 = divider.append_axes("top", axSize, pad=0.0, sharex=ax1)
        ax3 = divider.append_axes("right", axSize, pad=0.0, sharey=ax1)
        
        for tl in (ax2.get_xticklabels() + ax2.get_yticklabels() +
                   ax3.get_xticklabels() + ax3.get_yticklabels()):
            tl.set_visible(False)
        axeslist = (ax1, ax2, ax3)
    else:
        ax1, ax2, ax3 = axeslist

    # Thin and trim variables.
    if labels is None:
        passedlabels = False
        labels = [None, None]
    else:
        passedlabels = True

    """ # comment out this pymc bit
    for (ivar, variable) in enumerate(variables):
        # Get the trace if this is a pymc.Variable object.
        if isinstance(variable, pymc.Variable):
            variables[ivar] = variable.trace()
            if hasattr(variable, '__name__') and not passedlabels:
                labels[ivar] = variable.__name__ 
    """   

    if trimto is None:
        trimto = len(variables[0])
    x = variables[0][-trimto::thin]
    y = variables[1][-trimto::thin]

    ### Plot the variables. ###

    # Plot 2D scatter of variables.
    if plotscatter:
        style = {'ls':'', 'marker':',', 'color':'r', 'alpha':'0.5'}
        style.update(styleArgs)
        style.update(scatterstyle)
        ax1.plot(x, y, **style)

    if plotcontours:
        xkde = variables[0][-trimto::contourKDEthin]
        ykde = variables[1][-trimto::contourKDEthin]
        # Inspired by Abraham Flaxman's https://gist.github.com/626689
        style = {'linewidths':2.0, 'alpha':0.75, 'colors':'k',
                 #'cmap':matplotlib.cm.Greys,
                 'zorder':10}
        style.update(styleArgs)
        style.update(contourstyle)
        if 'color' in style:
            style['colors'] = style['color']
        gkde = scipy.stats.gaussian_kde([xkde,ykde],bw_method=bw_method)
        xgrid, ygrid = numpy.mgrid[min(x):max(x):contourNGrid * 1j,
                                   min(y):max(y):contourNGrid * 1j]
        zvals = numpy.array(gkde.evaluate([xgrid.flatten(),
                                           ygrid.flatten()])
                            ).reshape(xgrid.shape)
        contours = contour_enclosing(x, y, contourFractions, 
                                     xgrid, ygrid, zvals, 
                                     ax1, nstart=nstart, **style)
    # Plot marginal histograms.
    if plothists:
        style = {'histtype':'step', 'normed':True, 'color':'k'}
        style.update(styleArgs)
        style.update(histstyle)
        ax2.hist(x, histbinslist[0], **style)
        ax3.hist(y, histbinslist[1], orientation='horizontal', **style)

    # Plot lines for the true values.
    if truevalues is not None:
        ax1.axvline(x=truevalues[0], ls=':', c='k')
        ax1.axhline(y=truevalues[1], ls=':', c='k')
        ax2.axvline(x=truevalues[0], ls=':', c='k')
        ax3.axhline(y=truevalues[1], ls=':', c='k')

    if scaleview:
        ax2.relim()
        ax3.relim()
        ax1.relim()
        ax2.autoscale_view(tight=True)
        ax3.autoscale_view(tight=True)
        ax1.autoscale_view(tight=True)
        ax2.set_ylim(bottom=0)
        ax3.set_xlim(left=0)

    #myfs = 16 # font size for labels
    if labels[0] is not None:
        ax1.set_xlabel(labels[0],fontsize=myfs)
    if labels[1] is not None:
        ax1.set_ylabel(labels[1],fontsize=myfs)
        
    if plotcontours and labelcontours:
        frac_label_contours(x, y, contours)

    if returnfigure:
      if plotcontours and returncontours:
          return fig1, axeslist, contours
      else:
          return fig1, axeslist
    else:
      if plotcontours and returncontours:
          return axeslist, contours
      else:
          return axeslist
