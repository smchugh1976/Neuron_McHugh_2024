##############################################################
## Import libraries
##############################################################
import pandas as pd
import numpy as np
import sys as sys
import math as math
import os as os
import time as time
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.ticker as Ticker
import scipy as scipy
import scipy.stats as stats
import scipy.ndimage as spim
import scipy.signal as sig
from scipy.signal import savgol_filter
import pyentropy as pye
import fnmatch
import pycircstat as pcs
##############################################################
import vPlotFunctions as vp
import vBaseFunctions3 as vbf
import smBaseFunctions3 as sbf
from my_mpl_defaults import *
##############################################################

##############################################################################################
def getLFPchans(trodes,tet_pos='1'):
    tet_chans = [x for x in trodes['lfp_ch'].where(trodes['desel'] == str(tet_pos))]
    return [int(x) for x in tet_chans if ~np.isnan(x)]
##############################################################################################
def getThetaTimes(ipath,sindx,trodeNo,ftype='theta.cycles'):
    extt1 = '_' + str(sindx) + '.' + ftype + '.' + str(trodeNo)
    thetaTimes,fid = sbf.get_files(ipath,extt1,rev=False,npy=True)
    return thetaTimes
##############################################################################################
def getTrodeNo(trodes,tet_pos='1'):
    trodeNo = [x for x in trodes.index.where(trodes['desel'] == str(tet_pos))]
    return [int(x) for x in trodeNo if ~np.isnan(x)]
##############################################################################################
def get_ripple_chan(baseblock,trodes=None,tet_only=True):
    '''
    
    '''
    ## Get ripple tetrode and channel no, e.g. for theta oscillations
    try:
        tetProfile = np.load(baseblock + '.profile.swr',allow_pickle=True)
        tetLabel = int(tetProfile['info']['refch'])
    except IndexError:
        tetProfile = np.load(baseblock + '.profile.swr',allow_pickle=True).item()
        tetLabel = int(tetProfile['info']['refch'])
    except FileNotFoundError:
        tetProfile = np.load(baseblock + '.tetProfile', allow_pickle=True).item()
        tetLabel = int(tetProfile['ripch'])
    if tet_only:
        return tetLabel
    else:
        chLabel = int(trodes.iloc[tetLabel-1]['lfp_ch'])
        return tetLabel, chLabel
###############################################################################################
def plotRawLFP(idata,xpts=None,figsize=[12,6],lcol='k',lw=.7,pulse=False,axoff=True):
    '''
    
    '''
    alpha = 1.0
    rr,cc = 1,1
    fwid,fht = figsize[0],figsize[1]
    fig, ax = plt.subplots(rr,cc,figsize=sbf.cm2inch(fwid,fht),
                          gridspec_kw = {'wspace':0,'hspace':0})
    ############################################################################################
    if xpts is not None:
        ax.plot(xpts, idata, color=lcol, alpha=alpha, linewidth=lw)
    else:
        ax.plot(idata, color=lcol, alpha=alpha, linewidth=lw)
    if axoff:
        ax.set_axis_off()
    return fig,ax
################################################################################################
def addScaleBar(ax,barwid=250,sv=50,xlab='200 ms',tcol='Black',offset=[.9,.9]):
    '''
    
    '''
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
    from matplotlib.transforms import Bbox
    
    offsetwidth,offsetheight = offset
    asb = AnchoredSizeBar(ax.transData,
                          barwid,
                          xlab,
                          color=tcol,
                          loc='lower right',
                          pad=0.1, 
                          borderpad=0.2,
                          sep=5,
                          frameon=False,
                          size_vertical=sv,
                          bbox_to_anchor=Bbox.from_bounds(0, offsetheight, offsetwidth, 0),
                          bbox_transform=ax.figure.transFigure)
    ax.add_artist(asb)
###############################################################################################
def BPFilter(RawLFP,SamplingRate=1250,Low=30.,High=80.,FilterOrder=4,axis=-1):
    '''
    Basic bandpass filter
    Inputs:
        raw lfp, sampling rate, lower band, upper band, filter order, axis
    Outputs:
        filtered lfp signal
    '''
    LowNorm = (2./SamplingRate) * Low # Hz * (2/SamplingRate)
    HighNorm = (2./SamplingRate) * High # Hz * (2/SamplingRate)
    
    # filtering
    b,a = sig.butter(FilterOrder,[LowNorm,HighNorm],'band') # filter design
    FiltSig = sig.filtfilt(b,a,RawLFP,axis=axis)

    return FiltSig
################################################################################################
def spec_one_sess_median(FRList,sess,mlist=None):
    '''
    
    '''
    mouse = 0
    freqs = FRList[sess][mouse][0]
    nBins = FRList[sess][mouse][1].shape[1]
    
    if mlist is None:
        mlist = range(len(FRList[sess]))
    
    oMat = np.empty([len(mlist),nBins])
    
    for mindx,mouse in enumerate(mlist):
        oMat[mindx,:] = np.nanmedian(FRList[sess][mouse][1],axis=0)
    return freqs,oMat
################################################################################################
def spec_one_sess_all(FRList,sess,mlist=None):
    '''
    '''
    mouse = 0
    freqs = FRList[sess][mouse][0]
    nTrials = FRList[sess][mouse][1].shape[0]
    nBins = FRList[sess][mouse][1].shape[1]
    
    if mlist is None:
        mlist = range(len(FRList[sess]))    
    
    oMat = np.empty([len(mlist),nTrials,nBins])
    
    for mindx,mouse in enumerate(mlist):
        for tindx in range(nTrials):
            oMat[mindx,tindx,:] = FRList[sess][mouse][1][tindx,:]
    return freqs,oMat
################################################################################################
def get_novel_pos(order_nesw,mID,):
    '''
    
    '''
    nesw_to_nov = {'n':'north','e':'east','s':'south','w':'west'}
    nov_pos = []

    for tindx in range(4):
        pos = nesw_to_nov[order_nesw[mID][tindx]] + '_' + 'n' + str(tindx+2)
        nov_pos.append(pos)
        
    return nov_pos
###########################################################################################################################################
def get_fam_pos(nov_pos,allkeys):
    '''
    
    '''
    fam_pos = allkeys.copy()
    for val in nov_pos:
        fam_pos.remove(val)
    
    return fam_pos
############################################################################################################################################
def fam_from_nov(fam_pos,nov_pos,trial='n2'):
    '''
    This generates all e.g. north trials (familiar) based on a single novel trial at same location
    '''
    nov_trial = [x for x in nov_pos if x.endswith(trial)]
    nov_loc = nov_trial[0].rsplit('_', 1)[0]
    
    return [x for x in fam_pos if x.startswith(nov_loc)]
#############################################################################################################################################
def fam_nov_same_trial(nov_pos,fam_pos,sess='n2'):
    '''
    
    '''
    nov_sess = [x for x in nov_pos if x.endswith(sess)][0]
    fam_sess = [x for x in fam_pos if x.endswith(sess)]
    
    return nov_sess,fam_sess
#############################################################################################
def get_stim_inds():
    '''
    indices in green refer to mouse IDs as a list
    '''
    stim_inds = {}
    stim_inds['n2'],stim_inds['n4'] = [0,3,4,7],[0,3,4,7]
    stim_inds['n3'],stim_inds['n5'] = [1,2,5,6],[1,2,5,6]
    
    return stim_inds
#############################################################################################
def get_meanph_diff(ph0c,ph1c):
    '''
    
    '''
    ph0mph = np.angle(np.mean(ph0c))
    ph1mph = np.angle(np.mean(ph1c))

    meanph_diff = np.angle(np.exp(1j*(ph0mph-ph1mph)))

    return meanph_diff
#############################################################################################
def perm_test_circ(idata_ph0,idata_ph1,nperm=1000,plot_fig=True):
    '''
    Function to perform independent permutation test on circular data
    Input:  2 vectors of circular data in radians (idata_ph0, idata_ph1)
            Number of runs for the surrogate data (nperm)
            Whether to output plot (plot_fig)
    Output:
            fig,ax,pvalue if plot_fig=True
            pvalue if plot_fig=False
    '''
    # transforms to polar/complex numbers 
    ph0c = np.exp(1j*idata_ph0) 
    ph1c = np.exp(1j*idata_ph1)
    
    # difference of the mean phases
    meanph_diff = get_meanph_diff(ph0c,ph1c)
    
    # below it's just to ease actual data implementation
    n0_ = len(idata_ph0)
    #n1_ = len(idata_ph1)

    pool = np.concatenate((ph0c,ph1c))
    meanph_diff_perm = np.zeros(nperm)
    
    # test if mean phase diff is significantly different from 0
    np.random.seed(9999) # fix seed?

    for permui in range(nperm):

        ph0_surids = np.random.choice(len(pool),n0_,replace=False)
        ph1_surids = np.setdiff1d(np.arange(len(pool)),ph0_surids)

        ph0c_ = pool[ph0_surids]
        ph1c_ = pool[ph1_surids]

        meanph_diff_perm_ = get_meanph_diff(ph0c_,ph1c_)
        meanph_diff_perm[permui] = meanph_diff_perm_

    # PVALUE FOR TESTING IF DIFFERENCE OF MEANS IS SIGNICANTLY FURTHER AWAY FROM ZERO
    pvalue = np.mean(np.abs(meanph_diff)<=np.abs(meanph_diff_perm))
    
    if plot_fig:
        
        rr,cc = 2,1
        fwid,fht = 20,20
        fig, ax = plt.subplots(rr,cc,figsize=sbf.cm2inch(fwid,fht))

        phedges = np.linspace(-np.pi,np.pi,16)

        count0,bcount,_ = vbf.hist(np.angle(ph0c),phedges)
        count1,bcount,_ = vbf.hist(np.angle(ph1c),phedges)

        ax[0].plot(bcount,count0/np.sum(count0),bcount,count1/np.sum(count1),lw=3)
        ax[0].grid(True)

        phdiff_rad = np.round(meanph_diff,3)
        phdiff_deg = np.round(np.rad2deg(phdiff_rad),1)
        ax[0].set_title('actual mean phase diff = ' + str(phdiff_rad) + ' rad, ' + str(phdiff_deg) + ' deg')
        #ax[0].figure()

        nullcount,bcount,_ = vbf.hist(meanph_diff_perm,30)
        nullcount = nullcount/np.sum(nullcount)

        vp.bar(bcount,nullcount,label='control')

        ax[1].plot(meanph_diff+np.array([0,0]),[0,np.max(nullcount)/2],color='r',lw=5,label='actual')
        ax[1].set_title('pvalue = '+str(np.round(pvalue,2)))
        ax[1].legend()

        return fig,ax,pvalue
    else:
        return pvalue
#########################################################################################################
def total_light_duration(intervalTimes):
    '''
    
    '''
    light_start = intervalTimes['begin'].tolist()
    light_end = intervalTimes['end'].tolist()
    light_dur = [x-y for x, y in zip(light_end, light_start)]
    
    return np.sum(light_dur)
################################################################################
def get_phase_stats(iDict,cluIDs,nbins=24):
    '''

    '''
    ################################################
    nCells = len(cluIDs)

    r = np.empty((nCells))
    mu = np.empty((nCells))
    prph = np.empty((nCells,nbins))
    total_counts = np.empty((nCells))

    for cindx,cluID in enumerate(cluIDs):
        templist = [] # this is for each cluID
        for key,val in iDict.items():
            templist.extend(iDict[key][cindx])

        r[cindx] = pcs.resultant_vector_length(np.array(templist))
        mu[cindx] = pcs.mean(templist)
        #if mu[cindx] < 0:
        #    mu[cindx] += 2*np.pi
        counts,bins,_ = vbf.hist(templist,nbins)
        prph[cindx,:] = counts
        total_counts[cindx] = len(templist)

    phaxis = bins

    return r,mu,prph,phaxis,total_counts
#####################################################################################################################
def theta_coupling(database,output_ext,SF=True,nbins=24,celltype=['p1','p3','pdg','pdgL','b1','b3','bdg','bdgL']):
    '''
    
    '''
    mouseID,allBaseblock,allPar,alldesen,units = sbf.get_all_mouse_db_info(database,SF=SF)
    
    #### create coherence array for each cell type
    allCtypeCohC = {}
    allCtypeMeanPh = {}
    allCtypePrPh = {}
    allCtypeTotalCounts = {}
    allCtypeTotaDur = {}
    allCtypeClusPerMouse = {}
    allDaySince = {}

    for cindx,ctype in enumerate(celltype):

        cohC = np.array([])
        meanPh = np.array([])
        prPh = np.array([]).reshape((0,nbins))
        total_counts = np.array([])
        total_dur = np.array([])
        days_since = np.array([])
        print()
        for mouse,recdayname in enumerate(database):

            # get basic info for each mouse/recday
            desen,baseblock,ipath,bsnm = sbf.get_mouse_info(alldesen,allBaseblock,mouse)
            os.chdir(ipath)

            ## get indices for all cells of given ctype
            temp_cells = units[mouse].loc[units[mouse]['des'].str.strip()==ctype]
            cluID_list = list(temp_cells.index)
            select_clus = [x-2 for x in cluID_list]
            print(recdayname,ctype,len(select_clus))

            array_to_add = np.repeat(sbf.days_since_injection(bsnm),len(select_clus))
            days_since = np.concatenate((days_since,array_to_add))

            # load data from bsnm folder
            trodes = vbf.LoadTrodes(baseblock)
            tetProfile = np.load(baseblock + output_ext,allow_pickle=True).item()

            # select data and concatenate arrays
            # this is r = mean vector length
            cohCtemp = tetProfile['mrl'][select_clus] # r = mean resultant length
            cohC = np.concatenate((cohC,cohCtemp))
            print('{0:} {1:} mean mrl = {2:.2f}'.format(bsnm,ctype,np.nanmean(cohCtemp)))

            # this is a = mean phase angle
            meanPhtemp = tetProfile['mph'][select_clus] # a = mean phase angle
            meanPh = np.concatenate((meanPh,meanPhtemp))
            print('{0:} {1:} mean phase = {2:.2f}'.format(bsnm,ctype,
                                                          np.rad2deg(pcs.mean(meanPhtemp[~np.isnan(meanPhtemp)]))))
            
            try:
                print('CA1 tetrodeNo: {0:}'.format(tetProfile['ca1_trodeNo']))
            except KeyError:
                print('CA1 tetrodeNo: {0:}'.format(tetProfile['ripch']))

            # this is the probability of a spike for a given phase
            prPhtemp = tetProfile['prph'][select_clus,:] # 
            prPh = np.concatenate((prPh,prPhtemp),axis=0)

            tcountstemp = tetProfile['total_counts'][select_clus]
            total_counts = np.concatenate((total_counts,tcountstemp))

            tdurtemp = tetProfile['total_dur']
            total_dur = np.concatenate((total_dur,np.repeat(tdurtemp,len(select_clus))))
            print()

        # write array to data structures
        allCtypeCohC[ctype] = cohC
        allCtypeMeanPh[ctype] = meanPh
        allCtypePrPh[ctype] = prPh
        allCtypeTotalCounts[ctype] = total_counts
        allCtypeTotaDur[ctype] = total_dur
        allDaySince[ctype] = days_since
        phaxis = tetProfile['phaxis']
    
    odata = { 'mrl': allCtypeCohC,'mu': allCtypeMeanPh, 'prph': allCtypePrPh, 
             'total_counts': allCtypeTotalCounts, 'total_dur': allCtypeTotaDur,
             'days_since': allDaySince, 'phaxis': phaxis }
    
    return odata
#################################################################################################################
def plot_polar(idata_mrl,idata_mu,ctype,figsize=[12,12],cellcol=None,xticklabs=[],area=.04):
    '''
    
    '''
    try: sns.reset_orig()
    except: print()
    #################################################################################################################
    r = idata_mrl
    theta = idata_mu
    group_r = np.nanmean(r)
    group_mu = pcs.mean(np.array(theta[~np.isnan(theta)]))
    #################################################################################################################
    rr,cc = 1,1
    fwid,fht = figsize
    polar_yscale = .5
    glw = .8
    msize = 1.0 * fwid/2
    gcolor = 'k'
    lw = 0.5
    area = area * (fwid/2) #(fwid*2) * r**2
    gray2 = [0.6,0.6,0.6]
    if cellcol is None:
        cellcol = {'pdgL':PURPLE,'pdg':BLUE,'p3':RED,'p1':ORNG,'bdg':GREEN,'b3':gray2,'b1':PINK}
        colors = cellcol[ctype]
    else:
        colors = cellcol
    #################################################################################################################
    fig = plt.figure(figsize=sbf.cm2inch(fwid,fht))
    fig.subplots_adjust(wspace=0.2)
    #################################################################################################################
    ax = fig.add_subplot(111, projection='polar')
    c = ax.scatter(theta, r, c=colors, s=area, cmap='hsv', alpha=0.75)
    ################################################################################################################
    ax.set_ylim(0,1)
    ax.yaxis.set_major_locator(Ticker.MultipleLocator(polar_yscale))
    ax.axes.grid(linestyle='--',linewidth=lw,color=gray2)
    ax.set_yticklabels([])
    if xticklabs == 'normal':
        ax.set_xticklabels(['0$^\circ$', '', '90$^\circ$', '', '180$^\circ$', '', '270$^\circ$', ''])
    else:
        ax.set_xticklabels(xticklabs)
    
    ax.xaxis.set_tick_params(labelsize=SMALL_SIZE*fwid/4)
    ax.axvline(group_mu,ymin=0,ymax=group_r,color=gcolor,linewidth=glw,linestyle='-',marker='.',markersize=msize)
    
    return fig,ax,[group_mu,group_r]
######################################################################################################################
def min_spikes(iDict,keys,celltype,min_spk=200):
    '''
    
    '''
    oDict = {}
    
    for kindx,key in enumerate(keys):
        tempDict = {}
        for cindx,ctype in enumerate(celltype):
            tempDat = np.zeros((iDict[key][ctype.shape]))
            indx_sp =  np.argwhere(iDict['total_counts'][ctype] > min_spk)
            tempDict[ctype] = np.ravel(iDict[key][ctype][indx_sp])
    
        oDict[key] = tempDict
    return oDict
######################################################################################################################

