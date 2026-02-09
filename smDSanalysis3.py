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
import matplotlib.ticker as Ticker
import scipy as scipy
import scipy.stats as stats
import scipy.ndimage as spim
import pyentropy as pye
import fnmatch
import collections
#####################################################################################
import vBaseFunctions3 as vbf
import smBaseFunctions3 as sbf
import smLfpFunctions3 as slf
#####################################################################################
'''
class OrderedSet(collections.Set):
    def __init__(self, iterable=()):
        self.d = collections.OrderedDict.fromkeys(iterable)

    def __len__(self):
        return len(self.d)

    def __contains__(self, element):
        return element in self.d

    def __iter__(self):
        return iter(self.d)
'''
#######################################################################
def get_ds_info(offset):
    '''

    '''
    lfp_sr = 1250
    spk_sr = 20000
    tconv = (lfp_sr / spk_sr)
    sampleDur = 2 * offset
    ds_ref = (offset * lfp_sr)
    dur_ms = sampleDur * lfp_sr
    
    return lfp_sr,spk_sr,tconv,sampleDur,ds_ref,dur_ms
#####################################################################################
def getDSParams(ipath,ftype='.ds.thresh',nsplit=2,rev=False,npy=False):
    '''
    '''
    extt1 = ftype
    tempDat,fids = sbf.get_files(ipath,extt1,rev=rev,npy=npy,pprint=False)
    
    odata = {}
    
    for kindx,ikey in enumerate(fids):
        okey = ikey.rsplit('.',nsplit)[0]
        odata[okey] = tempDat[kindx]
        
    return odata
#################################################################################################################
def get_swr_params(ipath,bsnm,ftype='.swr.events',okey='events'):
    '''
    Returns a list [no. sessions] with each an [events, 3] numpy array of times at 1250Hz sampling rate
    '''
    fullpath = ipath + '/' + bsnm + ftype
    try:
        tempDat = np.load(fullpath,allow_pickle=True).item()
    except AttributeError:
        tempDat = np.load(fullpath,allow_pickle=True)
    except FileNotFoundError:
        print('{0:}.swr.events does not exist in the directory. Please run SM_EMD_batch on this dir'.format(bsnm))
        return

    return tempDat[okey]
####################################################################################################################
def get_ds_session(ipath,ds_times_ext = '.ds'):
    '''
    
    '''
    os.chdir(ipath)
    print(ipath)
    ######################################################################################
    DStimes = getDSParams(ipath,ftype=ds_times_ext)
    ######################################################################################
    max_count = 0
    for kindx,key in enumerate(DStimes):
        sess_indx = int(key.rsplit('_',1)[-1])
        print(sess_indx,key,DStimes[key].shape[0])
        if sess_indx > 1:
            temp_count = DStimes[key].shape[0]
            if temp_count > max_count:
                max_count = temp_count
                okey = key

    return okey,max_count
#############################################################################################################
def get_sess_key_from_filebase(desen,okey):
    '''
    
    '''
    return desen[desen['filebase']==okey]['desen'].values[0]
#############################################################################################################
def get_ds_data(DStimes,nch,chNo,offset=0.08,conversion=0.195):
    '''
    
    '''
    lfp_sr,spk_sr,tconv,sampleDur,ds_ref,dur_ms = get_ds_info(offset)
    
    allDat = {}
    for tnindx,tn in enumerate(chNo):
        nBins = int(sampleDur * lfp_sr)
        oMat = np.empty([0,nBins])

        for kindx,key in enumerate(DStimes):
            fname = key + '.eeg'
            #print(fname)
            eegMap = vbf.MapLFPs(fname,nch)

            for dindx in range(DStimes[key].shape[0]):
                try:
                    startIndx = int(DStimes[key][dindx] * tconv) - int(offset * lfp_sr)
                except TypeError:
                    startIndx = int(DStimes[key][dindx][0] * tconv) - int(offset * lfp_sr)
                endIndx = int(startIndx + (lfp_sr * sampleDur))
                dur_ms = (endIndx - startIndx) * (1000 / lfp_sr)
                tempDat = (eegMap[chNo[tnindx]][startIndx:endIndx]).reshape(1,-1)
                try:
                    oMat = np.concatenate((oMat,tempDat),axis=0)
                except ValueError:
                    print('''Couldn't add trial - start or end of file??''')

        allDat[tn] = oMat * conversion
        
    return allDat
#######################################################################################
def get_triggered_waveform(itimes,nch,chNo,offset=0.08,conversion=0.195):
    '''
    
    '''
    lfp_sr,spk_sr,tconv,sampleDur,ds_ref,dur_ms = get_ds_info(offset)
    
    #allDat = {}
    #for tnindx,tn in enumerate(chNo):
    nBins = int(sampleDur * lfp_sr)
    oMat = np.empty([0,nBins])

    for kindx,key in enumerate(itimes):
        fname = key + '.eeg'
        #print(fname)
        eegMap = vbf.MapLFPs(fname,nch)

        for dindx in range(itimes[key].shape[0]):
            startIndx = int(itimes[key][dindx] * tconv) - int(offset * lfp_sr)
            endIndx = int(startIndx + (lfp_sr * sampleDur))
            #dur_ms = (endIndx - startIndx) * (1000 / lfp_sr)
            tempDat = (eegMap[chNo,startIndx:endIndx]).reshape(1,-1)
            try:
                oMat = np.concatenate((oMat,tempDat),axis=0)
            except ValueError:
                print('''Couldn't add trial - start or end of file??''')

    oMat = oMat * conversion
        
    return oMat
#######################################################################################
def get_slope(idata,xrange=[50,100],nbins=None):
    '''

    '''
    if nbins is None:
        nbins = xrange[1] - xrange[0]
    X = np.linspace(xrange[0],xrange[1],nbins)
    Y = idata[xrange[0]:xrange[1]]

    return np.polyfit(X,Y,1)[0]
#######################################################################################
def get_max_chan(iDict,pprint=False):
    '''

    '''
    max_amp = 0
    oDict = {}
    for key,val in iDict.items():
        idata = np.nanmedian(iDict[key],axis=0)
        oDict[key],maxval = abs(np.nanmax(idata)),abs(np.nanmax(idata))
        if pprint:
            print(key,maxval)
        if maxval > max_amp:
            max_amp = maxval
            max_key = key

    return oDict,max_key
######################################################################################
def get_thresh_chan(iDict,trodes,conversion=0.195):
    '''
    
    '''
    oDict = {}
    for key,val in iDict.items():
        trode_key = str(trodes['lfp_ch'][int(key.rsplit('.',1)[-1])])
        oDict[trode_key] = iDict[key] * conversion
    
    return oDict
######################################################################################
def crossings_nonzero_all(idata):
    '''

    '''
    pos = idata > 0
    npos = ~pos

    return ((pos[:-1] & npos[1:]) | (npos[:-1] & pos[1:])).nonzero()[0]
######################################################################################
def calc_DS_width(idata,samp_to_ms=(1000/1250.),min_tbin=40):
    '''

    '''
    zero_crossings = crossings_nonzero_all(idata)
    print(len(zero_crossings),zero_crossings)

    if (len(zero_crossings) == 2) and (zero_crossings[0] > min_tbin):
        odata = (zero_crossings[1] - zero_crossings[0]) * samp_to_ms
    else:
        print('More than 2 crossings detected or crossing too early')
        try:
            odata = (zero_crossings[-1] - zero_crossings[-2]) * samp_to_ms
        except:
            odata = np.nan

    return odata
################################################################################################
def save_ds_data(opath,odata,group_type,ctype,first_str='Allcells_',last_str='',cellaxis=-1):
    '''

    '''
    import datetime

    os.chdir(opath)
    fname = first_str + group_type + '_n' + str(odata.shape[cellaxis]) + '_' + ctype + last_str + '.npy'
    print('saving file {}'.format(fname))
    now = datetime.datetime.now()
    print ('Time saved: ' + now.strftime("%Y-%m-%d %H:%M:%S"))
    fHand = open(fname,'wb')
    np.save(fHand,odata)
    fHand.close()
################################################################################################
def calc_DS_rate(DStimes,ext='.immobile_dur'):
    '''
    
    '''
    DSrate = {}
    immobile = {}

    for kindx,key in enumerate(DStimes):
        fname = key + ext
        immobile[key] = np.load(fname,allow_pickle=True)
        try:
            DSrate[key] = DStimes[key].shape[0] / immobile[key]
        except:
            DSrate[key] = np.nan

    return DSrate,immobile
################################################################################################
def append_new_line(opath,ofname,text_to_append):
    '''
    Go to output path
    Open file in append, binary mode (even though txt)
    Append text as a new line at end of file
    '''
    os.chdir(opath)
    print('saving to: {}'.format(opath))
    with open(ofname, "a+") as file_object:
        # Move read cursor to the start of file.
        file_object.seek(0)
        # If file is not empty then append '\n'
        data = file_object.read(100)
        if len(data) > 0:
            file_object.write("\n")
        # Append text at the end of file
        file_object.write(text_to_append)
#################################################################################################
def ds_times_by_class(DStimes,DSclass,target):
    '''

    '''
    odata = {}
    for ikey,ival in DStimes.items():
        tempClass = DSclass[ikey]
        tempTimes = DStimes[ikey]
        odata[ikey] = tempTimes[tempClass==target-1]

    return odata
#####################################################################
def count_ds(DStimes):
    '''
    
    '''
    total_ds = 0
    for kindx,key in enumerate(DStimes):
        total_ds += DStimes[key].shape[0]

    return total_ds
#####################################################################
def ds_class_freq(idata):
    (unique, counts) = np.unique(idata,return_counts=True)

    return np.asarray((unique,counts)).T
#####################################################################
def reverse_keys(iDict):
    '''
    reverses an outer for an inner key in a nested dictionary
    '''
    from collections import defaultdict
    
    flipped = defaultdict(dict)
    for key,val in iDict.items():
        for subkey,subval in val.items():
            flipped[subkey][key] = subval

    return flipped
#####################################################################################
def get_ds_data_by_class(DStimes,DSclass,nch,chNo,offset=0.08,conversion=0.195):
    '''
    
    '''
    lfp_sr,spk_sr,tconv,sampleDur,ds_ref,dur_ms = get_ds_info(offset)

    allDat = {}
    for tnindx,tn in enumerate(chNo):
        nBins = int(sampleDur * lfp_sr)
        oMat = np.empty([0,nBins])

        for kindx,key in enumerate(DStimes):
            fname = key + '.eeg'
            eegMap = vbf.MapLFPs(fname,nch)
            
            for dindx in range(DStimes[key].shape[0]):
                startIndx = int(DStimes[key][dindx] * tconv) - int(offset * lfp_sr)
                endIndx = int(startIndx + (lfp_sr * sampleDur))
                dur_ms = (endIndx - startIndx) * (1000 / lfp_sr)
                tempDat = (eegMap[chNo[tnindx]][startIndx:endIndx]).reshape(1,-1)
                try:
                    oMat = np.concatenate((oMat,tempDat),axis=0)
                except:
                    print('''Couldn't add trial - start or end of file??''')

        allDat[tn] = oMat * conversion
        
    return allDat
#################################################################################################################
def pulsecorrelation(baseblock,desen,sessions,cluID,all_times=None,binwidth=1,nms=[40.,40.],ext='.light_pulse'):
    '''
    This function takes a baseblock,desen dataframe and a list of sessions
    and a cluID and generates an cross-correlation based on pulsetimes in all sessions given
    binwidth and nmsBefore, nmsAfter are passed as options
    returns a matrix (oMat) which is timebins x 1 x totalspikes
    use plot_ac to visualize
    '''
    # Create output matrix
    tbins  = int(2*(nms[0]/binwidth))
    oMat = np.zeros((tbins,1,0))
    bsnm = baseblock.rsplit('/',1)[-1]
    # Loop through sessions
    for sindx,sess in enumerate(sessions):
        ### Select session
        sessionLabel = sbf.get_descode(desen,sess)
        fname = baseblock + '_' + str(sessionLabel['filebase'].index[0])
        okey = bsnm + '_' + str(sessionLabel['filebase'].index[0])
        ##############################################################################################
        res,clu = \
        vbf.LoadSpikeTimes(fname,trode=None,MinCluId=2,res2eeg=(20000./20000))
        ipath = os.path.split(baseblock)[0] + '/'
        #refTimes = []
        if all_times is not None:
            print('Using passed refTimes for pulses')
            try:
                refTimes = all_times[okey]
            except KeyError:
                refTimes = False
        else:
            print('Getting refTimes from file, with ext {0:}'.format(ext))
            refTimes = sbf.get_pulsetimes(ipath,desen,sess,ext,tconv=None,debug=False)
            refTimes = refTimes['begin'].values
        if len(refTimes) > 0:
            refEdges = sbf.generate_edges(refTimes,binwidth=binwidth,nmsBefore=nms[0],nmsAfter=nms[1])
            tempMat = sbf.generate_3Difr_matrix(res,clu,[cluID],refEdges)
            oMat = np.concatenate((oMat,tempMat),axis=2)
        ##############################################################################################

    return oMat # timebins x 1 x totalspikes
######################################################################################################


######################################################################################################
def get_two_sessions(desen,indx1=0,indx2=-1):
    '''
    Designed to get first and last sessions
    from a dataframe
    '''
    filebase1 = desen['filebase'].iloc[indx1]
    filebase2 = desen['filebase'].iloc[indx2]
    
    return filebase1, filebase2
######################################################################################################
def get_dsrate_from_file(ipath,bsnm_sess,ext='.DSrate'):
    '''
    e.g. ipath = '/mnt/smchugh/lfpd4/SF/msm06-161024'
    fname = 'msm06-161024_4'
    ext = '.DSrate'
    '''
    fullpath = ipath + '/' + bsnm_sess + ext
    odata = np.load(fullpath)

    return odata
######################################################################################################
def plot_median_data(idata,ftitle,offset=0.08,xpts=None,thresh=None,figsize=None,xlim=[0,200],ylim=None,xoffset=80,
                     xtick_width=50,ytick_width=500,lw=.5,tfsize=7,axoff=True,zero_lines=True,scalebar=False):
    '''

    '''
    try: sns.reset_orig()
    except NameError: print('No Seaborn')

    if figsize is None:
        figsize = [12,12]
    lfp_sr,spk_sr,tconv,sampleDur,ds_ref,dur_ms = get_ds_info(offset)
    dur_ms = sampleDur * lfp_sr
    soffset = [1.02,-.01]
    sv = 500 # scale bar height in microvolts
    xlab = '40 ms'
    lcol = 'k'
    twid = dur_ms / 4.0 # scale bar width in ms
    ##########################################################################
    ## Create figure/axis and plot data
    fig,ax = slf.plotRawLFP(idata,xpts=xpts,figsize=figsize,lcol=lcol,lw=lw,axoff=axoff)
    ## Adjust plot
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    # Set width of tick marks
    ax.xaxis.set_major_locator(Ticker.MultipleLocator(xtick_width))
    ax.yaxis.set_major_locator(Ticker.MultipleLocator(ytick_width))
    # Rescale x-axis labels
    xtlabs = ax.get_xticks().tolist()
    xtlabs = [format(x*(1000/lfp_sr)-xoffset,'.0f') for x in xtlabs]
    ax.xaxis.set_ticklabels(xtlabs, fontsize=tfsize)
    ##
    if thresh is not None:
        ax.axhline(thresh,color='r',linewidth=lw,linestyle='--')
    if zero_lines:
        ax.axhline(0,color='k',linewidth=lw,linestyle='-')
        ax.axvline(ds_ref,color='k',linewidth=lw,linestyle='--')
    ax.set_title(ftitle,loc='left',fontsize=10)
    if scalebar:
        slf.addScaleBar(ax,barwid=dur_ms/4.0,sv=5,xlab=xlab,offset=soffset)
        slf.addScaleBar(ax,barwid=1,sv=sv,xlab='',offset=soffset)

    return fig,ax
########################################################################
def get_max_zval(idata):
    '''
    
    '''
    maxvalz = np.nanmax(idata,axis=0)
    argmaxz = np.argmax(idata,axis=0)

    return maxvalz[np.isfinite(maxvalz)],argmaxz[np.isfinite(argmaxz)]
#######################################################################
def phat_cells(idata,zthresh=3):
    return idata[idata>zthresh].shape[0] / idata.shape[0]
#######################################################################
def get_metric(idata,method=np.nanmax,axis=0):
    return method(idata,axis=axis)
####################################################################
def ms_to_bin(nms,binwidth,binval):
    '''
    
    '''
    min_ms,max_ms = -nms[0],nms[1]
    ms_range = np.arange(min_ms,max_ms,binwidth)
    #print(ms_range)
    odata = np.where(ms_range==binval)[0][0]
    #print(ms_range.shape,odata)

    return odata
####################################################################


###################################################################################################################
def get_cluIDs(origIDs,ctype,mouse):
    return origIDs[ctype][mouse]
###################################################################################################################
def plotRawLFP_raster(idata_lfp,idata_ca1,idata_raster,color_code,figsize=[36,24],lcol='k',cmap='jet',lw=.5,alpha=1.0,
                      axoff=True,sp_tick_height=.004,rast_w=2.5,v_ofs=0.01):
    '''
    
    '''
    from matplotlib import cm
    try: sns.reset_orig()
    except: print('No Seaborn')
    rr,cc = 3,1
    fwid,fht = figsize[0],figsize[1]
    vert_offset = sp_tick_height + v_ofs
    ############################################################################################
    ax = {}
    fig, ax = plt.subplots(rr,cc,
                           figsize=sbf.cm2inch(fwid,fht),
                           gridspec_kw = {'wspace':0,'hspace':0,'height_ratios': [1,2,8]},
                           sharex=True)
    ############################################################################################
    ax[0].plot(idata_ca1, color=lcol, alpha=alpha, linewidth=lw)
    ax[1].plot(idata_lfp, color=lcol, alpha=alpha, linewidth=lw)
    ############################################################################################
    clu_count = 0
    for clu_ in range(len(idata_raster)):
        if len(idata_raster[clu_]) > 0:
            clu_offset = clu_count * vert_offset
            clu_count += 1
            ymin = (.98-sp_tick_height) - clu_offset
            ymax = ymin + sp_tick_height
            if ymin < 0:
                print('ymin is off the screen for {0:}'.format(clu_))
            print(clu_count,clu_,len(idata_raster[clu_]),np.around(ymin,3),np.around(ymax,3))
            if cmap is not None:
                try:
                    colors = cm.get_cmap(cmap,len(idata_raster))
                    sp_col = colors(clu_ * (1/len(idata_raster)))
                except:
                    sp_col = color_code[clu_]
            else:
                sp_col = 'k'

            for sp_ in range(len(idata_raster[clu_])):
                ax[2].axvline(idata_raster[clu_][sp_],ymin=ymin,ymax=ymax,linewidth=lw*rast_w,color=sp_col)
    if axoff:
        for ax_ in ax:
            ax_.set_axis_off()

    return fig,ax
####################################################################################################################
def get_spike_data_col_code(origIDs,celltype,res,clu,startIndx,endIndx,mouse):
    '''
    
    '''   
    colc = [ORNG,RED,BLUE,PURPLE]

    cluID_list = {}
    idata_raster = []
    color_code = []

    for cindx,ctype in enumerate(celltype):
        cluID_list[ctype] = get_cluIDs(origIDs,ctype,mouse)
        for cluID in cluID_list[ctype]:
            spikes = res[clu==cluID]
            idata_raster.append([i-startIndx for i in spikes 
                                 if i >= startIndx and i <= endIndx])
            color_code.append(colc[cindx])
 
    return idata_raster,color_code
######################################################################################################################
def DStimes_by_stim(skey,DStimes):
    '''
 
    '''
    newDStimes = {}
    for innerkey,val in skey.items():
        print(innerkey)
        temptimes = {}
        for outerkey in skey[innerkey]:
            print(outerkey)
            temptimes[outerkey] = DStimes[outerkey]
        newDStimes[innerkey] = temptimes

    return newDStimes
#######################################################################################################################

#######################################################################################################################
def DStimes_by_key(skey,DStimes,tconv):
    '''
    
    '''
    newDStimes = {}
    for innerkey,val in skey.items():
        print(innerkey)
        temptimes = {}
        for outerkey in skey[innerkey]:
            print(outerkey)
            if DStimes[outerkey].shape:
                temptimes[outerkey] = np.array([int(x) for x in DStimes[outerkey]])
            else:
                temptimes[outerkey] = None
        newDStimes[innerkey] = temptimes
        
    return newDStimes
######################################################################################################################
def SWRtimes_by_key(skey,SWRtimes):
    '''
    
    '''
    newSWRtimes = {}
    for innerkey,val in skey.items():
        print(innerkey)
        temptimes = {}
        for outerkey in skey[innerkey]:
            print(outerkey)
            temp_ind = int(outerkey.rsplit('_')[-1]) - 1
            #print(temp_ind)
            temptimes[outerkey] = SWRtimes[temp_ind][:,:]
            # 1st dim = # of swr events, 
            # 2nd dim: 0 = SWR onset time, 1 = SWR peak amp time, 2 = SWR end time (0.5 of threshold value)
        newSWRtimes[innerkey] = temptimes

    return newSWRtimes
######################################################################################################################
def resample(x, factor, kind='linear'):
    '''
    
    '''
    from scipy.interpolate import interp1d 
    n = int(np.ceil(x.size / factor))
    f = interp1d(np.linspace(0, 1, x.size), x, kind)
    
    return f(np.linspace(0, 1, n))
######################################################################################################################
def highres(y, kind='linear', factor=32):
    '''
    Interpolate data onto a higher resolution grid by a factor of *res*

    Args:
        y (1d array/list): signal to be interpolated
        kind (str): order of interpolation (see docs for scipy.interpolate.interp1d)
        factor (int): factor to increase resolution of data via linear interpolation
    
    Returns:
        shift (float): offset between target and reference signal 
    '''
    from scipy.interpolate import interp1d
    y = np.array(y)
    x = np.arange(0, y.shape[0])
    f = interp1d(x, y,kind=kind)
    xnew = np.linspace(0, x.shape[0]-1, x.shape[0]*factor)
    ynew = f(xnew)
    
    return ynew
#####################################################################################################################
def get_ds_outside_swr(targDat,refDat):
    '''
    When passed here, both targ and ref are in 1250Hz
    No offset for pre and post ripple is defined here yet.
    '''
    odata = []
    pre_ofs = int(20 * 1.25)
    post_ofs = int(20 * 1.25)
    for each_ds in targDat:
        is_between = np.zeros((refDat.shape))
        for indx,(lb,ub) in enumerate(zip(refDat[:,0],refDat[:,2])):
            is_between[indx] = each_ds in range(lb - pre_ofs, ub + post_ofs)
            #print(indx,each_ds,lb,ub,is_between[indx])
        # checksum for is_between     
        if np.sum(is_between) == 0:
            odata.append(each_ds)

    return np.array(odata)
####################################################################################################################
def generate_ds_only_times(refDict,targDict,tconv,min_ds=50):
    '''
    Note that DS times are in 20k, SWR times are in 1250 
    ...so that the ref and targ can only be this way around
    '''
    odata = {}
    for ikey,val in refDict.items():
        print(ikey)
        tg = {}
        for okey,val2 in refDict[ikey].items():
            print(okey)
            refDat = np.array(val2)
            try:
                targDat = np.array(targDict[ikey][okey] * tconv['spk_lfp'],dtype=int) # converts DS times to 1250Hz
            except TypeError:
                targDat = np.array([])
            print(refDat.shape,targDat.shape)
            if targDat.shape[0] > min_ds:
                print('there are more than {0:} DS'.format(min_ds)) # fix this
                tempDat = get_ds_outside_swr(targDat,refDat)  
                print(ikey,okey,targDat.shape,tempDat.shape)
                tg[okey] = np.array(tempDat * tconv['lfp_spk'],dtype=int) # converts DS times back to 20,000 Hz
        odata[ikey] = tg

    return odata
####################################################################################################################
def save_new_times(opath,intv_times,fext=['.ds_','_no_swr']):
    '''
    Saves e.g. a text file of DS times in 20,000 Hz resolution
    '''
    for ikey,val in intv_times.items():
        intv_ext = fext[0] + ikey + fext[1]
        for okey,val in intv_times[ikey].items():
            fname = okey
            print(opath + '/' + fname)
            odata = intv_times[ikey][okey]
            sbf.write_to_text(opath,fname,odata,intv_ext)
####################################################################################################################

