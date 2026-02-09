##############################################################
## Import libraries
##############################################################
import pandas as pd
import numpy as np
import sys as sys
import math as math
import os as os
import time as time
import vBaseFunctions3 as vbf
import matplotlib.pyplot as plt
import scipy as scipy
import scipy.stats as stats
import scipy.ndimage as spim
import pyentropy as pye
import fnmatch
import smBaseFunctions3 as sbf
from my_mpl_defaults import *
###############################################################################################
def get_mouseID_and_order_nesw(database,mpath):
    '''
    
    '''
    order_nesw = {}
    mouseID = []

    for dinx,eachdir in enumerate(database):
        eachdir = mpath + eachdir
        os.chdir(eachdir)
        os.getcwd()
        # get basename from database
        ipath = eachdir
        bsnm = ipath.rsplit('/', 1)[-1]
        baseblock = ipath + '/' + bsnm
        print(dinx,baseblock)
        # load order of novel object (n2 to n5)
        orderFile = baseblock + '.order'
        order_nesw[bsnm] = np.loadtxt(orderFile,dtype='str')
        mouseID.append(bsnm)
        
    return mouseID, order_nesw
###############################################################################################
def get_stimlist(group_type,mouseID,day_type='obj2'):
    '''
    
    '''
    stimlist = {}
    if group_type == 'archT':
        stimlist['n2'] = [1,0,0,1,1,0,0,1]
        stimlist['n3'] = [0,1,1,0,0,1,1,0]
        stimlist['n4'] = [1,0,0,1,1,0,0,1]
        stimlist['n5'] = [0,1,1,0,0,1,1,0]
    elif group_type == 'gfp':
        stimlist['n2'] = [1,0,0,1,0,0,1,0,1,1]
        stimlist['n3'] = [0,1,1,0,1,1,0,1,0,0]
        stimlist['n4'] = [1,0,0,1,0,0,1,0,1,1]
        stimlist['n5'] = [0,1,1,0,1,1,0,1,0,0]
    elif (group_type == 'aged' and day_type == 'obj2'):
        stimlist['n2'] = [1,0,1,0,0,1,0,1]
        stimlist['n3'] = [0,1,0,1,1,0,1,0]
        stimlist['n4'] = [1,0,1,0,0,1,0,1]
        stimlist['n5'] = [0,1,0,1,1,0,1,0]
    elif (group_type == 'aged' and day_type == 'obj3'):
        stimlist['n2'] = [1,0,1,0,0,1,1]
        stimlist['n3'] = [0,1,0,1,1,0,0]
        stimlist['n4'] = [1,0,1,0,0,1,1]
        stimlist['n5'] = [0,1,0,1,1,0,0]
    elif (group_type == 'grm' and day_type == 'obj3'):
        stimlist['n2'] = [0,1,0,1,0,0,0,1,0,0,1,0,0,1,1,0,0,0,1,0,0]
        stimlist['n3'] = [1,0,1,0,0,0,0,1,0,0,1,0,0,1,1,0,0,0,1,0,0]
        stimlist['n4'] = [0,1,0,1,0,0,0,]
        stimlist['n5'] = [1,0,1,0,0,0,0,]
    elif (group_type == 'grm' and day_type == 'obj4'):
        stimlist['n2'] = [1,0,1,0,0,1,1,0,1,0]
        stimlist['n3'] = [1,0,1,0,0,1,1,0,1,0]
        stimlist['n4'] = []
        stimlist['n5'] = []
    elif (group_type == 'grm' and day_type == 'obj5'):
        stimlist['n2'] = [0,1,0,1,0,0,0]
        stimlist['n3'] = [1,0,1,0,0,0,0]
        stimlist['n4'] = [0,1,0,1,0,0,0]
        stimlist['n5'] = [1,0,1,0,0,0,0]
    elif (group_type == 'grm' and day_type == 'obj_msm32'):
        stimlist['n2'] = [0,1,1,0]
        stimlist['n3'] = [0,1,1,0]
        stimlist['n4'] = []
        stimlist['n5'] = []
    else:
        nMice = len(mouseID)
        stimlist['n2'] = [0] * nMice
        stimlist['n3'] = [0] * nMice
        stimlist['n4'] = [0] * nMice
        stimlist['n5'] = [0] * nMice

    print('This stimlist is hard coded - you must double check')

    return stimlist
###############################################################################################
def get_order_nesw(database,mpath='/mnt/smchugh2'):
    '''
    
    '''
    order_nesw = {}
    for dindx,eachdir in enumerate(database):
        eachdir = mpath + eachdir
        os.chdir(eachdir)
        # get basename from database
        ipath = eachdir
        bsnm = ipath.rsplit('/', 1)[-1]
        baseblock = ipath + '/' + bsnm
        # load order of novel object (n2 to n5)
        order_file = baseblock + '.order'
        order_nesw[bsnm] = np.loadtxt(order_file,dtype='str')
    
    return order_nesw
###############################################################################################
def nov_trial_matrix(mouseID,order_nesw,ntrials=4,debug=False):
    '''
    
    '''
    compass_to_col = {'n':0,'e':1,'s':2,'w':3}
    tnov = np.empty([ntrials,len(mouseID)])
    for trial in range(ntrials):
        for mindx,val in enumerate(mouseID):
            tnov[trial,mindx] = compass_to_col[order_nesw[val][trial]]
            if debug:
                print(trial,mindx,val,compass_to_col[order_nesw[val][trial]],order_nesw[val])

    return tnov
###############################################################################################
def generate_ordlist(mouseID,sessList,order_nesw,ntrials=4,debug=False):
    '''
    
    '''
    tnov = nov_trial_matrix(mouseID,order_nesw,ntrials=ntrials,debug=debug)
    ordlist = {}
    
    for sindx,sess in enumerate(sessList):
        ordlist[sess] = tnov[sindx,:]
    
    return ordlist
###############################################################################################
def nov_percent(iDict,sessList,ordlist,nObj=4):
    '''
    
    '''
    nMice = len(iDict[sessList[0]])
    odata = {}    
    for sindx,sess in enumerate(sessList):
        novTemp = []
        famTemp = []
        for mindx in range(nMice):
            famIndx = list(range(nObj))
            idata = iDict[sess][mindx]
            novIndx = int(ordlist[sess][mindx])
            famIndx.pop(novIndx)
            novTemp.append( ( idata[novIndx] / np.sum(idata) ) * 100)
            famTemp.append( ( (np.nanmean(idata[famIndx])) / np.sum(idata) ) * 100)
	
        odata[sess] = [novTemp,famTemp]

    return odata
#############################################################################################
def nov_percent_2levels(iDict,sessList,ordlist,nObj=4):
    '''
    
    '''
    nMice = len(iDict[sessList[0]])
    odata = {}    
    for sindx,sess in enumerate(sessList):
        novTemp = []
        famTemp = []
        tempDict = {}
        for mindx in range(nMice):
            famIndx = list(range(nObj))
            idata = iDict[sess][mindx]
            novIndx = int(ordlist[sess][mindx])
            famIndx.pop(novIndx)
            novTemp.append( ( idata[novIndx] / np.sum(idata) ) * 100)
            famTemp.append( ( (np.nanmean(idata[famIndx])) / np.sum(idata) ) * 100)
        tempDict['fam'] = famTemp
        tempDict['nov'] = novTemp
        odata[sess] = tempDict

    return odata
###############################################################################################
def get_nov_data(iDict,ikeys,stimlist,col_ind,stim=False):
    '''

    '''
    odata = []
    templist = {}
    for sindx,sess in enumerate(ikeys):
        if stim:
            templist[sess] = stimlist[sess]
        else:
            templist[sess] = [ (1-x) for x in stimlist[sess] ]
        datList = np.array(iDict[sess][col_ind])
        indList = np.where(templist[sess])
        odata.extend(datList[indList])
        
    return odata
###########################################################################################################################
def generate_dabest_data(percent_obj,group_type,sessList,stimlist):
    '''
    
    '''
    tempdict = {}

    if group_type == 'archT' or group_type == 'gfp':
        all_params = [ [0,0,'nov_nostim'], [1,0,'fam_nostim'] , [0,1,'nov_stim'] , [1,1,'fam_stim'] ]
        for indx, vals in enumerate(all_params):
            print(vals)
            tempdict[vals[2]] = get_nov_data(percent_obj[group_type],sessList,stimlist,col_ind=vals[0],stim=vals[1])
        tempdict['nov'] = tempdict['nov_nostim'] + tempdict['nov_stim']
        tempdict['fam'] = tempdict['fam_nostim'] + tempdict['fam_stim']

    if group_type == 'chr2':
        all_params = [ [0,0,'nov'], [1,0,'fam'] ]
        for indx, vals in enumerate(all_params):
            print(vals)
            tempdict[vals[2]] = get_nov_data(percent_obj[group_type],sessList,stimlist,col_ind=vals[0],stim=vals[1])
    
    return tempdict
############################################################################################################################
def get_objPos_new(mazedim):
    '''
    
    '''
    objPos = {}
    ######################################################
    print(mazedim)
    xmin,xmax,ymin,ymax = mazedim
    xmid = xmin + ((xmax - xmin) / 2)
    ymid = ymin + ((ymax - ymin) / 2)
    ######################################################
    objPos['n'] = [xmid+2.5,ymin+40]
    objPos['e'] = [xmax-30,ymid+10]
    objPos['s'] = [xmid+2.5,ymax-40]
    objPos['w'] = [xmin+40,ymid+10]
    
    return objPos   
####################################################################################################
def gen_hist2d(xdat,ydat,xedg,yedg,gdim):
    '''

    '''
    H, xe, ye = np.histogram2d(xdat, ydat, bins=(xedg, yedg))
    H = H.T
    mid = int((gdim - 1) / 2)
    edge = int(gdim-1)
    print(mid,edge)
    nesw_cnt = [H[0,mid],H[mid,edge],H[edge,mid],H[mid,0]] # note: NESW order

    return H, nesw_cnt
####################################################################################################
def get_nesw(track,mazedim,trialT=300,grid=[3,3],gridoffset=[5,5],fps=39.025,debug=False):
    '''
    
    '''
    
    objPos = get_objPos_new(mazedim) # import object locations
    
    x_divs,y_divs = grid[0],grid[1] # divide arena into e.g. 3 x 3 grid

    gridxOffset,gridyOffset = gridoffset[0],gridoffset[1] # add a few pixels to the edges

    # g stands for grid
    gxmin = objPos['w'][0]-gridxOffset
    gxmax = objPos['e'][0]+gridxOffset
    gymin = objPos['n'][1]-gridyOffset
    gymax = objPos['s'][1]+gridyOffset

    pix_x = (gxmax - gxmin)/(x_divs*1.0)
    pix_y = (gymax - gymin)/(y_divs*1.0)

    xedges=[gxmin]
    yedges=[gymin]

    for x in range(x_divs):
        xedges.append(np.round(gxmin+(x+1)*pix_x))  #gxmin+(x+1)*pix_x   
    for y in range(y_divs):
        yedges.append(np.round(gymin+(y+1)*pix_y))  #gxmin+(x+1)*pix_x    

    tind = int(trialT * fps)
    xvalid = track['x'].values[:tind]
    yvalid = track['y'].values[:tind]
    
    hist_all,hist_nesw = gen_hist2d(xvalid,yvalid,xedges,yedges,gdim=grid[0])
    
    return hist_all,hist_nesw
####################################################################################################
def plot_track(bsnm,mazedim,track,sess,OccMap,nbins=30,smooth=1.2,cmap=None,pprint=False):
    '''
    
    '''
    objPos = get_objPos_new(mazedim)
    if pprint:
        print([round(x,1) for x in mazedim])
        for key,value in objPos.items():
            print(key,np.round(value,2))
    
    rr,cc = 1,2
    wcm,hcm = 4*cc,4*rr
    fig,ax = plt.subplots(rr,cc,figsize=sbf.cm2inch(wcm,hcm),\
                     gridspec_kw = {'wspace':0.1,'hspace':0}
                         )
    font = {'family': 'serif',
        'color':  'red',
        'weight': 'normal',
        'size': 16,
        }
    #########################################################################################
    ax[0].plot(track['x'],track['y'],color='grey',linewidth=1.0)
    ax[0].set_xlim(mazedim[0]-5,mazedim[1]+5)
    ax[0].set_ylim(mazedim[3]+5,mazedim[2]-5,) # note this is backwards but should now match!
    #ax[0].set_axis_off()
    subtitle = sess.replace(" ","") + ': ' #+ str(np.round(np.nanmean(speed[skey]),1))
    ax[0].set_title(subtitle,y=1,fontsize=8)
    ##########################################################################################
    ax[0].text(objPos['n'][0],objPos['n'][1], 'N', fontdict=font, ha='center', va='center')
    ax[0].text(objPos['e'][0],objPos['e'][1], 'E', fontdict=font, ha='center', va='center')
    ax[0].text(objPos['s'][0],objPos['s'][1], 'S', fontdict=font, ha='center', va='center')
    ax[0].text(objPos['w'][0],objPos['w'][1], 'W', fontdict=font, ha='center', va='center')              
    ##########################################################################################
    value = 0
    if cmap is None:
        cmap = plt.get_cmap('jet')
    cmap.set_bad(color='white') 
    hmin,hmax = 0,8
    ######################################################################################
    if smooth is not None:
        smoothMap = spim.filters.gaussian_filter(OccMap,smooth,mode='constant')
        masked_data1 = np.ma.masked_where(OccMap == value, smoothMap)
    else:
        masked_data1 = np.ma.masked_where(OccMap == value, OccMap)
    ax[1].imshow(masked_data1.T,cmap=cmap,interpolation='Nearest')
    ax[1].set_xlim(0,nbins)
    ax[1].set_ylim(nbins,0) # note this is backwards but should now match!
    ax[1].set_axis_off()
    fig.suptitle(bsnm,y=1.0,fontsize=10)
    ###############################################################################################
    plt.show()
###############################################################################################

##################################################################################################
def plot_track_simple(track,select_indx,objLoc,pos,mazedim,Xedges,Yedges,axes_off=True):
    '''
    
    '''
    xdata = track['x'][:select_indx]
    ydata = track['y'][:select_indx]
    colList = ['yellow','orange','k','r']
    ##############################################################################################
    rr,cc = 1,1
    wcm = 8 # should be 2
    hcm = 8 # should be 2
    fig, ax = plt.subplots(rr,cc,figsize = sbf.cm2inch(wcm,hcm),
                           gridspec_kw = {'wspace':0,'hspace':0})
    
    ax.plot(xdata,ydata)

    for lindx,loc in enumerate(pos):
        ax.plot(objLoc[loc]['x'],objLoc[loc]['y'],color=colList[lindx])
        ax.text(objLoc[loc]['x'],objLoc[loc]['y'],loc,ha='center', va='center')

    ax.set_xlim(mazedim[0]-5,mazedim[1]+5)
    ax.set_ylim(mazedim[3]+5,mazedim[2]-5)
    for v in Xedges:
        ax.axvline(v,color='grey')
    for h in Yedges:
        ax.axhline(h,color='grey')
    if axes_off:
        ax.set_axis_off()

    return fig,ax
##############################################################################################
def add_obj_pos(ax,objPos,fcol='k',fsize=16):
    '''
    
    '''
    font = {'family': 'sans-serif',
            'color':  fcol,
            'weight': 'bold',
            'size': fsize,
            }
    props = dict(facecolor='white', alpha=0.5, linewidth=0)
    for key,val in objPos.items():
        ax.text(objPos[key][0],objPos[key][1], key.upper(), fontdict=font, ha='center', va='center', bbox=props)

    return ax
##############################################################################################
def get_mazedim(bsnm,from_file=False,mainpath='/mnt/smchugh2/lfpd4/SF/'):
    '''

    '''
    baseblock = mainpath + bsnm + '/' + bsnm
    if from_file:
        try:
            mazedim = np.loadtxt(baseblock + '.mazedim',dtype='int')
        except OSError:
            print('Cant find file so using default dimensions')
            mazedim = [265,640,130,530]
    else: 
        if 'msm19' in bsnm:
            mazedim = [230,615,90,465]
        elif 'msm22' in bsnm:
            if (bsnm == 'msm22-210308') or (bsnm == 'msm22-210309'):
                mazedim = [205,635,100,500]
            else:
                mazedim =  [180,610,55,455]
        elif 'msm23' in bsnm:
            mazedim = [200,640,80,480]
        else:
            mazedim = [265,640,130,530]
            
    return mazedim
###############################################################################################
def bin_compass_points(track,selectPos,Xedges,Yedges,maxSamp=-1,pprint=False):
    '''
    
    
    '''
    # Change this so works with any n x n grid
    north = [ Xedges[1],Xedges[2],Yedges[0],Yedges[1] ]
    east = [ Xedges[2],Xedges[3],Yedges[1],Yedges[2] ]
    south = [ Xedges[1],Xedges[2],Yedges[2],Yedges[3] ]
    west = [ Xedges[0],Xedges[1],Yedges[1],Yedges[2] ]
    
    compassDict = {'north':north,'east':east,'south':south,'west':west}
    sq = compassDict[selectPos]
    if pprint:
        print('{0:} edges are: {1:}'.format(selectPos,sq))
        print('Caution: edges are hard-coded for 3 x 3 grid')

    odata = {}
    tempX = []
    tempY = []
    tempInds = []
    outInds = []

    # Updated Nov 2022
    for sampIndx,(sampleX,sampleY) in enumerate(zip(track['x'][:maxSamp],track['y'][:maxSamp])):
        if (sq[0] <= sampleX <= sq[1]) and (sq[2] <= sampleY <= sq[3]):
            tempX.append(int(sampleX))
            tempY.append(int(sampleY))
            tempInds.append(sampIndx)
        else:
            outInds.append(sampIndx)

    odata['x'] = np.array(tempX)
    odata['y'] = np.array(tempY)
    odata['inds'] = np.array(tempInds)
    odata['outinds'] = np.array(outInds)
    
    return odata
################################################################################################
def generate_start_end_pulses(iDict,tconv=(20000/39.0625)):
    '''
    iDict = objLoc[loc]
    '''
        
    start_mask = np.concatenate(([True],np.diff(iDict['inds'])>1))
    end_mask = np.diff(iDict['outinds']) > 1
    
    start_inds = iDict['inds'][start_mask]
    end_inds = iDict['outinds'][1:][end_mask]
    
    if (start_inds.shape[0] > end_inds.shape[0]) and (iDict['inds'][-1] > iDict['outinds'][-1]):
        end_inds = np.concatenate((end_inds,[iDict['inds'][-1]]))
        
    otimes = np.empty((start_inds.shape[0],2)).astype(int)
    
    otimes[:,0] = start_inds * tconv
    otimes[:,1] = end_inds * tconv
    
    return otimes
################################################################################################
def genEdges2d(mazedim,nbins=3):
    '''
    
    '''
    nEdges = nbins+1
    Xedges = [int(x) for x in np.linspace(mazedim[0],mazedim[1],nEdges)]
    Yedges = [int(x) for x in np.linspace(mazedim[2],mazedim[3],nEdges)]

    return Xedges,Yedges
################################################################################################
def get_object_sq_coords(Xedges,Yedges,selectPos,pprint=False):
    '''
    
    '''
    n = len(Xedges)-1
    xmid = int((len(Xedges)/2)-1)
    ymid = int((len(Yedges)/2)-1)

    north = [ Xedges[xmid],Xedges[xmid+1],Yedges[0],Yedges[1] ]
    east = [ Xedges[n-1],Xedges[n],Yedges[ymid],Yedges[ymid+1] ]
    south = [ Xedges[xmid],Xedges[xmid+1],Yedges[n-1],Yedges[n] ]
    west = [ Xedges[0],Xedges[1],Yedges[ymid],Yedges[ymid+1] ]

    compassDict = {'n':north,'e':east,'s':south,'w':west}
    sq = compassDict[selectPos]
    if pprint:
        print('{0:} edges are: {1:}'.format(selectPos,sq))
    
    return sq
############################################################################################
def get_masked_trk(idata,sq):
    '''
    
    '''
    odata = {}
    tempX = []
    tempY = []
    tempInds = []
    try:
        xdat = idata[0,:]
        ydat = idata[1,:]
    except:
        xdat = idata['x']
        ydat = idata['y']
    
    # This would be much more efficient using np.argwhere!! But this works
    for sampIndx,(sampleX,sampleY) in enumerate(zip(xdat,ydat)):
        if  sq[0] <= sampleX <= sq[1] and sq[2] <= sampleY <= sq[3]:
            tempX.append(int(sampleX))
            tempY.append(int(sampleY))
            tempInds.append(sampIndx)

    odata['x'] = tempX
    odata['y'] = tempY
    odata['inds'] = tempInds
    
    return odata
#####################################################################################################
def plot_trk(idata,mazedim,Xedges,Yedges,lw=0.5,axes_off=True):
    '''
    
    '''
    xdata = idata[0,:]
    ydata = idata[1,:]
    colList = ['yellow','orange','k','r']
    ##############################################################################################
    rr,cc = 1,1
    wcm = 8 # should be 2
    hcm = 8 # should be 2
    fig, ax = plt.subplots(rr,cc,figsize = sbf.cm2inch(wcm,hcm),\
                           gridspec_kw = {'wspace':0,'hspace':0})
    
    ax.plot(xdata,ydata,color=gray2,marker='.',markersize=3,alpha=.3)

    ax.set_xlim(mazedim[0]-5,mazedim[1]+5)
    ax.set_ylim(mazedim[3]+5,mazedim[2]-5)
    for v in Xedges:
        ax.axvline(v,color='b',linewidth=lw)
    for h in Yedges:
        ax.axhline(h,color='b',linewidth=lw)
    if axes_off:
        ax.set_axis_off()
        
    return fig,ax
##############################################################################################
def animate_tracking(track,
                     mazedim,
                     Xedges,
                     Yedges,
                     order_nesw,
                     sess='',
                     figsize=(10,10),
                     interval=5,
                     axes_off=True,
                     remove_nan=False):
    '''
    figsize = (15,15)
    fig,ax,anim = animate_tracking(track,
                                   mazedim,
                                   Xedges,
                                   Yedges,
                                   order_nesw,
                                   sess=sess,
                                   figsize=figsize,
                                   interval=5,
                                   axes_off=True,
                                   remove_nan=False)
    '''
    
    from matplotlib.animation import FuncAnimation
    if remove_nan:
        track['x'][np.isnan(track['x'])] = -1
        track['y'][np.isnan(track['y'])] = -1

        x = track['x'][track['x']>0].values.reshape(-1, 1)
        y = track['y'][track['y']>0].values.reshape(-1, 1)
    else:
        x = track['x'].values.reshape(-1, 1)
        y = track['y'].values.reshape(-1, 1)
    
    frames = x.shape[0]
    print(frames)
    
    try:
        print(sess,order_nesw)
    except:
        print('sess or order_nesw not loaded')

    fwd,fht = figsize
    fig,ax = plt.subplots(1,1,figsize=sbf.cm2inch(fwd,fht))

    ax.set_xlim(mazedim[0]-5,mazedim[1]+5)
    ax.set_ylim(mazedim[3]+5,mazedim[2]-5)

    for v in Xedges:
        ax.axvline(v,color='grey')
    for h in Yedges:
        ax.axhline(h,color='grey')
    if axes_off:
        ax.set_axis_off()
        
    time_text = ax.text(0.05, 0.95,'',horizontalalignment='left',verticalalignment='top',transform=ax.transAxes)

    objPos = get_objPos_new(mazedim)
    ax = add_obj_pos(ax,objPos)
    line1, = ax.plot([], [], color='r', lw=3, marker='o')

    def animate(i):
        if i-10 < 0:
            past_frames = 0
        else:
            past_frames = i-10
        line1.set_data(x[past_frames:i, 0], y[past_frames:i, 0])
        time_text.set_text('time = %.1d' % i)
        return line1,[time_text]

    anim = FuncAnimation(fig,animate,frames=frames,interval=interval,repeat=False) #,save_count=frames)
    
    return fig,ax,anim
########################################################################################################################################
def generate_target_vector(idata,subset):
    '''

    '''
    odata = np.zeros((idata.shape[1]))
    obj_loc = ['n','e','s','w']
    for lindx,loc in enumerate(obj_loc):
        inds = subset[loc]['inds']
        odata[inds] = lindx + 1

    return odata
#########################################################################################################
def generate_subset(idata,objPos,Xedges,Yedges):
    '''
    
    '''
    sq = {}
    subset = {}
    
    for key, val in objPos.items():
        sq[key] = get_object_sq_coords(Xedges,Yedges,key)
        subset[key] = get_masked_trk(idata,sq[key])
        
    return sq, subset
##########################################################################################################
def check_session(sessions,sess_to_check):
    '''
    
    '''
    if sess_to_check not in sessions:
            odata = sess_to_check + 'stim'
    else:
        odata = sess_to_check
            
    return odata
##########################################################################################################
def log_reg(X,y,nValidations=100,test_size=0.2):
    '''
    
    '''
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix
    
    nObj = len(np.unique(y))
    confusion_ = np.zeros((nObj,nObj))
    scores = []
    #try:
    for _ in range(nValidations): # number of monte-carlo cross validations
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        #print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
        a = LogisticRegression().fit(X_train, y_train)
        y_pred = a.predict(X_test)
        #print(y_pred,y_test)
        scores.append(a.score(X_test,y_test))

        # add results to confusion matrix
        for i in range(len(y_pred)):
            confusion_[int(y_pred[i]-1),int(y_test[i]-1)] += 1

    # normalise confusion matrix
    confusion_norm = confusion_ / np.sum(confusion_,0)
    #except ValueError:
    #    print('Not enough theta cycles for the session')
    
    return scores, confusion_norm
######################################################################################################
