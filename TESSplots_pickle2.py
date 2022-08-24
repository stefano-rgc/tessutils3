#!/usr/bin/env python
import time, re, functools, pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import gridspec
import lightkurve as lk 
import peakutils
import matplotlib.cbook
from astropy.io import fits
import astropy
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
# Suppress some MatPlotLib warnings
import warnings
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

def scalesymbols(mags, min_mag, max_mag, scale=120):
    """
        Author: 
            Timothy
    
        Purpose:
            A simple routine to determine the scatter marker sizes, based on the TESS magnitudes
        
        Parameters:
            mags (numpy array of floats): the TESS magnitudes of the stars
            min_mag (float): the smallest magnitude to be considered for the scaling
            max_mag (float): the largest magnitude to be considered for the scaling
        Returns:
            sizes (numpy array of floats): the marker sizes
    """
    
    sizes = scale * (1.1*max_mag - mags) / (1.1*max_mag - min_mag)
    
    return sizes

def overplot_mask(ax, mask, ec='r', fc='none', lw=1.5, alpha=1):
    """
        Author: 
            Timothy
    
        Purpose:
            (Over)plotting the mask (on a selected frame in the axis "ax").
    """
    
    for i in range(len(mask[:,0])):
        for j in range(len(mask[0,:])):
            if mask[i, j]:
                ax.add_patch(mpl.patches.Rectangle((j-0.5, i-0.5), 1, 1, edgecolor=ec, facecolor=fc,linewidth=lw,alpha=alpha))

def get_from_tbhdu(tbhdu, varname):
    if isinstance(varname,str):
        size = tbhdu.header[f'N_{varname}']
        var = tbhdu.data[f'{varname}'][:size]
        return var
    elif isinstance(varname,list):   
        var = [get_from_tbhdu(tbhdu, var) for var in varname]
        return var
    else:
        raise TypeError

def chunks(lst, n):
    """
    Source
        https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
    
    Purpose
        Yield successive n-sized chunks from lst.
    """
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
        
# def Normalize_lc(flux):
#     '''Function applied to light curves of individual TESS sectors before stitching them'''
#     # flux -= np.median(flux)
#     median = np.median(flux)
#     flux = (flux-median)/median
#     return flux

def Normalize_lc(flux):
    '''Function applied to light curves of individual TESS sectors before stitching them'''
    try:
        # flux -= np.median(flux)
        median = np.median(flux)
        flux = (flux-median)/median
        return flux
    except Exception:
        lc = flux
        flux = lc.flux.value
        time = lc.time.value
        median = np.median(flux)
        flux = (flux-median)/median
        return lk.LightCurve(time=time, flux=flux)
    
def plot_images(fig,grid,FITS,TitleFontSize,AnnFontSize):
    
    # Get the names  of the extension in the FITS file
    ExtNames = [FITS[i].header['EXTNAME'] for i in range(1,len(FITS))]
    # Get the TESS sector
    sector = FITS['OLD.PRIMARY'].header['SECTOR']
    
    # Create the tetra axes for the images
    tetragrid = grid.subgridspec(2, 2, hspace=0, wspace=0)

    # Plot median of TESS images
    image = FITS['PRIMARY'].data
    ax_im1 = fig.add_subplot(tetragrid[0,0])
    im = ax_im1.imshow(np.log10(image), origin='lower', cmap = plt.cm.YlGnBu_r)
    ax_im1.set_title(f'Median Sec {sector}',size=TitleFontSize, pad=2)
    ax_im1.axes.get_yaxis().set_visible(False)
    ax_im1.axes.get_xaxis().set_visible(False)
    ax_im1.set_zorder(2)
    
    # Plot the aperture and background masks
    if 'MASKAP' in ExtNames:
        # Load the masks
        apmask = FITS['MASKAP'].data.astype(bool)
        bkgmask = FITS['MASKBKG'].data.astype(bool)
        ax_im3 = fig.add_subplot(tetragrid[1,0])
        im = ax_im3.imshow(np.log10(image), origin='lower', cmap = plt.cm.YlGnBu_r)
        overplot_mask(ax_im3,bkgmask,ec='w',lw=0.1, fc='w', alpha=0.3)
        overplot_mask(ax_im3,bkgmask,ec='w',lw=0.1, fc='none', alpha=1.0)
        overplot_mask(ax_im3,apmask,ec='r',lw=0.1, fc='r', alpha=0.3)
        overplot_mask(ax_im3,apmask,ec='r',lw=0.1, fc='none', alpha=1.0)
        ax_im3.axes.get_yaxis().set_visible(False)
        ax_im3.axes.get_xaxis().set_visible(False)
        ax_im3.set_title('Masks',size=TitleFontSize, pad=2)
        ax_im3.set_zorder(2)
    # If no masks extension in FITS
    else:
        if not 'TABLE' in ExtNames:
            return False
        

    # Plot neighbour stars
    if 'TABLE' in ExtNames:
        # Load the table data
        tbhdu = FITS['TABLE']
        if len(tbhdu.columns.names)==0:
            return False
        # Load coordinate data
        wcs = astropy.wcs.WCS(header=FITS['OLD.PIXELS'].header)
        target_ra = FITS['OLD.PRIMARY'].header['ra_obj']
        target_dec = FITS['OLD.PRIMARY'].header['dec_obj']
        used_nb_ra, used_nb_dec, used_nb_tmag, target_tmag = get_from_tbhdu(tbhdu,['used_nb_ra','used_nb_dec','used_nb_tmag','target_tmag'])
        target_tmag = target_tmag.item()
        # To AstroPy SkyCoord
        target_coord = SkyCoord(target_ra, target_dec, unit = "deg")
        nb_coords = [ SkyCoord(ra, dec, unit = "deg") for ra,dec in zip(used_nb_ra,used_nb_dec)]
        # To pixels
        target_coord_pix = target_coord.to_pixel(wcs,origin=0)
        nb_coord_pix = np.array([ c.to_pixel(wcs,origin=0) for c in nb_coords ])

        # Plot neighbour stars
        ax_im4 = fig.add_subplot(tetragrid[1,1])
        im = ax_im4.imshow(np.log10(image), origin='lower', cmap = plt.cm.YlGnBu_r)
        ax_im4.set_title('Neighbours',size=TitleFontSize, pad=2)
        ax_im4.scatter(target_coord_pix[0],target_coord_pix[1],s=scalesymbols(target_tmag,np.amin(target_tmag), np.amax(target_tmag), scale=15.),c='r',edgecolors='k', linewidth=0.2, zorder=5, label=f'{target_tmag:.1f}')
        if nb_coord_pix.size > 0:
            ax_im4.scatter(nb_coord_pix[:,0],nb_coord_pix[:,1],s=scalesymbols(used_nb_tmag,np.amin(used_nb_tmag), np.amax(used_nb_tmag), scale=15.),c='w',edgecolors='k', linewidth=0.2, zorder=5)

        # Legend
        if nb_coord_pix.size > 0:
            ax_im4.scatter(-1, -1, s=scalesymbols(10.*np.ones(1),np.amin(used_nb_tmag), np.amax(used_nb_tmag), scale=15.), c='w', edgecolors='k', linewidth=0.2, label='10')
            ax_im4.scatter(-1, -1, s=scalesymbols(12.*np.ones(1),np.amin(used_nb_tmag), np.amax(used_nb_tmag), scale=15.), c='w', edgecolors='k', linewidth=0.2, label='12')
            ax_im4.scatter(-1, -1, s=scalesymbols(14.*np.ones(1),np.amin(used_nb_tmag), np.amax(used_nb_tmag), scale=15.), c='w', edgecolors='k', linewidth=0.2, label='14')
        ax_im4.set_xlim(0,image.shape[0]-1)
        ax_im4.set_ylim(0,image.shape[1]-1)
        lgnd = ax_im4.legend(ncol=4, loc=(-0.9,-0.3), markerscale=1,frameon=False, columnspacing=0, handletextpad=0.1)
        ax_im4.axes.get_yaxis().set_visible(False)
        ax_im4.axes.get_xaxis().set_visible(False)
        ax_im4.set_zorder(2)
        
    # If no table extension in FITS
    else:
        return False

    # Plot neighbour stars
    if 'FITIMAGE' in ExtNames:
        # Load fitted image
        fitimage = FITS['FITIMAGE'].data
        ax_im2 = fig.add_subplot(tetragrid[0,1])
        im = ax_im2.imshow(np.log10(fitimage), origin='lower', cmap = plt.cm.YlGnBu_r)
        ax_im2.axes.get_yaxis().set_visible(False)
        ax_im2.axes.get_xaxis().set_visible(False)
        ax_im2.set_title('Fit',size=TitleFontSize, pad=2)
        ax_im2.set_zorder(2)

        # Annotations
        C, S, apthreshold = get_from_tbhdu(tbhdu,['ap_contamination','frac_bkg_change','ap_threshold'])
        C, S, apthreshold = 100*C[0], 100*S[0], apthreshold[0]
        text = f'C={C:.1f}% S={S:.1f}% thr={apthreshold:.1f}'
        ax_im4.annotate(text, (0.0,-0.5), xycoords='axes fraction', fontsize=AnnFontSize, ha='center', va='bottom')
        
        # Only return True if the plot reaches a satisfactory fitted image
        return True
    else:
        return False


def plot_images_pickle(fig,grid,result,TitleFontSize,AnnFontSize):
    
    sector = result['sector']
    
    # Create the tetra axes for the images
    tetragrid = grid.subgridspec(2, 2, hspace=0, wspace=0)

    # Plot median of TESS images
    image = result['median_image']
    ax_im1 = fig.add_subplot(tetragrid[0,0])
    im = ax_im1.imshow(np.log10(image), origin='lower', cmap = plt.cm.YlGnBu_r)
    ax_im1.set_title(f'Median Sec {sector}',size=TitleFontSize, pad=2)
    ax_im1.axes.get_yaxis().set_visible(False)
    ax_im1.axes.get_xaxis().set_visible(False)
    ax_im1.set_zorder(2)
    
    # Plot the aperture and background masks
    if not result['masks']['aperture'] is None:
        # Load the masks
        apmask = result['masks']['aperture'].astype(bool)
        bkgmask  = result['masks']['background'].astype(bool)
        ax_im3 = fig.add_subplot(tetragrid[1,0])
        im = ax_im3.imshow(np.log10(image), origin='lower', cmap = plt.cm.YlGnBu_r)
        overplot_mask(ax_im3,bkgmask,ec='w',lw=0.1, fc='w', alpha=0.3)
        overplot_mask(ax_im3,bkgmask,ec='w',lw=0.1, fc='none', alpha=1.0)
        overplot_mask(ax_im3,apmask,ec='r',lw=0.1, fc='r', alpha=0.3)
        overplot_mask(ax_im3,apmask,ec='r',lw=0.1, fc='none', alpha=1.0)
        ax_im3.axes.get_yaxis().set_visible(False)
        ax_im3.axes.get_xaxis().set_visible(False)
        ax_im3.set_title('Masks',size=TitleFontSize, pad=2)
        ax_im3.set_zorder(2)
    # If no masks
    else:
        if not result['neighbours_used'] is None:
            return False
        

    # Plot neighbour stars
    if not result['neighbours_used'] is None:
        
        # Magnitudes
        used_nb_tmag = result['neighbours_used']['mag'] 
        target_tmag = result['target']['mag']
        # To pixels
        target_coord_pix = result['target']['pix']
        nb_coord_pix = result['neighbours_used']['pix'] 

        # Plot neighbour stars
        ax_im4 = fig.add_subplot(tetragrid[1,1])
        im = ax_im4.imshow(np.log10(image), origin='lower', cmap = plt.cm.YlGnBu_r)
        ax_im4.set_title('Neighbours',size=TitleFontSize, pad=2)
        ax_im4.scatter(target_coord_pix[:,0],target_coord_pix[:,1],s=scalesymbols(target_tmag,np.amin(target_tmag), np.amax(target_tmag), scale=15.),c='r',edgecolors='k', linewidth=0.2, zorder=5, label=f'{target_tmag:.1f}')
        if nb_coord_pix.size > 0:
            ax_im4.scatter(nb_coord_pix[:,0],nb_coord_pix[:,1],s=scalesymbols(used_nb_tmag,np.amin(used_nb_tmag), np.amax(used_nb_tmag), scale=15.),c='w',edgecolors='k', linewidth=0.2, zorder=5)

        # Legend
        if nb_coord_pix.size > 0:
            ax_im4.scatter(-1, -1, s=scalesymbols(10.*np.ones(1),np.amin(used_nb_tmag), np.amax(used_nb_tmag), scale=15.), c='w', edgecolors='k', linewidth=0.2, label='10')
            ax_im4.scatter(-1, -1, s=scalesymbols(12.*np.ones(1),np.amin(used_nb_tmag), np.amax(used_nb_tmag), scale=15.), c='w', edgecolors='k', linewidth=0.2, label='12')
            ax_im4.scatter(-1, -1, s=scalesymbols(14.*np.ones(1),np.amin(used_nb_tmag), np.amax(used_nb_tmag), scale=15.), c='w', edgecolors='k', linewidth=0.2, label='14')
        ax_im4.set_xlim(0,image.shape[0]-1)
        ax_im4.set_ylim(0,image.shape[1]-1)
        lgnd = ax_im4.legend(ncol=4, loc=(-0.9,-0.3), markerscale=1,frameon=False, columnspacing=0, handletextpad=0.1)
        ax_im4.axes.get_yaxis().set_visible(False)
        ax_im4.axes.get_xaxis().set_visible(False)
        ax_im4.set_zorder(2)
        
    # If no table extension in FITS
    else:
        return False

    # Plot neighbour stars
    if not result['fit'] is None:
        # Load fitted image
        fitimage = result['fit']['fitted_image']
        ax_im2 = fig.add_subplot(tetragrid[0,1])
        im = ax_im2.imshow(np.log10(fitimage), origin='lower', cmap = plt.cm.YlGnBu_r)
        ax_im2.axes.get_yaxis().set_visible(False)
        ax_im2.axes.get_xaxis().set_visible(False)
        ax_im2.set_title('Fit',size=TitleFontSize, pad=2)
        ax_im2.set_zorder(2)

        # Annotations
        C = 100 * result['fit']['fraction_contamination_ap']
        S = 100 * result['fit']['fraction_bkg_change']
        apthreshold  = result['aperture_threshold'] 
        text = f'C={C:.1f}% S={S:.1f}% thr={apthreshold:.1f}'
        ax_im4.annotate(text, (0.0,-0.5), xycoords='axes fraction', fontsize=AnnFontSize, ha='center', va='bottom')
        
        # Only return True if the plot reaches a satisfactory fitted image
        return True
    else:
        return False


    
def triple_stacked_and_PCA_plot(fig,hgrid,FITS,tbhdu,fsize):
    
        # Create the 3 stacked axes in column 1
        stackgrid = hgrid[0].subgridspec(3, 1, hspace=0)
        
        # Plot Centroids
        col, row, time = get_from_tbhdu(tbhdu,['centroid_col','centroid_row','centroid_time'])
        ax_col = fig.add_subplot(stackgrid[0])
        ax_col.plot(time, np.sqrt(col**2+row**2), color='r', rasterized=False, label='centroid: $\sqrt{col^2+row^2}$')
#         ax_col.plot(time, col, color='r', rasterized=False, lw=0.1)
        ax_col.set_ylabel('Pix')
#         ax_col.tick_params('y', colors='r')
        ax_col.label_outer()
#         ax_row = ax_col.twinx()
#         ax_row.plot(time, row, color='g', rasterized=False, lw=0.1)
#         ax_row.tick_params('y', colors='g')
#         ax_row.set_zorder(2)
        # Overplot excluded intervals
        intervals = get_from_tbhdu(tbhdu,'excluded_day_intervals')
        t1s, t2s = intervals[::2], intervals[1::2]
        for i,(t1,t2) in enumerate(zip(t1s,t2s)):
            ax_col.axvspan(t1,t2, color='yellow', alpha=0.3, label='excluded' if i==0 else None)
        lgnd = ax_col.legend(ncol=1, loc='best', markerscale=1,frameon=False, columnspacing=0)

        # Plot raw light curve and trend
        ax_lc1 = fig.add_subplot(stackgrid[1], sharex=ax_col)
        fluxraw,timeraw = get_from_tbhdu(tbhdu,['raw_flux','raw_time'])
        ax_lc1.plot(timeraw,fluxraw, color='k', rasterized=False, zorder=3, label='Raw LC')
        ax_lc1.set_ylabel('e$^{-}$/s')
        try:
            fluxtrend,timetrend = get_from_tbhdu(tbhdu,['trend_flux','trend_time'])
            ax_lc1.scatter(timetrend,fluxtrend, color='dodgerblue',marker='*', rasterized=False, s=1, zorder=4, linewidths=0, label='PCA trend')
        # If no trend model
        except KeyError:
            ax = fig.add_subplot(stackgrid[2], sharex=ax_col)
            text = FITS['PRIMARY'].header['TAG']
            text = '\n'.join([t for t in chunks(text,47)]  )
            ax.text(0.5,0.5, text, transform=ax.transAxes, fontsize=1.1*fsize, ha='center', va='center', color='green', wrap=True)
            return False
        finally:
            lgnd = ax_lc1.legend(ncol=1, loc='best', markerscale=1,frameon=False, columnspacing=0)
            ax_lc1.label_outer()
        
        # Plot detrended light curve with sigma clipped outliers
        flux, time = get_from_tbhdu(tbhdu,['detrended_flux','detrended_time'])
        ax_lc2 = fig.add_subplot(stackgrid[2], sharex=ax_col)
        ax_lc2.plot(time,flux, color='k', rasterized=False, label='Detrended LC')
        ax_lc2.set_ylabel('e$^{-}$/s')
        ax_lc2.label_outer()
        lgnd = ax_lc2.legend(ncol=1, loc='best', markerscale=1,frameon=False, columnspacing=0)

        # Overplot excluded intervals
        intervals = get_from_tbhdu(tbhdu,'excluded_day_intervals')
        t1s = intervals[::2]
        t2s = intervals[1::2]
        for t1,t2 in zip(t1s,t2s):
            ax_lc1.axvspan(t1,t2, color='yellow', alpha=0.3)
            ax_lc2.axvspan(t1,t2, color='yellow', alpha=0.3)
        
        # Plot PCA
        time = get_from_tbhdu(tbhdu,'detrended_time')
        npc_all = tbhdu.header['N_PC_coef_all']
        npc_used = tbhdu.header['N_PC_coef_used']
        pcs = get_from_tbhdu( tbhdu, [ f'PC{i}' for i in range(npc_all-1) ] )
        ax_pca = fig.add_subplot(hgrid[1])
        for i,pc in enumerate(pcs[:npc_used-1]):
            offset = i*0.2
            ax_pca.scatter(time,pc+offset, marker='.', rasterized=True, s=1, zorder=3, linewidths=0)
        for pc in pcs[npc_used-1:]:
            i+=1
            offset = i*0.2
            ax_pca.scatter(time,pc+offset, marker='.', color='k', rasterized=True, s=1, zorder=2, linewidths=0)
        ax_pca.text(0.5, 1.0, f'Princ. comp.', transform=ax_pca.transAxes, fontsize=fsize, ha='center', va='top', color='k')
        ax_pca.axes.get_yaxis().set_visible(False)
        ax_pca.set_zorder(1)
        
        return True
    
def triple_stacked_and_PCA_plot_picke(fig,hgrid,result,fsize):
    
        # Create the 3 stacked axes in column 1
        stackgrid = hgrid[0].subgridspec(3, 1, hspace=0)
        
        # Plot Centroids
        centroid = result['centroids']['sqrt_col2_row2']
        time = result['centroids']['time']
        ax_col = fig.add_subplot(stackgrid[0])
        ax_col.plot(time, centroid , color='r', rasterized=False, label='centroid: $\sqrt{col^2+row^2}$')
#         ax_col.plot(time, col, color='r', rasterized=False, lw=0.1)
        ax_col.set_ylabel('Pix')
#         ax_col.tick_params('y', colors='r')
        ax_col.label_outer()
#         ax_row = ax_col.twinx()
#         ax_row.plot(time, row, color='g', rasterized=False, lw=0.1)
#         ax_row.tick_params('y', colors='g')
#         ax_row.set_zorder(2)
        # Overplot excluded intervals
        intervals = result['excluded_intervals']
        for i,interval in enumerate(intervals):
            t1,t2 = interval
            ax_col.axvspan(t1,t2, color='yellow', alpha=0.3, label='excluded' if i==0 else None)
        lgnd = ax_col.legend(ncol=1, loc='best', markerscale=1,frameon=False, columnspacing=0)

        # Plot raw light curve and trend
        ax_lc1 = fig.add_subplot(stackgrid[1], sharex=ax_col)
        fluxraw = result['lc_raw']['flux']
        timeraw = result['lc_raw']['time']
        ax_lc1.plot(timeraw,fluxraw, color='k', rasterized=False, zorder=3, label='Raw LC')
        ax_lc1.set_ylabel('e$^{-}$/s')
        if not result['lc_trend'] is None:
            fluxtrend = result['lc_trend']['flux']
            timetrend = result['lc_trend']['time']
            ax_lc1.scatter(timetrend,fluxtrend, color='dodgerblue',marker='*', rasterized=False, s=1, zorder=4, linewidths=0, label='PCA trend')
        else:
            ax = fig.add_subplot(stackgrid[2], sharex=ax_col)
            text = result['tag']
            text = '\n'.join([t for t in chunks(text,47)]  )
            ax.text(0.5,0.5, text, transform=ax.transAxes, fontsize=1.1*fsize, ha='center', va='center', color='green', wrap=True)
            lgnd = ax_lc1.legend(ncol=1, loc='best', markerscale=1,frameon=False, columnspacing=0)
            ax_lc1.label_outer()
            return False
        
        lgnd = ax_lc1.legend(ncol=1, loc='best', markerscale=1,frameon=False, columnspacing=0)
        ax_lc1.label_outer()
              
        # Plot detrended light curve with sigma clipped outliers
        flux = result['lc_regressed_notoutlier']['flux']
        time = result['lc_regressed_notoutlier']['time']
        ax_lc2 = fig.add_subplot(stackgrid[2], sharex=ax_col)
        ax_lc2.plot(time,flux, color='k', rasterized=False, label='Detrended LC')
        ax_lc2.set_ylabel('e$^{-}$/s')
        ax_lc2.label_outer()
        lgnd = ax_lc2.legend(ncol=1, loc='best', markerscale=1,frameon=False, columnspacing=0)

        # Overplot excluded intervals
        for interval in intervals:
            t1,t2 = interval
            ax_lc1.axvspan(t1,t2, color='yellow', alpha=0.3)
            ax_lc2.axvspan(t1,t2, color='yellow', alpha=0.3)
        
        # Plot PCA
        time = result['lc_regressed']['time']
        # This does not include the constant term
        npc_all = result['pca_all']['npc'] 
        npc_used = result['pca_used']['npc'] 
        # Not not load the constant term
        pc_used = result['pca_used']['pc'][:-1] 
        pc_all = result['pca_all']['pc'][:-1] 
        ax_pca = fig.add_subplot(hgrid[1])
        for i,pc in enumerate(pc_used):
            offset = i*0.2
            ax_pca.scatter(time,pc+offset, marker='.', rasterized=True, s=1, zorder=3, linewidths=0)
        for pc in pc_all[npc_used:]:
            i+=1
            offset = i*0.2
            ax_pca.scatter(time,pc+offset, marker='.', color='k', rasterized=True, s=1, zorder=2, linewidths=0)
        ax_pca.text(0.5, 1.0, f'Princ. comp.', transform=ax_pca.transAxes, fontsize=fsize, ha='center', va='top', color='k')
        ax_pca.axes.get_yaxis().set_visible(False)
        ax_pca.set_zorder(1)
        
        return True

def sort_names(namelist,
               pattern):
    '''
    Purpose:
        Return the indices of the list sorted by the integer number specified
        between parenthesis in `pattern`
    
    Inputs:
        - namelist: list of str's
            names to be sorted. Ex: namelist=['file1.txt','file12.txt','file2.txt']
            
        - pattern:
            Regular expression that groups the number to be used for sorting
            the name. Ex: pattern='tess\d+_sec(\d+)_corrected.fits'
    
    Return:
        Same namelist but ordered
    '''

    # Ensure namelist is a list of str instances
    if not isinstance(namelist,list):
        raise TypeError("namelist must be a list of str instances. Ex: namelist=['file1.txt','file12.txt','file2.txt']")
    for name in namelist:
        if not isinstance(name,str):
            raise TypeError('namelist must be a list of str instances. Ex: filenames=["file1.txt"]')

    # Parse the nmber in the names
    numbers = [re.search(pattern,name).group(1) for name in namelist]
    numbers = np.array(numbers, dtype=int)
    ind = np.argsort(numbers)
    namelist = np.array(namelist)[ind].tolist()
    
    return namelist

def finder_TIC_files(TIC,
                     inputdir=Path.cwd(),
                     file_pattern='tess{TIC}_sec*.fits',
                     sort_pattern=None):
    '''
    Purpose
    -------
    Find all the files that matches the given TIC number

    Parameters
    ----------
    TIC : str
        TIC number
        
    inputdir : pathlib.Path, optional
        Directory where to perform the search. The default is Path.cwd().
        
    file_pattern : srt, optional
        Pattern used to perform the search. Note that {TIC} will be replaced
        for the TIC number value. The default is 'tess{TIC}_sec*.fits'.
    
    sort_pattern : raw str, optional
        Regular expression used to find the number in the filename to be ussed
        to sort the output list. Ex: r'tess\d+_sec(\d+).fits'. The 
        default is None
    '''
    
    # Ensure TIC is a string (and not a number)
    if not isinstance(TIC,str):
        raise TypeError('TIC must be a string instance. Ex: TIC="349092922"')
    # Ensure inputdir is a Path instance
    if not isinstance(inputdir,Path):
        raise TypeError('inputdir must be a Path instance. Ex: outputdir=pathlib.Path.cwd()')
    # Ensure file_pattern is a string
    if not isinstance(file_pattern,str):
        raise TypeError('file_pattern must be a string instance that contains the characters {TIC}. Ex: "tess{TIC}_sec*.fits"')
    else:        
        if (not '{TIC}' in file_pattern) \
        or (not file_pattern.endswith('.fits')):
            raise TypeError('file_patternmust be a string instance containing the characters {TIC} and {SECTOR}. Ex: "tess{TIC}_sec*.fits"')
    # Ensure sort_pattern is a string (and not a number)
    if not sort_pattern is None:
        if not isinstance(sort_pattern,str):
            raise TypeError('sort_pattern must be a string instance. Ex: sort_pattern=r"tess\d+_sec(\d+)_corrected.fits"')

    file_pattern = file_pattern.format(TIC=TIC)
    filenames = inputdir.glob(file_pattern)
    filenames = [file.as_posix() for file in filenames]
    if not sort_pattern is None:
        filenames = sort_names(filenames,sort_pattern)
    return filenames

def create_only_plot_pickle(filenames):
    '''

    Parameters
    ----------
    filenames : pathlib.Path
        Path to pickled file of corrected light curves ciontaining all sectors
        of a TIC number

    Raises
    ------
    TypeError
        DESCRIPTION.

    Returns
    -------
    fig : TYPE
        DESCRIPTION.

    '''
    
    # Ensure filenames is a pathlib.Path instances
    if not isinstance(filenames,Path):
       raise TypeError('filenames must be apathlib.Path instances.')

    with open(filenames.as_posix(), 'rb') as picklefile:
        results = pickle.load(picklefile)

    nsectors = len(results)
    
    # Plot parapeters
    fsize=7
    params = {'axes.labelsize': fsize,
              'axes.titlesize': fsize,
              'legend.fontsize': fsize/1.5,
              'xtick.major.size':fsize/7,
              'xtick.major.width':fsize/7/10,
              'xtick.minor.size':0,
              'xtick.minor.width':fsize/7/10,
              'ytick.major.size':fsize/7,
              'ytick.major.width':fsize/7/10,
              'ytick.minor.size':0,
              'ytick.minor.width':fsize/7/10,
              'axes.linewidth': fsize/7/10,
              'lines.linewidth': fsize/7/10,
              'xtick.labelsize': fsize*0.75,
              'ytick.labelsize': fsize*0.75}
    plt.rcParams.update(params)
    fig_width, fig_height = 8.27, 24
    ax_sector_heights = np.repeat(3,nsectors)
    ax_stitched_heights = np.array([1.3,1.3])
    fig_sectors_height = np.sum(ax_sector_heights)
    fig_stitcheds_height = np.sum(ax_stitched_heights)
    fig_stitcheds_sectors_hratio = fig_stitcheds_height/fig_sectors_height
    fig_sectors_height = fig_height/(fig_stitcheds_sectors_hratio+1)
    fig_stitcheds_height = fig_height-fig_sectors_height
    fig_sectors_height /= 13
    fig_sectors_height *= nsectors
    fig_height = fig_sectors_height+fig_stitcheds_height

    # Create figure
    fig =  plt.figure(figsize=(fig_width, fig_height), constrained_layout=False, dpi=300)
    
    # Initialize main grid of vertical axes
    outer_grid = fig.add_gridspec(nsectors+2, 1, height_ratios=[*ax_sector_heights,*ax_stitched_heights], hspace=0.2)
    
    # List to store the detrended light curves from each sector
    lcs = []
    # List to store the time intervals of each sector
    sector_intervals = []
    
    # Plots for each sector
    for i,result in enumerate(results):
        
        print(f"SECTOR: {result['sector']}")
        
        # Create the 3 horizontal axes
        hgrid = outer_grid[i,0].subgridspec(1, 3, width_ratios=[5,1,1], wspace=0)

        # Plot TESS images
        flag = plot_images_pickle(fig,hgrid[2],result,fsize*0.75,fsize*0.75)
        print('SAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA')
        # If not all images
        if flag == False:
            ax = fig.add_subplot(hgrid[0])
            text = result['tag']
            text = '\n'.join([t for t in chunks(text,47)]  )
            ax.text(0.5,0.5, text, transform=ax.transAxes, fontsize=1.1*fsize, ha='center', va='center', color='green', wrap=True)
            continue

        # Plot the stacked plot and PCA plot
        flag = triple_stacked_and_PCA_plot_picke(fig,hgrid,result,fsize)
        if flag == False:
            continue

        # Collect the detrended light curve without outliers of each sector
        flux = result['lc_regressed_notoutlier']['flux']
        time = result['lc_regressed_notoutlier']['time']
        lc = lk.LightCurve(time=time, flux=flux)
        lcs.append(lc)
        # Collect the time intervals of each sector
        time = result['lc_raw']['time']
        tmin, tmax = np.min(time), np.max(time)
        sector_intervals.append((tmin,tmax))        

    # Set the title
    tic = result['tic']
    axs = fig.get_axes()
    pos = np.array([ [ax.get_position().xmin,ax.get_position().ymax] for ax in axs ])
    ind = np.argmax(pos[:,1], axis=0)
    axs[ind].text(0.5, 1.0, f'TIC {tic}', transform=axs[ind].transAxes, fontsize=fsize, ha='center', va='bottom', color='k')

    
    # Plot also stitched light curves without extra filters + its periodogram
    if len(lcs) > 1:
        
        # Sticht the light curve
        lc = lk.LightCurveCollection(lcs).stitch(corrector_func=Normalize_lc)        

        # Plot the stitched light curve
        ax_stitched = fig.add_subplot(outer_grid[-2])
        try:
            ax_stitched.scatter(lc.time.value, lc.flux.value*1000, s=2, marker='.', linewidths=0)
        except AttributeError:
            ax_stitched.scatter(lc.time, lc.flux*1000, s=2, marker='.', linewidths=0)
        for i,interval in enumerate(sector_intervals):
            tmin, tmax = interval[0], interval[1]
            ax_stitched.axvspan(tmin, tmax, facecolor='gray', alpha=0.30 if i%2==0 else 0.15, edgecolor='None')
        ax_stitched.set_ylabel('e$^{-}/s$')

        # Generate the Lomb-Scarglet periodogram
        pg = lc.to_periodogram()

        # Get the periodogram SNR by smoothing the power to obtain the background noise level
        snr_spectrum, pg_bkg = pg.flatten(method='logmedian', filter_width=0.3, return_trend=True)
        # Select SNR greater than 4 and 3
        mask_SNR4 = snr_spectrum.power >=  4
        mask_SNR3 = snr_spectrum.power >=  3
        # Find the estimative period of the peaks
        ind_peaks4 = peakutils.indexes(snr_spectrum.power[mask_SNR4], thres=0.0, min_dist=1, thres_abs=True)
        ind_peaks3 = peakutils.indexes(snr_spectrum.power[mask_SNR3], thres=0.0, min_dist=1, thres_abs=True)

        # Plot the LS periodogram
        ax_pg = fig.add_subplot(outer_grid[-1])
        x, y = pg.period.value, pg.power.value
        ax_pg.plot(x, y, zorder=2)
        # Plot detections
        if len(ind_peaks3) > 0:
            ax_pg.scatter(x[mask_SNR3][ind_peaks3], y[mask_SNR3][ind_peaks3], marker='o', c='yellow', s=2, label='SNR > 3', rasterized=False, zorder=3, edgecolors='k', linewidth=0.2) #, edgecolors='k', linewidth=0.2
        if len(ind_peaks4) > 0:
            ax_pg.scatter(x[mask_SNR4][ind_peaks4], y[mask_SNR4][ind_peaks4], marker='o', c='red', s=1, label='SNR > 4', rasterized=False, zorder=4)
        # Plot background
        x, y = pg_bkg.period.value, pg_bkg.power.value
        ax_pg.plot(x, y, ls='dotted', color='lime', label=None, lw=1, rasterized=False)
        # Axes limit
        if len(ind_peaks3) > 0:
            ind, mask = ind_peaks3, mask_SNR3
            if len(ind_peaks4) > 0:
                ind, mask = ind_peaks4, mask_SNR4
            scale_factor = 1.1
            xmin = x[mask][ind].min()
            xmax = x[mask][ind].max()
            xmax_minus_xmin = xmax - xmin
            xmax_plus_xmin = xmax + xmin
            xmax_new = 0.5*( scale_factor*xmax_minus_xmin + xmax_plus_xmin) 
            xmin_new = xmax_plus_xmin - xmax_new
        else:
            xmin_new, xmax_new = 0, 20
        ax_pg.set_xlim(xmin_new,xmax_new)
        lgnd = ax_pg.legend(ncol=1, title_fontsize=3, loc='best', fontsize=0.75*fsize, markerscale=1,frameon=False, handletextpad=0.1)
        ax_pg.set_xlabel('Period (days)')
        ax_pg.set_ylabel('e$^{-}/s$')

    fig.tight_layout()
    
    return fig




def make_pdfs(filenames,
              pdfname=Path('diagnosis.pdf')):
    
    
    with PdfPages(pdfname) as pdf:

        fig = create_only_plot(filenames)

        pdf.savefig(fig,bbox_inches='tight')
    


if __name__ == '__main__':
    
    # Example of a custom run:

    # # Unbuffered print as default
    # print = functools.partial(print, flush=True)

    # # Target to be plotted
    # TIC='306631043'
    # TIC='349522044'
    # TIC='149931294'
    # TIC='167206809'
    # TIC='150102343'
    
    # # Directory where to search the files that will generate the plots
    # inputdir = Path('/lhome/stefano/Documents/work/refinemont_lc_extraction/tpfs_test/corrected/arich')
    # file_pattern='tess{TIC}_sec*_corrected.fits'
    # sort_pattern=r'tess\d+_sec(\d+)_corrected.fits'
    # files = finder_TIC_files(TIC, inputdir=inputdir, file_pattern=file_pattern, sort_pattern=sort_pattern)
    # if len(files) == 0 : raise ValueError('No files found')
    
    # # PDF to be created
    # # pdfname = f'/lhome/stefano/Documents/work/arich/TICv8_sCVZ/new_extraction_lc/test{TIC}.pdf'
    # pdfname = f'/lhome/stefano/Documents/work/meetings/May11/Cases_tagged_as_bad_by_new_code/TIC{TIC}.pdf'
    # pdfname = Path(pdfname)
    
    # # Make plot
    # create_plot(files, pdfname=pdfname)


    
    ########### PDF pages
    
    import pandas as pd
    import itertools
    
    # Unbuffered print as default
    print = functools.partial(print, flush=True)

    
    # I/O directories
    # outputdir = Path('/lhome/stefano/Documents/work/refinemont_lc_extraction/tpfs_test/corrected/arich/pickled/sector_grouped')
    # inputdir = Path('/lhome/stefano/Documents/work/refinemont_lc_extraction/tpfs_test/corrected/arich/pickled/sector_grouped')
    # outputdir = Path('/lhome/stefano/Documents/work/catalogs/Luc')
    # inputdir = Path('/lhome/stefano/Documents/work/catalogs/Luc/set')
    outputdir = Path('../plots1')
    if not outputdir.exists():
        outputdir.mkdir(parents=True)

    inputdir = Path('../sector_grouped1')
    
    file_pattern='tess{TIC}_allsectors_corrected.pickled'
    
    # A PDF with many pages can use lots of RAM
    chunksize = 10
    
    # TICs to plot
    # TIC_filelist = '/lhome/stefano/Documents/work/arich/TICv8_sCVZ/corrected_lcs_pickled/gmode_comb_candidates/TICs.list'
    # TICs = pd.read_csv(TIC_filelist, header=None, names=['tic'])
    # TICs = TICs.tic.astype(str).values.flatten().tolist()
    
    
    # TICs = '30268695'
    # TICs = '30322459'
    # TICs = '40795195'
    # TICs = '96651307'
    # TICs = '279952991'
    TICs = '374944608'
    
    
    # TICs = pd.read_csv('/lhome/stefano/Documents/work/catalogs/Luc/TICs_candidates.list')
    # TICs = TICs.astype(str).values.flatten().tolist()
    
    
    # Ensure TICs is not an int instance
    if isinstance(TICs,int):
        raise TypeError('TICs must be a string instance (ex: TIC="349092922") or an iterable of strings (ex: TICs=["349092922","55852823"])')
    # If TICs is a plain string, convert to list
    if isinstance(TICs,str):
        TICs = [TICs]
    # Ensure TICs is iterable
    try:
        _ = iter(TICs)
        del _
    except TypeError:
        raise TypeError('TICs has to be an iterable of strings. Ex: TICs=["349092922","55852823"]')

    nTICs = len(TICs)    
    
    # PDF output name
    if nTICs == 1:
        TIC = TICs[0]
        pdfname = Path(f'diagnosis_TIC{TIC}.pdf')
    else:
        pdfname = Path('diagnosis.pdf')
    
    # If chunks, prepare the PDF
    if chunksize is None:
        outputfile = (outputdir/pdfname).as_posix() 
        pdf = PdfPages(outputfile)
    else:
        ichunk_counter = itertools.count(0)
        
    # Iterate over the TIC numbers
    for i,tic in enumerate(TICs):
        
        # # Manually skip
        # if i+1 < 81:
        #     continue
        
        if not chunksize is None:
            # Open/close the PDF in chunks
            if i % chunksize == 0:
                try:
                    pdf.close()
                except NameError:
                    pass
                ichunk = next(ichunk_counter)
                pdfname_chunk = Path(pdfname.stem+f'_{ichunk}'+pdfname.suffix)
                outputfile = (outputdir/pdfname_chunk).as_posix() 
                pdf = PdfPages(outputfile)
        
        print(f'Plotting TIC {tic} ..... [{i+1}/{nTICs}]')
        
        # files = finder_TIC_files(tic, inputdir=inputdir, file_pattern=file_pattern, sort_pattern=sort_pattern)
        filepath = file_pattern.format(TIC=tic)
        filepath = inputdir/Path(filepath)
        fig = create_only_plot_pickle(filepath)
        pdf.savefig(fig,bbox_inches='tight')
        plt.close(fig)
    
    pdf.close()
    