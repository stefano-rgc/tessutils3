#!/usr/bin/env python
import signal, functools, warnings, time, pickle
from pathlib import Path
import numpy as np
import pandas as pd
import lightkurve as lk 
from scipy import ndimage
from astropy.stats.funcs import median_absolute_deviation as MAD
import astropy.units as u
from astropy.io import fits
from astropy.modeling import fitting, functional_models
from astropy.coordinates import SkyCoord
from astroquery.mast import Catalogs
from joblib import Parallel, delayed
from IPython import embed

########################################################################
### Supporting functions of the main function `extract_light_curve()`
########################################################################

def threshold_mask(image, threshold=3, reference_pixel='center'):
    """
    Source: LightKurve module
    Note: The original only works only as a method. I made it a function.
    ----------
    Returns an aperture mask creating using the thresholding method.
    This method will identify the pixels in the TargetPixelFile which show
    a median flux that is brighter than `threshold` times the standard
    deviation above the overall median. The standard deviation is estimated
    in a robust way by multiplying the Median Absolute Deviation (MAD)
    with 1.4826.
    If the thresholding method yields multiple contiguous regions, then
    only the region closest to the (col, row) coordinate specified by
    `reference_pixel` is returned.  For exmaple, `reference_pixel=(0, 0)`
    will pick the region closest to the bottom left corner.
    By default, the region closest to the center of the mask will be
    returned. If `reference_pixel=None` then all regions will be returned.
    Parameters
    ----------
    threshold : float
        A value for the number of sigma by which a pixel needs to be
        brighter than the median flux to be included in the aperture mask.
    reference_pixel: (int, int) tuple, 'center', or None
        (col, row) pixel coordinate closest to the desired region.
        For example, use `reference_pixel=(0,0)` to select the region
        closest to the bottom left corner of the target pixel file.
        If 'center' (default) then the region closest to the center pixel
        will be selected. If `None` then all regions will be selected.
    Returns
    -------
    aperture_mask : ndarray
        2D boolean numpy array containing `True` for pixels above the
        threshold.
    """
    if reference_pixel == 'center':
        # reference_pixel = (image.shape[2] / 2, image.shape[1] / 2)
        # Stefano added:
        reference_pixel = (image.shape[1] / 2, image.shape[0] / 2)
    vals = image[np.isfinite(image)].flatten()
    # Calculate the theshold value in flux units
    mad_cut = (1.4826 * MAD(vals) * threshold) + np.nanmedian(image)
    # Create a mask containing the pixels above the threshold flux
    threshold_mask = np.nan_to_num(image) > mad_cut
    if (reference_pixel is None) or (not threshold_mask.any()):
        # return all regions above threshold
        return threshold_mask
    else:
        # Return only the contiguous region closest to `region`.
        # First, label all the regions:
        labels = ndimage.label(threshold_mask)[0]
        # For all pixels above threshold, compute distance to reference pixel:
        label_args = np.argwhere(labels > 0)
        distances = [np.hypot(crd[0], crd[1])
                     for crd in label_args - np.array([reference_pixel[1], reference_pixel[0]])]
        # Which label corresponds to the closest pixel?
        closest_arg = label_args[np.argmin(distances)]
        closest_label = labels[closest_arg[0], closest_arg[1]]
        return labels == closest_label

def query_TIC(target, target_coord, tic_id=None, search_radius=600.*u.arcsec, **kwargs):
        """
            Source: Courtesy of Dr. Timothy Van Reeth
            Note: I modified the behaviour when `tic_id` is given
            Retrieving information from the TESS input catalog. 
            
            Parameters:
                target: target name
                target_coord (optional): target coordinates (astropy Skycoord)
                search_radius: TIC entries around the target coordinaes wihtin this radius are considered.
                **kwargs: dict; to be passed to astroquery.Catalogs.query_object or query_region.
        """
        
        def _tic_handler(self,signum):
            '''Supporting function of `query_TIC`'''
            print('the query of the TIC is taking a long time... Something may be wrong with the database right now...')

        deg_radius = float(search_radius / u.deg)
        arc_radius = float(search_radius / u.arcsec)
        
        tic = None
        tess_coord = None
        tmag = None 
        nb_coords = []
        nb_tmags = []
        tic_index = -1
        
        try:
            # The TIC query should finish relatively fast, but has sometimes taken (a lot!) longer.
            # Setting a timer to warn the user if this is the case...
            signal.signal(signal.SIGALRM,_tic_handler)
            signal.alarm(30) # This should be finished after 30 seconds, but it may take longer...
            
            catalogTIC = Catalogs.query_region(target_coord, catalog="TIC", radius=deg_radius,**kwargs)
            signal.alarm(0)
            
        except:
            print(f"no entry could be retrieved from the TIC around {target}.")
            catalogTIC = []
        
        if(len(catalogTIC) == 0):
            print(f"no entry around {target} was found in the TIC within a {deg_radius:5.3f} degree radius.")
        
        else:
            if not (tic_id is None):
                # tic_index = np.argmin((np.array(catalogTIC['ID'],dtype=int) - int(tic_id))**2.)
                # Stefano added:
                tic_index = np.argwhere(catalogTIC['ID'] == str(tic_id))
                if tic_index.size == 0:
                    return '-1', None, None, None, None

                else:
                    tic_index = tic_index.item()
            else:
                tic_index = np.argmin(catalogTIC['dstArcSec'])
        
            if(tic_index < 0):
                print(f"the attempt to retrieve target {target} from the TIC failed.")
            
            else:
                tic = int(catalogTIC[tic_index]['ID'])
                ra = catalogTIC[tic_index]['ra']
                dec = catalogTIC[tic_index]['dec']
                tmag = catalogTIC[tic_index]['Tmag']
                
                # Retrieve the coordinates
                tess_coord = SkyCoord(ra, dec, unit = "deg")
                
                # Collecting the neighbours
                if(len(catalogTIC) > 1):
                    for itic, tic_entry in enumerate(catalogTIC):
                        if(itic != tic_index):
                            nb_coords.append(SkyCoord(tic_entry['ra'], tic_entry['dec'], unit = "deg"))
                            nb_tmags.append(tic_entry['Tmag'])
        
        nb_tmags = np.array(nb_tmags)
        
        return tic, tess_coord, tmag, nb_coords, nb_tmags

def check_aperture_mask(aperture, prepend_err_msg=''):
    '''
    Purpose: 
        Apply geometric criteria to assess the validity of the aperture mask
    '''

    # Parameters / Criteria
    max_elongation = 14 # pix
    min_pixels = 4 # pix

    # Initializations
    err_msg = ''
    OK_aperture = True

    # Convert from boolean to int
    aperture = aperture.astype(int)

    # If not apperture found
    if not np.any(aperture) and OK_aperture:
        err_msg = print_err('Not aperture found.', prepend=prepend_err_msg)
        OK_aperture = False

    # If too large aperture
    # if False:
    #     if not np.sum(aperture.astype(int)) < 9*9:
    #         return False

    # If too elongated aperture (column)
    if not np.all(aperture.sum(axis=0) <= max_elongation) and OK_aperture:
        # Check if is a bad defined aperture
        if len( set( aperture.sum(axis=0) ) ) < 4:
            err_msg = print_err('Too elongated aperture (column).', prepend=prepend_err_msg)
            OK_aperture = False
        if np.any(aperture.sum(axis=0) == 0):
            err_msg = print_err('Too elongated aperture (column), bad TESS image.', prepend=prepend_err_msg)
            OK_aperture = False

    # If too elongated aperture (row)
    if not np.all(aperture.sum(axis=1) <= max_elongation) and OK_aperture:
        # Check if is a bad defined aperture
        if len( set( aperture.astype(int).sum(axis=1) ) ) < 4:
            err_msg = print_err('Too elongated aperture (row).', prepend=prepend_err_msg)
            OK_aperture = False
        if np.any(aperture.sum(axis=1) == 0):
            err_msg = print_err('Too elongated aperture (row), bad TESS image.', prepend=prepend_err_msg)
            OK_aperture = False

    # If too small aperture
    if not np.sum(aperture) >= min_pixels and OK_aperture:
        err_msg = print_err('Too small aperture.', prepend=prepend_err_msg)
        OK_aperture = False

    return OK_aperture, err_msg

def find_fainter_adjacent_pixels(seeds, image, max_iter=100):    
    '''
    Purpose:
        Given an initial pixel(s), find surrounding ones until the pixel value increases
    '''

    # Check that the dimension of `seeds` is correct
    try:
        if seeds.ndim != 2:
            raise ValueError('`seeds` has to be a 2D Numpy array whose second dimension has lengh 2. E.g.: np.array([[0,1], [5,5], [7,8]])')
        if seeds.shape[1] != 2:
            raise ValueError('`seeds` has to be a 2D Numpy array whose second dimension has lengh 2. E.g.: np.array([[0,1], [5,5], [7,8]])')
    except AttributeError:
        raise AttributeError('`seeds` has to be a 2D Numpy array whose second dimension has lengh 2. E.g.: np.array([[0,1], [5,5], [7,8]])')
        
    # Here we'll keep track of which pixels are what:
    # * -1: Not part of the mask.
    # *  0: Part of the mask and previous seed
    # * +1: Part of the mask
    score = np.repeat(-1,image.size).reshape(image.shape)
    
    # The center defined by the initial seeds
    score[seeds[:,0],seeds[:,1]] = 1

    # Initialize the counter
    counter = 0

    while True:

        # Check the counter
        if counter > max_iter:
            print(f'Maximum number of iterations exceeded: max_iter={max_iter}')
            break
            
        # Find which pixels use as centers
        centers = np.argwhere(score==1)

        # List to store the indices of the pixels to be included as part of the mask
        inds = []

        # Evaluate the condition for each center (i.e., search for adjacent fainter pixels)
        for center in centers:
            # Find the 4 adjacent pixels of pixel center
            mask = np.repeat(False,image.size).reshape(image.shape)
            mask[center[0],center[1]] = True
            mask = ~ndimage.binary_dilation(mask)
            mask[center[0],center[1]] = True
            masked_image = np.ma.masked_where(mask, image)

            # Find which of the adjacent pixels are not brighter than pixel center
            ind = np.argwhere(masked_image <= masked_image.data[center[0],center[1]])
            inds.append(ind)

        # If any pixel fainter than the center one
        if len(inds) > 0:
            ind = np.concatenate(inds)
            # Extend the mask
            score[ind[:,0],ind[:,1]] = 1
            # Flag the all previous centers
            score[centers[:,0],centers[:,1]] = 0
        # If no pixel fainter than the center one
        else:
            break

    mask = score + 1
    mask = mask.astype(bool)
    
    return mask

def mag2flux(mag, zp=20.60654144):
    """
    Source: https://tasoc.dk/code/_modules/photometry/utilities.html#mag2flux

    Convert from magnitude to flux using scaling relation from
    aperture photometry. This is an estimate.

    The scaling is based on fast-track TESS data from sectors 1 and 2.

    Parameters:
        mag (float): Magnitude in TESS band.

    Returns:
        float: Corresponding flux value
    """
    return np.clip(10**(-0.4*(mag - zp)), 0, None)

def tie_sigma(model):
    '''Used to constrain the sigma of the Gaussians when performing the fit in `contamination()`'''
    try:
        return model.x_stddev
    except AttributeError:
        return model.x_stddev_0
    
def tie_amplitude(model,factor=1):
    '''Used to constrain the amplitude of the Gaussians when performing the fit in `contamination()`'''
    return model.amplitude_0*factor

def contamination(results,image,aperture,target_coord_pixel,target_tmag,nb_coords_pixel,nb_tmags,wcs,median_bkg_flux,prepend_err_msg=''):
    '''
    Purpose:
        Calculate the fraction of flux in the aperture the mask that comes from
        neighbour stars. Done by means of fitting 2D Gaussians and a plane to
        the image
    '''
    
    # Initializations
    err_msg = ''

    # Set a maximum of 40 neighbour stars to fit
    if nb_tmags.size > 0:
        nb_tmags = nb_tmags[:40]
        nb_coords_pixel = nb_coords_pixel[:40,:]

    # Gaussian locations
    locations = np.array([*target_coord_pixel,*nb_coords_pixel])
    xpos = locations[:,0]
    ypos = locations[:,1]
    
    # Convert the magnitudes of the stars to flux
    fluxes = mag2flux( np.concatenate( [np.array([target_tmag]), nb_tmags] ) )
    
    # Create model
    Gaussian2D = functional_models.Gaussian2D
    Planar2D = functional_models.Planar2D
    
    Gaussians = [ Gaussian2D(amplitude=a,
                             x_mean=x, 
                             y_mean=y, 
                             x_stddev=1, 
                             y_stddev=1) for a,x,y in zip(fluxes,xpos,ypos) ]
    
    nGaussians = len(Gaussians)
    model = np.sum(Gaussians)
    
    # Set the constrains in the model
    if nGaussians == 1:
        getattr(model,'x_mean').fixed = True
        getattr(model,'y_mean').fixed = True
        getattr(model,'y_stddev').tied = tie_sigma
        getattr(model,'amplitude').min = 0.0

    else:
        for i in range(nGaussians):
            # Tie all Gaussian Sigmas to the x-dimension Sigma of the target star
            getattr(model,f'y_stddev_{i}').tied = tie_sigma
            getattr(model,f'x_stddev_{i}').tied = tie_sigma
            # Fix the Gaussian positions
            getattr(model,f'x_mean_{i}').fixed = True
            getattr(model,f'y_mean_{i}').fixed = True
            # Tie all the Gaussian amplitudes to the one of the target star
            fraction = fluxes[i]/fluxes[0]
            tmp = functools.partial(tie_amplitude, factor=fraction)
            getattr(model,f'amplitude_{i}').tied = tmp
            # Untie and unfix for the target star
            if i==0:
                getattr(model,f'amplitude_{i}').min = 0.0
                getattr(model,f'x_stddev_{i}').tied = False
                getattr(model,f'amplitude_{i}').tied = False
                
    # Add a 2D plane to the model
    model += Planar2D(slope_x=0, slope_y=0, intercept=median_bkg_flux)

    # Make the fit
    (xsize,ysize) = image.shape
    y, x = np.mgrid[:xsize, :ysize]
    fitter = fitting.LevMarLSQFitter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            Fit = fitter(model,x,y, image)
        except RecursionError as e:
            err_msg = print_err(str(e), prepend=prepend_err_msg)
            return None, err_msg

    # Results
    fitted_image = Fit(x,y)
    Plane = Fit[-1]
    TargetStar = Fit[0]
    if nGaussians > 1:
        Neighbours = []
        for i in range(1,nGaussians):
            Neighbours.append(Fit[i])
        Neighbours = np.sum(Neighbours)
    else:
        Neighbours = None
        
    # Find the significance of the non-homogeneous background
    v1 = (0,0,1)
    v2 = (Plane.slope_x.value, Plane.slope_y.value,-1)
    v1 /=  np.linalg.norm(v1)
    v2 /=  np.linalg.norm(v2)
    cos = np.dot(v1,v2)
    tan = np.sqrt(1-cos**2)/cos
    bkg_change = xsize*tan
    fraction_bkg_change = bkg_change/Plane(xsize/2,ysize/2)[0]
    fraction_bkg_change = np.abs(fraction_bkg_change)

    # Find the flux contribution of neighbor stars to the aperture mask
    #target_flux = np.sum( TargetStar(x,y)[aperture] )
    neighbour_flux = np.sum( (Fit(x,y)-TargetStar(x,y)-Plane(x,y))[aperture] )
    target_flux = np.sum( TargetStar(x,y)[aperture] )
    bkg_flux = np.sum( Plane(x,y)[aperture] )
    fraction_ap_contamination = neighbour_flux/target_flux 


    # In order to pickle, remove functions references in tie attribute of the fit components 
    TargetStar.y_stddev.tied = None
    if nGaussians > 2:
        try:
            for Gaussian in Neighbours:
                Gaussian.amplitude.tied = None
                Gaussian.x_stddev.tied = None
                Gaussian.y_stddev.tied = None
        except Exception:
            print('######################################')
            embed()
    
    elif nGaussians == 2:
        Neighbours.amplitude.tied = None
        Neighbours.x_stddev.tied = None
        Neighbours.y_stddev.tied = None
    
    # Store to results
    results['fit'] = {'fitted_image':fitted_image,\
                      'intercept':Plane.intercept.value,\
                      'Plane':Plane,\
                      'TargetStar':TargetStar,\
                      'Neighbours':Neighbours,\
                      'slope_y':Plane.slope_y.value,\
                      'slope_x':Plane.slope_x.value,\
                      'neighbour_flux_ap':neighbour_flux,\
                      'target_flux_ap':target_flux,\
                      'bkg_flux_ap':bkg_flux,\
                      'fraction_contamination_ap':fraction_ap_contamination,\
                      'fraction_bkg_change':fraction_bkg_change}
    
    return fitted_image, err_msg

def refine_aperture(results,tic,ra,dec,wcs,aperture,threshold,image,prepend_err_msg=''):
    '''
    Purpose:
        Find an aperture mask that only contains one source and only
        decresing pixels in flux from that source
    '''

    # Parameters / Criteria
    thresholds = iter([7.5, 10, 15, 20, 30, 40, 50])
    arcsec_per_pixel = 21 * u.arcsec # TESS CCD
    nb_mags_below_target = 4

    # Initialization
    err_msg = ''

    # Query surrounding area in MAST
    search_radius_pixel = np.sqrt(2*np.max(image.shape)**2)/2
    search_radius = search_radius_pixel * arcsec_per_pixel
    target_coord = SkyCoord(ra, dec, unit = "deg")
    tic_tmp,\
    tess_coord, target_tmag,\
    nb_coords, nb_tmags = query_TIC(f'TIC {tic}',target_coord, search_radius=search_radius, tic_id=tic)

    # Check we retrieve the correct target
    if tic != str(tic_tmp):
        err_msg = print_err('The TIC number from the MAST query does not match the one from the TESS FITS image.', prepend=prepend_err_msg)
        return None, None, None, None, None, err_msg
        
    # Make neighbor coordenates into NumPy arrays
    nb_coords = np.array(nb_coords)

    # Store to results (get plain numbers from the AstroPy instance)
    results['neighbours_all'] = {'mag':nb_tmags,\
                                 'ra' :np.array([coord.ra.deg  for coord in nb_coords]),\
                                 'dec':np.array([coord.dec.deg for coord in nb_coords])}

    # Filter neighbour stars: Remove too faint stars
    nb_faintest_mag = target_tmag + nb_mags_below_target
    mask = nb_tmags <= nb_faintest_mag
    nb_tmags =  nb_tmags[mask]
    nb_coords = nb_coords[mask]

    # Convert coordenates to pixels
    target_coord_pixel = np.array( [target_coord.to_pixel(wcs,origin=0)] )

    # Store to results (get plain numbers from the AstroPy instance)
    results['target'] = {'mag':target_tmag,\
                         'ra' :target_coord.ra.deg,\
                         'dec':target_coord.dec.deg,\
                         'pix':target_coord_pixel}

    if nb_coords.size > 0:
        nb_coords_pixel = np.array( [coord.to_pixel(wcs,origin=0) for coord in nb_coords], dtype=float)
        # Filter neighbour stars: Remove stars outside the image bounds
        nb_pixel_value = ndimage.map_coordinates(image, [nb_coords_pixel[:,1], nb_coords_pixel[:,0]], order=0)
        mask = nb_pixel_value != 0
        nb_tmags =  nb_tmags[mask]
        nb_coords = nb_coords[mask]
        nb_coords_pixel = nb_coords_pixel[mask,:]
    
    else:
        nb_coords_pixel = np.array([])

    # Store to results (get plain numbers from the AstroPy instance)
    results['neighbours_used'] = {'mag':nb_tmags,\
                                  'ra' :np.array([coord.ra.deg  for coord in nb_coords]),\
                                  'dec':np.array([coord.dec.deg for coord in nb_coords]),\
                                  'pix':nb_coords_pixel}

    if nb_coords.size > 0:
        
        # Make neighbour pixels coordenates match the image grid, ie, bin them
        nb_coords_pixel_binned = np.floor(nb_coords_pixel+0.5)

        while True:
            
            # Find if a neighbour is within the aperture mask
            overlaps = ndimage.map_coordinates(aperture.astype(int), \
                           [nb_coords_pixel_binned[:,1], nb_coords_pixel_binned[:,0]], order=0)

            if overlaps.sum() == 0:
                break
            else:
                try:    
                    # Make a new aperture mask
                    threshold = next(thresholds)
                    aperture = threshold_mask(image, threshold=threshold, reference_pixel='center')
                
                except StopIteration:
                    # If no more thresholds to try, set the aperture to `None`
                    err_msg = print_err('Not isolated target star.', prepend=prepend_err_msg)
                    results['masks']['aperture'] = None
                    return None, target_coord_pixel, target_tmag, nb_coords_pixel, nb_tmags, err_msg

                if np.sum(aperture.astype(int)) == 0:
                    # If no aperture left, set the aperture to `None`
                    err_msg = print_err('Not isolated target star.', prepend=prepend_err_msg)
                    results['masks']['aperture'] = None
                    return None, target_coord_pixel, target_tmag, nb_coords_pixel, nb_tmags, err_msg

    # Store to results
    results['aperture_threshold'] = threshold
    
    # Find the brightest pixel within the mask
    ap_image = np.ma.masked_where(~aperture, image)
    seeds = np.argwhere(ap_image==ap_image.max())
    # Ensure the mask contains only pixels with decreasing flux w.r.t. the brightest pixel
    aperture = find_fainter_adjacent_pixels(seeds,ap_image)
    
    # Make target pixel coordenate match the image grid, ie, bin it
    target_coords_pixel_binned = np.floor(target_coord_pixel+0.5)
    # Find if a target is within the aperture mask
    overlaps = ndimage.map_coordinates(aperture.astype(int), \
                   [target_coords_pixel_binned[:,1], target_coords_pixel_binned[:,0]], order=0)
    if overlaps.sum() == 0:
        err_msg = print_err('Target star not within the mask.', prepend=prepend_err_msg)
        results['masks']['aperture'] = None
        return None, target_coord_pixel, target_tmag, nb_coords_pixel, nb_tmags, err_msg

    # Store to results
    results['masks']['aperture'] = aperture

    return aperture, target_coord_pixel, target_tmag, nb_coords_pixel, nb_tmags, err_msg

def exclude_interval(tpf,sector,results):
    '''
    Purpose:
        Remove cadences of the LightKurve target pixel file (tpf) based on time ranges (in days)
    '''

    # Initializations
    intervals = []
    
    if int(sector) == 1:
        intervals = [ (1334.8, 1335.1),
                      (1347.0, 1349.5) ] # days

    if int(sector) == 2:
        intervals = [ (1356.2, 1356.5),
                      (1361.0, 1361.3),
                      (1363.5, 1363.8),
                      (1365.9, 1366.2),
                      (1373.8, 1374.1),
                      (1375.8, 1376.0),
                      (1377.9, 1378.7) ] # days

    if int(sector) == 3:
        intervals = [ (1380.0, 1385.0),
                      (1387.6, 1387.9),
                      (1390.1, 1390.4),
                      (1392.6, 1392.9),
                      (1395.1, 1395.4),
                      (1398.6, 1398.9),
                      (1400.6, 1400.9),
                      (1402.6, 1402.9),
                      (1404.6, 1404.9),
                      (1406.1, 1406.4) ] # days

    if int(sector) == 4:
        intervals = [ (1420.0, 1427.0) ]

    if int(sector) == 5:
        intervals = [ (1463.0, 1465.0) ] # days

    if int(sector) == 6:
        intervals = [ (1476.0, 1479.0) ] # days

    if int(sector) == 7:
        intervals = [ (1502.5, 1506.0) ] # days

    if int(sector) == 8:
        intervals = [ (1529.5, 1533.0) ] # days

    # for interval in intervals:
    for interval in intervals:
        # Find the indices of the quality mask that created tpf.time
        ind = np.argwhere(tpf.quality_mask == True)
        mask  = tpf.time > interval[0]
        mask &= tpf.time < interval[1]
        # Set to False the indices to be masked (ignored).
        # Note that we take a subset from `ind` because the masks where defined from tpf.time
        tpf.quality_mask[ind[mask]] = False

    # Store to results
    intervals = np.array(intervals)
    results['excluded_intervals'] = intervals

def find_number_of_PCs(results,regressors,lc):
    '''
    Purpose:
        Find the number of principal components to use in the PCA method
    '''

    # Parameters / Criteria
    npc = 7
    nbins = 40 # number of bins to divide the each PC
    threshold_variance = 1e-4 # 5 # -> 500% # THIS IS NOT USED.

    dm = lk.DesignMatrix(regressors, name='regressors').pca(npc).append_constant()
    rc = lk.RegressionCorrector(lc)
    lc_regressed = rc.correct(dm)

    # Find the median of the moving variance for each PCs
    boxsize = np.floor( dm.values[:,0].size/nbins ).astype(int)
    pcs_variances = []
    for ipc in range(npc):
        pcs_variances.append( pd.Series(dm.values[:,ipc]).rolling(boxsize).var().median() )
    # pcs_variances = [ pd.Series(dm.values[:,ipc]).rolling(boxsize).var().median() for ipc in range(npc) ]
    pcs_variances = np.array(pcs_variances)
    relative_change_variances = np.abs(np.diff(pcs_variances)/pcs_variances[:-1]) # < THIS IS NOT USED.

    # Select PCs with variance above threshold value
    ind_bad_pcs = np.argwhere(pcs_variances>threshold_variance)
    if ind_bad_pcs.size > 0:
        # Find index first PC that exceeds threshold value. This is the new npc
        new_npc = ind_bad_pcs[0].item()
    else:
        print(f'No PCs with variance>{threshold_variance}. All {npc} PCs used.')
        new_npc = npc

    # Store to results
    results['pca_all'] = {'coef':rc.coefficients,\
                          'pc':[dm.values[:,i] for i in range(dm.rank)],\
                          'dm':dm,\
                          'rc':rc,\
                          'npc':npc,\
                          'npc_used':new_npc,\
                          'pc_variances':pcs_variances,\
                          'threshold_variance':threshold_variance,\
                          'nbins':nbins}

    return new_npc, dm, rc

def print_err(err_msg, prepend=''):
    '''Convenience function to print error messages'''
    err_msg = prepend + err_msg
    print(err_msg)
    return err_msg

################################################################
### Main function to extract light curves from the TESS images
################################################################

def extract_light_curve(fits_filename,outputdir,return_msg=True):

    # Output name
    output = Path(fits_filename.stem+'_corrected.pickled')
    output = outputdir/output

    # Parameters and  Criteria:
    sigma_clipping = 5 # To be applied after the detrending of the light curve

    # Structure the data to be saved
    results = {'tic':                    None,
               'sector':                 None,
               'ra':                     None,
               'dec':                    None,
               'headers':                None,  # Headers from the original FITS file
               'fit':                    None,  # Result from fit
               'neighbours_all':         None,  # All neighbours stars info
               'neighbours_used':        None,  # Used neighbours stars info
               'target':                 None,  # Target star info
               'aperture_threshold':     None,  # HDU to store tabular information
               'pca_all':                None,  # PCA results
               'pca_used':               None,  # PCA results
               'centroids':              None,  # Centroids results
               'excluded_intervals':     None,  # Excluded intervals in days
               'lc_raw':                 None,  # Light curves
               'lc_raw_nonan':           None,  # Light curves
               'lc_trend':               None,  # Light curves
               'lc_regressed':           None,  # Light curves
               'lc_regressed_notoutlier':None,  # Light curves
               'median_image':           None,  
               'masks':                  None,
               'tag':                    None}  

    # Save information from header from original FITS file
    HDUL, ext = [], 0
    while True:
        try:
            HDUL.append( fits.getheader(fits_filename.as_posix(), ext=ext).tostring() )
            ext += 1
        except IndexError:
            break
        except Exception as e:
            print('Unexpected exception when reading headers from FITS: ', e)
            break
    results['headers'] = HDUL

    # Load the TESS taret pixel file
    try:
        tpf = lk.TessTargetPixelFile(fits_filename.as_posix())
    except Exception as e:
        # Save results
        err_msg = f'"lightkurve.TessTargetPixelFile()" could not open file {fits_filename.as_posix()}. Exception: {e}'
        results['tag'] = err_msg
        picklefile = open(output.as_posix(), 'wb')
        pickle.dump(results, picklefile)
        picklefile.close()
        if return_msg: return err_msg
        return
    tic = str( tpf.get_keyword('ticid') )
    sector = str( tpf.get_keyword('sector') )
    target_ra = np.float( tpf.ra )
    target_dec = np.float( tpf.dec )
    # Store to results
    results['tic'], results['sector'] = tic, sector
    results['ra'], results['dec'] = target_ra, target_dec

    # Initialize messages
    id_msg = f'TIC {tic} Sector {sector}: Skipped: '
    OK_msg = f'TIC {tic} Sector {sector}: OK'
    
    # Calculate the median image
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        median_image = np.nanmedian(tpf.flux, axis=0)
        # Store to results
        results['median_image'] = median_image

    # Estimate of aperture mask and background mask
    ap_mask_threshold, bkg_mask_threshold = 5, 3
    ap_mask =   threshold_mask(median_image, threshold=ap_mask_threshold, reference_pixel='center')
    ap_bkg  = ~ threshold_mask(median_image, threshold=bkg_mask_threshold, reference_pixel=None)
    # Exclude NaN values outside the camera
    ap_bkg &= ~np.isnan(median_image) 
    # Estimate the median flux background
    median_bkg_flux = np.median(median_image[ap_bkg])
    # Store to results
    results['masks'] = {'aperture':ap_mask,\
                        'background':ap_bkg}
    
    # Check validity of aperture mask
    OK_ap_mask, err_msg = check_aperture_mask(ap_mask, id_msg)
    # If aperture is not good, exit program with corresponding message
    if not OK_ap_mask:
        # Save results
        results['tag'] = err_msg
        picklefile = open(output.as_posix(), 'wb')
        pickle.dump(results, picklefile)
        picklefile.close()
        if return_msg: return err_msg
        return
   
    # Refine aperture
    try:
        WCS = tpf.wcs
    except IndexError:
        # Save results
        err_msg = id_msg+'No WCS info in header'
        results['tag'] = err_msg
        picklefile = open(output.as_posix(), 'wb')
        pickle.dump(results, picklefile)
        picklefile.close()
        if return_msg: return err_msg
        return
    ap_mask,\
    target_coord_pixel, target_tmag,\
    nb_coords_pixel, nb_tmags,\
    err_msg = refine_aperture(results, tic, target_ra, target_dec, WCS,\
                  ap_mask, ap_mask_threshold, median_image, prepend_err_msg=id_msg)
    # If not satisfactory aperture mask
    if ap_mask is None:
        # Save results
        results['tag'] = err_msg
        picklefile = open(output.as_posix(), 'wb')
        pickle.dump(results, picklefile)
        picklefile.close()
        if return_msg: return err_msg
        return

    # Variation in time of aperture's center of mass
    centroid_col, centroid_row = tpf.estimate_centroids(aperture_mask=ap_mask, method='quadratic')
    centroid_col -= tpf.column
    centroid_row -= tpf.row
    sqrt_col2_row2 = np.sqrt(centroid_col**2+centroid_row**2)
    # Store to results
    results['centroids'] = {'col':centroid_col,\
                            'row':centroid_row,\
                            'sqrt_col2_row2':sqrt_col2_row2,\
                            'time':tpf.time}

    # Fit the image and find the contamination fraction within the aperture mask
    fitted_image, err_msg = contamination(results, median_image,ap_mask,\
                                 target_coord_pixel, target_tmag,\
                                 nb_coords_pixel, nb_tmags,\
                                 tpf.wcs,median_bkg_flux,
                                 prepend_err_msg=id_msg)
    if fitted_image is None:
        # Save results
        results['tag'] = err_msg
        picklefile = open(output.as_posix(), 'wb')
        pickle.dump(results, picklefile)
        picklefile.close()
        if return_msg: return err_msg
        return

    # Generate the raw light curve
    lc_raw = tpf.to_lightcurve(aperture_mask=ap_mask, method='aperture')
    # Store to results
    results['lc_raw'] = {'flux':lc_raw.flux,\
                         'time':lc_raw.time}

    # Find the indices of the quality mask that created the light curve
    ind = np.argwhere(tpf.quality_mask == True)
    # Masks with True value the light curve times with null or NaN flux
    mask  = lc_raw.flux == 0
    mask |= lc_raw.flux == np.nan
    # Set to False the indices to be masked (ignored).
    # Note that we take a subset from `ind` because the masks where defined from the light curve
    tpf.quality_mask[ind[mask]] = False

    # Exclude intervals previously decided
    exclude_interval(tpf, sector, results)

    # Generate the light curve
    lc = tpf.to_lightcurve(aperture_mask=ap_mask, method='aperture')
    # Store to results
    results['lc_raw_nonan'] = {'flux':lc.flux,\
                               'time':lc.time}

    # Make a design matrix and pass it to a linear regression corrector
    regressors = tpf.flux[:, ap_bkg]

    # Number of PCs to use
    npc, dm, rc = find_number_of_PCs(results,regressors,lc)

    if npc == 0:
        # Save results
        results['tag'] = id_msg+'None PC used, no detrended done.'
        picklefile = open(output.as_posix(), 'wb')
        pickle.dump(results, picklefile)
        picklefile.close()
        if return_msg: return err_msg
        return

    try:
        # Detrend light curve using PCA
        dm = lk.DesignMatrix(regressors, name='regressors').pca(npc).append_constant()
        rc = lk.RegressionCorrector(lc)
        lc_regressed = rc.correct(dm)
        lc_trend = rc.diagnostic_lightcurves['regressors']

        # Sigma clipping the remove outliers
        lc_regressed_no_outliers, lc_mask_regressed_outliers = lc_regressed.remove_outliers(return_mask=True, sigma=sigma_clipping)

        # Store to results
        results['lc_trend']                = {'flux':lc_trend.flux,\
                                              'time':lc_trend.time}
        results['lc_regressed']            = {'flux':lc_regressed.flux,\
                                              'time':lc_regressed.time,\
                                              'outlier_mask':lc_mask_regressed_outliers,\
                                              'sigma_clipping':sigma_clipping}
        results['lc_regressed_notoutlier'] = {'flux':lc_regressed_no_outliers.flux,\
                                              'time':lc_regressed_no_outliers.time}
        results['pca_used']                = {'coef':rc.coefficients,\
                                              'pc':[dm.values[:,i] for i in range(dm.rank)],\
                                              'dm':dm,\
                                              'rc':rc,\
                                              'npc':npc}

        # Save results
        results['tag'] = 'OK'
        picklefile = open(output.as_posix(), 'wb')
        pickle.dump(results, picklefile)
        picklefile.close()
        if return_msg: return OK_msg
        return
        
    except Exception as e:
        print('!!!!!!!!!!!!!!!!!!!!!!!!!')
        print(f'   Sector {sector}.')
        print('EXCEPTION:', e)
        print('!!!!!!!!!!!!!!!!!!!!!!!!!')
        if return_msg: return id_msg+'::'+repr(e)+'::'+str(e)
        return

if __name__ == '__main__':

    ### Custom runs

    print = functools.partial(print, flush=True) # Unbuffer print as default

    
    # RUN 1: Get light curves for all TPFs in the folder `tpfs` and store results in `processed`
    # outputdir = Path('processed')
    # fitsfile = Path('tess139369511_sec3.fits')
    # msg = extract_light_curve(fitsfile,outputdir)
    # print(msg)

    # RUN 2: Same as RUN 1 but skip .fits files with the characters "corrected" in its filename
    # outputdir = Path('processed')
    # fitsfile = Path('tess139369511_sec3.fits')
    # for file in inputdir.glob('*fits'):
    #     if not 'corrected' in file.stem:
    #         msg = extract_light_curve(file,outputdir)
    #         print(msg)


    # RUN 3: PARALLEL run
    outputdir = Path('processed')
    fitsfile = Path('tpfs')
        
    # Find all files that have not been processed
    inputfiles = [ f for f in inputdir.glob('*fits') ]
    donefiles  = [ f.name for f in outputdir.glob('tess*_corrected.pickled') ]
    inputfiles = [ f for f in inputfiles if f.name.replace('.fits','_corrected.pickled') not in donefiles ]
    
    num_cores = 40

    # Time execution
    time1 = time.perf_counter()
    msgs = Parallel(n_jobs=num_cores)(delayed(extract_light_curve)(file,outputdir) for file in inputfiles)
    time2 = time.perf_counter()

    # Write the log
    outputfile = outputdir/Path('output.txt')
    outputfile.touch()
    with open(outputfile.as_posix(), 'w') as f: 
        for msg in msgs: 
            f.write(msg+'\n\n\n') 
        f.write(f'seconds: {time2-time1}')
