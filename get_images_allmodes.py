#!/usr/bin/env python
import re, functools, time
import numpy as np
from pathlib import Path
from joblib import Parallel, delayed
import lightkurve as lk # Installation: https://docs.lightkurve.org/about/install.html
import utils
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Pool
from tqdm import tqdm
from functools import partial

def download_tesscuts_single(TIC,
                             outputdir=Path.cwd(),
                             imsize=20,
                             overwrite=False,
                             max_tries_download=10,
                             max_tries_save=2,
                             max_tries_query=3,
                             name_pattern='tess{TIC}_sec{SECTOR}.fits',
                             onlysectors=None):
    '''
    Purpose:
        Downoad the TESS cut for all the available sectors given the TIC number
    
    Args:
        - TIC: string
            TIC number of the target star. 
        
        - outputdir: pathlib.Path
            Directory where to store the images to be downloaded.
            If the directory does not exist, it will be created
            
        - imsize: int
            Size in pixels of the square images to be downloaded.
        
        - overwrite: bool
            If True, then overwrite the FITS images.
            Note that the code actually skips the files that havealready been
            downloaded and so, ovewrite=True should not be needed.
        
        - max_tries_download: int
            Maximum number of attempts to download a same TESS sector.
            
        - max_tries_save: int
            Maximum number of attempts to save a same FITS image.
            
        - max_tries_query: int
            Maximum number of attempts to querry MAST for a particular TIC.   
            
        - name_pattern: str
           Pattern use to name the files to be saved. The five charachters
           {TIC} and the eight characters {SECTOR} are mandatory in the pattern
           and will replaced for the TIC number and sector number,
           respectively.
    '''

    # Ensure TIC is a string (and not a number)
    if not isinstance(TIC,str):
        raise TypeError('TIC must be a string instance. Ex: TIC="349092922"')
    # Ensure outputdir is a Path instance
    if not isinstance(outputdir,Path):
        raise TypeError('outputdir must be a Path instance. Ex: outputdir=pathlib.Path.cwd()')
    # Ensure imsize is an integer instance
    if not isinstance(imsize,int):
        raise TypeError('imsize must be an int instance. Ex: imsize=20')
    # Ensure name_pattern is a string instance
    if not isinstance(name_pattern,str):
        raise TypeError('name_pattern must be a string instance containing the characters {TIC} and {SECTOR}. Ex: "tess{TIC}_sec{SECTOR}.fits"')
    # Ensure name_pattern contains characters {TIC} and {SECTOR} and ends with .fits
    else:
        if (not '{TIC}' in name_pattern) \
        or (not '{SECTOR}' in name_pattern) \
        or (not name_pattern.endswith('.fits')):
            raise TypeError('name_pattern must be a string instance containing the characters {TIC} and {SECTOR}. Ex: "tess{TIC}_sec{SECTOR}.fits"')

    # Create the output directory if needed
    if outputdir.exists():
        if not outputdir.is_dir():
            raise ValueError('The outputdir exist but is not a directory. It must be a directory')
    else:
        outputdir.mkdir()

    # Search MAST for Full Frame Images availables for TIC in question
    tries_query = 1
    while True:
        if tries_query > max_tries_query:
            print(f'Skipped TIC = {TIC}: Maximum number of MAST query retries ({max_tries_query}) exceeded')
            return
        try: 
            tesscuts = lk.search_tesscut(f'TIC {TIC}')
            break
        except Exception as e:
            # If exception rised
            ename = e.__class__.__name__
            print(f'MAST query attempt {tries_query}, TIC = {TIC}. Excepion {ename}: {e}')
        # Count it as one attempt
        tries_query += 1

    if len(tesscuts) == 0:
        print(f'No images found for TIC={TIC}')
        return

    # Check that the returned ids match the TIC number
    ids = np.unique(tesscuts.table['targetid'].data)
    if not ids.size == 1:
        print(f'The MAST query returned multiple ids: {ids}')
        print('No FITS files saved')
        return
    _TIC = re.match( 'TIC (\d+)', ids.item() ).group(1)
    if TIC != _TIC:
        print(f'The MAST query returned a different id: {ids}')
        print('No FITS files saved')
        return
    
    # Get the sector numbers
    sectors = np.array([ re.match('TESS Sector (\d+)', text).group(1) for text in tesscuts.table['observation'] ])
    
    # Filter only requested sectors
    if not onlysectors is None:
        ind =[True if sec in onlysectors else False for sec in sectors.astype('int32')]
        tesscuts = tesscuts[ind]
        
    # Get the sector numbers
    sectors = np.array([ re.match('TESS Sector (\d+)', text).group(1) for text in tesscuts.table['observation'] ])
    
    # Generate the output names
    outputnames = np.array([outputdir/Path(name_pattern.format(TIC=TIC, SECTOR=s)) for s in sectors])
    
    # Skip already downloaded files
    files = np.array([file.exists() for file in outputnames])
    ind = np.argwhere(files==True).flatten()
    if len(ind) > 0:
        print(f'Skipped already downloaded sectors for TIC {TIC} = {sectors[ind]}')
        ind = np.argwhere(files==False).flatten().tolist()
        tesscuts = tesscuts[ind]
        if len(tesscuts) == 0:
            print(f'Skipped: No new images to download for TIC={TIC}')
            return

    # Download the cut target pixel files
    tries = 1
    while True:
        if tries > max_tries_download:
            print(f'Skipped TIC = {TIC}: Maximum number of retries ({max_tries_download}) exceeded')
            return
        try:
            tpfs = tesscuts.download_all(cutout_size=imsize)
            break
        except TypeError as e:
            ename = e.__class__.__name__
            print(f'Skipped TIC = {TIC}: There seems to be a problem with the requested image: Excepion {ename}: {e}.')
            return
        except Exception as e:
            # If exception rised
            ename = e.__class__.__name__
            if ename == 'SearchError':
                print(f'Skipped TIC = {TIC}: There seems to be a problem with the requested image: Excepion {ename}: {e}.')
                return
            print(f'Attempt {tries}, TIC = {TIC}. Excepion {ename}: {e}')
        # Count it as one attempt
        tries += 1

    # Save as FITS files
    for tpf in tpfs:
        # Store TIC number in the header
        tpf.header.set('TICID',value=TIC)
        sector = tpf.sector 
        outputname = outputdir/Path(name_pattern.format(TIC=TIC, SECTOR=sector))
        counter = 1
        # Attempt to write FITS file
        while True:
            if counter > max_tries_save:
                print(f'Skipped TIC = {TIC}: Maximum number of retries ({max_tries_save}) exceeded')
                return
            try:
                tpf.to_fits(outputname.as_posix(), overwrite=overwrite)
                break
            except Exception as e:
                # If exception rised
                ename = e.__class__.__name__
                print(f'Attempt {counter} when saving FITS file, TIC = {TIC}. Excepion {ename}: {e}')
                time.sleep(0.5)
                # Count it as one attempt
            counter += 1

        print(f'Saved: {outputname.as_posix()}')
    
def download_tesscuts(TICs, nThreads=1, **kwargs):
    '''
    Purpose:
        Handle the parallel runs of the single version
    
    Args:
        TICs: str | iterable of str's
            TIC number(s) of the target star(s).
            
        nThreads: int
            Numbers of parallel jobs

        kwargs:
            kwargs passed to `download_tesscuts_single()`
    
    Outputs:
        TESS cuts saved as FITS files under the respective name: f"tess{TIC}_sec{SECTOR}.fits"
        
        For instancce, TIC='2831936' consists of only 2 TESS sectors: 8 and 9. Therefore, the
        output files will be: tess2831936_sec8.fits and tess2831936_sec9.fits

    Examples:
        
        # Save images to the current work directory
        download_tesscuts('130415266')
        
        # Save images to a new folder in the home directory
        from pathlib import Path
        download_tesscuts('130415266', outputdir=Path('~/NewFolder'))

        # Do multiple TIC numbers
        download_tesscuts(['130415266','324123409'])

        # Do multiple TIC numbers in parallel
        TICs = ['130415266','324123409']
        download_tesscuts(TICs, nThreads=10)

        # Save images of 100 by 100 pixels
        download_tesscuts('130415266', imsize=100)
        
        # Save images with a custom name
        download_tesscuts(TICs, name_pattern='TICnumber{TIC}SECTORnumber{SECTOR}.fits'
    '''

    def run_download_tesscuts_single(TIC,i,n=None, **kwargs ):
        '''Print the progress of the parallel runs and run the single version'''
        print(f'Working on {i+1}/{n}, TIC = {TIC}')
        download_tesscuts_single(TIC, **kwargs)
    
    # Ensure TICs is not an int instance
    if isinstance(TICs,int):
        raise TypeError('TICs must be a string instance (ex: TIC="349092922") or an iterable of strings (ex: TICs=["349092922","55852823"])')

    if isinstance(TICs,str):
        # If TICs is a plain string, run the single version 
        download_tesscuts_single(TICs, **kwargs)
    else:
        # If TICs is not a plain string, ensure TICs is iterable
        try:
            _ = iter(TICs)
            del _
        except TypeError:
            raise TypeError('TICs has to be an iterable of strings. Ex: TICs=["349092922","55852823"]')
        # Run the parallel version 
        size = len(TICs)
        tmp = functools.partial(run_download_tesscuts_single, n=size, **kwargs)
        Parallel(n_jobs=nThreads)( delayed(tmp)(TIC,i) for i,TIC in enumerate(TICs) )

    print('After downloading images, we recommend to clean the cache images in .lightkurve-cache/tesscut')





def download_tpf(TIC,
                 imsize=20,
                 pattern=None,
                 outputdir=None,
                 max_queryTries=3,
                 max_downloadTries=10,
                 max_saveTries=2,
                 sectors=None,
                 overwrite=False):
    
    pattern = utils.validate_name_pattern(pattern)
    outputdir = Path('tpfs') if outputdir is None else Path(outputdir)
    if not outputdir.exists():
        outputdir.mkdir(parents=True)
    
    # Search MAST for all FFIs available for TIC
    tries = 1
    while True:
        if tries > max_queryTries:
            print(f'Skipped TIC={TIC}: Maximum number of MAST query retries ({max_queryTries}) exceeded.')
            return
        try: 
            tesscuts = lk.search_tesscut(f'TIC {TIC}')
            break # Exit the loop if TIC is found
        except Exception as e:
            e_name = e.__class__.__name__
            print(f'MAST query attempt {tries}, TIC = {TIC}. Excepion {e_name}: {e}')
        tries += 1

    if len(tesscuts) == 0:
        print(f'No images found for TIC={TIC}.')
        return

    # Check that there is only one returned ID
    ids = np.unique(tesscuts.table['targetid'].data)
    if not ids.size == 1:
        print(f'The MAST query returned multiple ids: {ids}')
        print('No FITS files saved')
        return
    # Check that the returned ID matches the TIC number
    if str(TIC) != re.match('TIC (\d+)',ids.item()).group(1):
        print(f'The MAST query returned a different id: {ids}')
        print('No FITS files saved')
        return
    
    # Get sector numbers
    try:
        secs = np.array([ re.match('TESS Sector (\d+)', text).group(1) for text in tesscuts.table['observation'] ])
    except KeyError:
        secs = np.array([ re.match('TESS Sector (\d+)', text).group(1) for text in tesscuts.table['mission'] ])
    # Filter only requested sectors
    if sectors:
        ind =[True if sec in sectors else False for sec in secs.astype('int32')]
        tesscuts = tesscuts[ind]
    # Get sector numbers again
    try:
        secs = np.array([ re.match('TESS Sector (\d+)', text).group(1) for text in tesscuts.table['observation'] ])
    except KeyError:
        secs = np.array([ re.match('TESS Sector (\d+)', text).group(1) for text in tesscuts.table['mission'] ])
    secs = secs.astype('int32')
    
    # Generate output names
    outputnames = np.array([outputdir/Path(pattern.format(TIC=TIC, SECTOR=sec)) for sec in secs])
    
    # Skip already downloaded files
    files = np.array([file.exists() for file in outputnames])
    ind = np.argwhere(files==True).flatten()
    if len(ind) > 0:
        skkiped_secs = ','.join(secs[ind].astype(str))
        print(f'Skipped: Already downloaded sectors for TIC {TIC}: {skkiped_secs}.')
        ind = np.argwhere(files==False).flatten().tolist()
        tesscuts = tesscuts[ind]
        if len(tesscuts) == 0:
            print(f'Skipped: No new images to download for TIC={TIC}.')
            return

    # Download TESS cut or target pixel file
    tries = 1
    while True:
        if tries > max_downloadTries:
            print(f'Skipped TIC={TIC}: Maximum number of download retries ({max_downloadTries}) exceeded.')
            return
        try:
            tpfs = tesscuts.download_all(cutout_size=imsize) # TODO: This may be a chance to use an async funtion or method
            break # Exit the loop if download is successful
        except TypeError as e:
            e_name = e.__class__.__name__
            print(f'Skipped TIC={TIC}: There seems to be a problem with the requested image. Excepion -> {e_name}: {e}.')
            return
        except Exception as e:
            # If exception rised
            e_name = e.__class__.__name__
            if e_name == 'SearchError':
                print(f'Skipped TIC = {TIC}: There seems to be a problem with the requested image. Excepion -> {e_name}: {e}.')
                return
            print(f'Download try number {tries} for TIC={TIC}. Excepion -> {e_name}: {e}')
            # ? Need to add return statement here ?
        tries += 1

    # Save as FITS files
    for tpf in tpfs:
        # Store TIC number in the header
        tpf.header.set('TICID',value=TIC)
        sector = tpf.sector 
        outputname = outputdir/Path(pattern.format(TIC=TIC, SECTOR=sector))
        tries = 1
        # Attempt to write FITS file
        while True:
            if tries > max_saveTries:
                print(f'Skipped TIC={TIC}: Maximum number of retries ({max_saveTries}) exceeded.')
                return
            try:
                tpf.to_fits(outputname.as_posix(), overwrite=overwrite)
                break # Exit the loop if save is successful
            except OSError as e:
                print('When saving FITS file for TIC={TIC}. Excepion -> OSError: {e}.')
            except Exception as e:
                e_name = e.__class__.__name__
                print(f'Attempt {tries} when saving FITS file, TIC = {TIC}. Excepion -> {e_name}: {e}.')
                time.sleep(0.5) # Allow time before next attempt
            tries += 1

        # Message for successful save
        print(f'Saved: {outputname.as_posix()}')

def download_tpfs_for_loop(TICs, **kwargs):
    for TIC in TICs:
        download_tpf(TIC, **kwargs)

def download_tpfs_map(TICs):
    _ = list(map(download_tpf, TICs))
    
def download_tpfs_threadpool(TICs):
    with ThreadPool() as pool:
        it = pool.imap_unordered(download_tpf, TICs)
        for _ in tqdm(it, total=len(TICs)):
            pass

def download_tpfs_pool_map(TICs):
    with Pool() as pool:
        it = pool.map(download_tpf, TICs)
        # for _ in tqdm(it, total=len(TICs)):
        #     pass

def download_tpfs_pool_imap(TICs):
    with Pool() as pool:
        it = pool.imap(download_tpf, TICs)
        for _ in tqdm(it, total=len(TICs)):
            pass

def download_tpfs_pool_imap_unordered(TICs):
    with Pool() as pool:
        it = pool.imap_unordered(download_tpf, TICs)
        for _ in tqdm(it, total=len(TICs)):
            pass

def download_tpfs_pool_imap_unordered_partial(TICs, progressbar=False, **kwargs):
    _download_tpf = partial(download_tpf, **kwargs)
    with Pool() as pool:
        it = pool.imap_unordered(_download_tpf, TICs)
        if progressbar:
            for _ in tqdm(it, total=len(TICs)):
                pass
        else:
            for _ in it:
                pass
            
def test_download_tpf():
    start_t =  time.perf_counter()
    TIC = 293270956
    outputdir = Path('tpfs_test2')
    download_tpf(TIC, outputdir=outputdir)
    end_t =  time.perf_counter()
    duration = end_t - start_t
    print(f'Duration: {duration}')

def test_download_tpfs_for_loop():
    start_t =  time.perf_counter()
    TICs = [
            293270956,
            32150270,
            349835272
            ]
    outputdir = Path('tpfs_test2')
    download_tpfs_for_loop(TICs, outputdir=outputdir)
    end_t =  time.perf_counter()
    duration = end_t - start_t
    print(f'Duration: {duration}')

def test_download_tpfs_map():
    start_t =  time.perf_counter()
    TICs = [
            293270956,
            32150270,
            349835272
            ]
    outputdir = Path('tpfs_test2')
    download_tpfs_map(TICs)
    end_t =  time.perf_counter()
    duration = end_t - start_t
    print(f'Duration: {duration}')

def test_download_tpfs_threadpool():
    start_t =  time.perf_counter()
    TICs = [
            293270956,
            32150270,
            349835272
            ]
    download_tpfs_threadpool(TICs)
    end_t =  time.perf_counter()
    duration = end_t - start_t
    print(f'Duration: {duration}')
    
def test_download_tpfs_pool_map():
    start_t =  time.perf_counter()
    TICs = [
            293270956,
            32150270,
            349835272
            ]
    download_tpfs_pool_map(TICs)
    end_t =  time.perf_counter()
    duration = end_t - start_t
    print(f'Duration: {duration}')

def test_download_tpfs_pool_imap():
    start_t =  time.perf_counter()
    TICs = [
            293270956,
            32150270,
            349835272
            ]
    download_tpfs_pool_imap(TICs)
    end_t =  time.perf_counter()
    duration = end_t - start_t
    print(f'Duration: {duration}')
    
def test_download_tpfs_pool_imap_unordered():
    start_t =  time.perf_counter()
    TICs = [
            293270956,
            32150270,
            349835272
            ]
    download_tpfs_pool_imap_unordered(TICs)
    end_t =  time.perf_counter()
    duration = end_t - start_t
    print(f'Duration: {duration}')

def test_download_tpfs_pool_imap_unordered_partial():
    start_t =  time.perf_counter()
    # TICs = [
    #         293270956,
    #         32150270,
    #         349835272
    #         ]
    TICs = [
            293270956,
            32150270,
            349835272,
            306631280,
            294092361,
            279510278,
            293221812
            ]
    outputdir = Path('tpfs_test3')
    download_tpfs_pool_imap_unordered_partial(TICs, progressbar=True, outputdir=outputdir)
    end_t =  time.perf_counter()
    duration = end_t - start_t
    print(f'Duration: {duration}')
    
if __name__ == '__main2__':

    # Example of a custom run:
    
    import pandas as pd
    
    # Use unbuffer print as default
    print = functools.partial(print, flush=True)
    
    # Directory where to store the TESS images
    outputdir=Path('tpfs')

    # Name pattern to store images
    name_pattern='tess{TIC}_sec{SECTOR}.fits'

    # Catalog containing the TIC numbers to download
    cat = '/STER/stefano/work/catalogs/TICv8_2+sectors/TIC_OBcandidates_FEROS_2+sectors_bright.csv'
    TICs = pd.read_csv(cat, usecols=['ID'], squeeze=True, dtype=str)
    
    # Format as a list of strings
    TICs = TICs.tolist()

    # Download only this sectors
    onlysectors = np.arange(1,14) # 1..13

    # Start the program
    download_tesscuts(TICs, outputdir=outputdir, nThreads=1, name_pattern=name_pattern,onlysectors=onlysectors)

def main():
    pass

def test():
    # test_download_tpf()
    # test_download_tpfs_for_loop()
    # test_download_tpfs_map()
    # test_download_tpfs_threadpool()
    # test_download_tpfs_pool_map()
    # test_download_tpfs_pool_imap()
    # test_download_tpfs_pool_imap_unordered()
    test_download_tpfs_pool_imap_unordered_partial()

if __name__ == '__main__':
    test()
