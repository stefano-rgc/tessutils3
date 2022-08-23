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
    if not sectors is None:
        if isinstance(sectors, int):
            sectors = [sectors]
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

def download_tpfs(TICs, progressbar=False, ncores=None, **kwargs):
    if isinstance(TICs, int):
        TICs = [TICs]
    _download_tpf = partial(download_tpf, **kwargs)
    with Pool(ncores) as pool:
        it = pool.imap_unordered(_download_tpf, TICs)
        if progressbar:
            it = tqdm(it, total=len(TICs))
        # Exhaust the iterator
        for _ in it:
            pass

def test_download_tpfs():
    start_t =  time.perf_counter()
    TICs = [
            293270956,
            32150270,
            349835272
            ]
    outputdir = Path('tpfs')
    sectors = np.arange(1,6)
    download_tpfs(TICs, progressbar=True, ncores=1, outputdir=outputdir, sectors=sectors)
    end_t =  time.perf_counter()
    duration = end_t - start_t
    print(f'Duration: {duration}')

def test():
    test_download_tpfs()
    
if __name__ == '__main__':
    test()
