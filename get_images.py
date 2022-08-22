#!/usr/bin/env python
import re, functools, time
import numpy as np
from pathlib import Path
from joblib import Parallel, delayed
import lightkurve as lk # Installation: https://docs.lightkurve.org/about/install.html

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


if __name__ == '__main__':

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

    # Skip TICs already downloaded (if there is already at least one downloaded sector for this TIC, this TIC is skipped)
    #files = [file.name for file in outputdir.glob(name_pattern.format(TIC='*', SECTOR='*'))]
    #return_TIC = lambda name: re.match(name_pattern.format(TIC='(\d+)', SECTOR='\d+'),name).group(1)
    #TICs_skip = pd.Series(files, dtype=str).apply(return_TIC)
    #TICs = pd.concat([TICs, TICs_skip]).drop_duplicates(keep=False)
    
    # Recycle tpfs already downloaded in other directory   
    #otherdir = Path('/STER/stefano/work/catalogs/TICv8_S-CVZ_OBAFcandidates/tpfs')
    #filepaths = [filepath for filepath in otherdir.glob(name_pattern.format(TIC='*', SECTOR='*'))]
    #return_TIC = lambda name: re.match(name_pattern.format(TIC='(\d+)', SECTOR='\d+'),name).group(1)
    #return_SEC = lambda name: re.match(name_pattern.format(TIC='\d+', SECTOR='(\d+)'),name).group(1)
    #df = pd.DataFrame({'path':filepaths})
    #df['name'] = df['path'].apply(getattr, args=('name',))
    #df['tic'] = df['name'].apply(return_TIC)
    #df['sec'] = df['name'].apply(return_SEC).astype('int32')
    #group = df.groupby('tic')    
    ## Create the soft links
    #for TIC in TICs:
    #    if TIC in group.groups:
    #        # Make soft link
    #        for filepath in group.get_group(TIC)['path']:
    #            command = f'ln -s {filepath} {outputdir}'
    #            os.system(command)   
    
    # Skip TICs with 13 sectors already downloaded
    files = [file.name for file in outputdir.glob(name_pattern.format(TIC='*', SECTOR='*'))]
    return_TIC = lambda name: re.match(name_pattern.format(TIC='(\d+)', SECTOR='\d+'),name).group(1)
    return_SEC = lambda name: re.match(name_pattern.format(TIC='\d+', SECTOR='(\d+)'),name).group(1)
    df = pd.DataFrame({'name':files})
    df['tic'] = df['name'].apply(return_TIC)
    df['sec'] = df['name'].apply(return_SEC).astype('int32')
    df.query('sec<=13', inplace=True) #Consider only sectors 1..13
    group = df.groupby('tic')    
    # TIC of images with 13 or moer sectors downloaded
    TICs_skip = pd.Series(group.count().query('sec >= 13').index.tolist(), dtype=str)
    TICs = pd.concat([TICs, TICs_skip]).drop_duplicates(keep=False)
    del df
    
    # Format as a list of strings
    TICs = TICs.tolist()

    # Download only this sectors
    onlysectors = np.arange(1,14) # 1..13

    # Start the program
    download_tesscuts(TICs, outputdir=outputdir, nThreads=1, name_pattern=name_pattern,onlysectors=onlysectors)

def main():
    pass

if __name__ == '__main__':
    main()
