#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path
import pickle, re, functools, time
from joblib import Parallel, delayed
import lightkurve as lk 
import peakutils
from IPython import embed

def Normalize_lc(flux):
    flux -= np.median(flux)
    return flux

def summary_table_single(TIC,
                         InputDir=Path.cwd(),
                         NamePattern_Input_PickledFiles='tess{TIC}_allsectors_corrected.pickled',
                         OutputDir=Path.cwd(),
                         NamePattern_Output_StitchedLC='lc_tess{TIC}_corrected_stitched.csv',
                         single_nThreads=1):

    print(f'Threads in the prewhitening routine: {single_nThreads}')
    
    # Ensure TIC is a string (and not a number)
    if not isinstance(TIC,str):
        raise TypeError('TIC must be a string instance. Ex: TIC="349092922"')
    # Ensure InputDir is a Path instance
    if not isinstance(InputDir,Path):
        raise TypeError('InputDir must be a Path instance. Ex: outputdir=pathlib.Path.cwd()')
        # Ensure outputdir is a Path instance
    if not isinstance(OutputDir,Path):
        raise TypeError('outputdir must be a Path instance. Ex: outputdir=pathlib.Path.cwd()')
    # Ensure NamePattern_Input_PickledFiles is a string instance
    if not isinstance(NamePattern_Input_PickledFiles,str):
        raise TypeError('NamePattern_Input_PickledFiles must be a string instance containing the characters {TIC} and ending with ".pickled". Ex: "tess{TIC}_allsectors_corrected.pickled"')
    # Ensure NamePattern_Input_PickledFiles contains characters {TIC} and ends with .pickled
    else:
        if (not '{TIC}' in NamePattern_Input_PickledFiles) \
        or (not NamePattern_Input_PickledFiles.endswith('.pickled')):
            raise TypeError('NamePattern_Input_PickledFiles must be a string instance containing the characters {TIC} and ending with ".pickled" Ex: "tess{TIC}_allsectors_corrected.pickled"')
    # Ensure NamePattern_Output_StitchedLC is a string instance
    if not isinstance(NamePattern_Output_StitchedLC,str):
        raise TypeError('NamePattern_Output_StitchedLC must be a string instance containing the characters {TIC} and ending with ".csv". Ex: "lc_tess{TIC}_corrected_stitched.csv"')
    # Ensure NamePattern_Output_StitchedLC contains characters {TIC} and ends with .csv
    else:
        if (not '{TIC}' in NamePattern_Output_StitchedLC) \
        or (not NamePattern_Output_StitchedLC.endswith('.csv')):
            raise TypeError('NamePattern_Output_StitchedLC must be a string instance containing the characters {TIC} and ending with ".csv". Ex: "lc_tess{TIC}_corrected_stitched.csv"')

    # Create the output directory if needed
    if OutputDir.exists():
        if not OutputDir.is_dir():
            raise ValueError('The outputdir exist but is not a directory. It must be a directory')
    else:
        OutputDir.mkdir()

    OutputDir_LC = OutputDir/Path('lc_stitched')
    if not OutputDir_LC.exists():
        OutputDir_LC.mkdir()

    # Pickled file containing LC of all sectors
    file = InputDir/Path(NamePattern_Input_PickledFiles.format(TIC=TIC))
    # Read pickled file
    tmp = open(file.as_posix(), 'rb')
    results = pickle.load(tmp)
    tmp.close() 

    # Lists to store the results
    thr =           []
    apsize =        []
    bkg_change =    []
    contamination = []
    tags =          []
    flux =          []
    time =          []
    tpoints =       []
    # Read the results
    nsecs = 0
    for result in results:
        if result['tag'] == 'OK': 
            nsecs += 1
            thr.append(           result['aperture_threshold']                            )
            apsize.append(        result['masks']['aperture'].astype(int).sum()           )
            bkg_change.append(    result['fit']['fraction_bkg_change']                    ) 
            contamination.append( result['fit']['fraction_contamination_ap']              )
            tags.append(          result['tag']                                           )
            time.append(          result['lc_regressed_notoutlier']['time']               )
            flux.append(          Normalize_lc(result['lc_regressed_notoutlier']['flux']) ) 
            tpoints.append(       result['lc_regressed_notoutlier']['time'].size          )

    # If no sectors OK
    if nsecs ==0:
        print(f'No TESS sectors found for {TIC}: {file} ')
        return None

    # Create table to be return
    table = {}

    # Fill table        
    table['tic'] = result['tic']
    ######## Threshold for the aperture mask
    table['max_thr']    = np.max(thr)
    table['min_thr']    = np.min(thr)
    table['median_thr'] = np.median(thr)
    table['mean_thr']   = np.mean(thr)
    ######## Pixels in the aperture mask
    table['max_apsize']    = np.max(apsize)
    table['min_apsize']    = np.min(apsize)
    table['median_apsize'] = np.median(apsize)
    table['mean_apsize']   = np.mean(apsize)
    ######## Contamination from neighbour stars
    table['max_contamination']    = np.max(contamination)
    table['min_contamination']    = np.min(contamination)
    table['median_contamination'] = np.median(contamination)
    table['mean_contamination']   = np.mean(contamination)
    ######## Background change estimate
    table['max_bkg_change']    = np.max(bkg_change)
    table['min_bkg_change']    = np.min(bkg_change)
    table['median_bkg_change'] = np.median(bkg_change)
    table['mean_bkg_change']   = np.mean(bkg_change)
    ######## Number of sectors (with the tag OK, i.e., correctly processed)
    table['nsecs']   = nsecs
    ######## Number of time measurements for the stitched light curve
    table['tpoints'] = np.sum(tpoints)
    ######## Coords
    table['ra'] = result['ra']
    table['dec'] = result['dec']

    tic  = result['tic']

    try:
        table['typelabel'] = cat.query('ID == @tic')['typelabel'].item()
    except ValueError:
        table['typelabel'] = '' 
     
    return table
    
def summary_table(TICs,
                  InputDir=Path.cwd(),
                  NamePattern_Input_PickledFiles='tess{TIC}_allsectors_corrected.pickled',
                  OutputDir=Path.cwd(),
                  Name_Output_SummaryTable='summary_table.csv',
                  nThreads=1,
                  **kwargs):

    def run_summary_table_single(TIC,i,n=None,**kwargs ):
        '''Print the progress of the parallel runs and run the single version'''
        print(f'Working on {i+1}/{n}, TIC {TIC}')
        return summary_table_single(TIC, **kwargs)
    
    if isinstance(TICs,int):
        # Ensure TIC is not a number
        raise TypeError('TICs must be a string instance (ex: TIC="349092922") or an iterable of strings (ex: TICs=["349092922","55852823"])')
    
    if TICs == 'all':
        # Search fot TIC number in all files that match NamePattern_Input_PickledFiles in the InputDir
        return_TIC = lambda name: re.match(NamePattern_Input_PickledFiles.format(TIC='(\d+)'),name).group(1)
        TICs = [ return_TIC(file.name) for file in InputDir.glob(NamePattern_Input_PickledFiles.format(TIC='*'))]

    if isinstance(TICs,str):
        # If TICs is a plain string, run the single version 
        table = prewhiten_single(TICs,
                                 InputDir=InputDir,
                                 NamePattern_Input_PickledFiles=NamePattern_Input_PickledFiles,
                                 OutputDir=OutputDir,
                                 **kwargs)
        df = pd.DataFrame(table)
    else:
        # If TICs is not a plain string, ensure TICs is iterable
        try:
            _ = iter(TICs)
            del _
        except TypeError:
            raise TypeError('TICs has to be an iterable of strings. Ex: TICs=["349092922","55852823"]')
        # Run the parallel version 
        size = len(TICs)
        tmp = functools.partial(run_summary_table_single,
                                n=size,
                                InputDir=InputDir,
                                NamePattern_Input_PickledFiles=NamePattern_Input_PickledFiles,
                                OutputDir=OutputDir,
                                **kwargs)

        # Store the results for each star
        tables = Parallel(n_jobs=nThreads)( delayed(tmp)(TIC,i) for i,TIC in enumerate(TICs) )

        # Concatenate the results into a single DataFrame
        #dfs = [pd.DataFrame(table) if not table['snr'] is np.nan else pd.DataFrame(table, index=[0]) for table in tables if table is not None]
        dfs = [pd.DataFrame(table, index=[0]) for table in tables if table is not None]
        df = pd.concat(dfs)
        

    # Save
    tmp = (OutputDir/Path(Name_Output_SummaryTable)).as_posix()
    df.to_csv(tmp, index=False)
    print(f'Saved: {tmp}')
    

if __name__ == '__main__':
    

    # Use unbuffer print as default
    import functools
    print = functools.partial(print, flush=True)

    # Example of a custom run:
    
    # Directory to search corrected LCs grouped by TIC and pickled
    InputDir = Path('/STER/stefano/work/catalogs/TICv8_S-CVZ_OBAFcandidates/lc_extraction/grouped')
    # Name pattern to read images
    NamePattern_Input_PickledFiles='tess{TIC}_allsectors_corrected.pickled'
    
    # Directory to store stitched LCs and detected frequencies
    OutputDir = Path('/STER/stefano/work/catalogs/TICv8_S-CVZ_OBAFcandidates/lc_extraction')
    # Name pattern to store images
    NamePattern_Output_StitchedLC='lc_tess{TIC}_corrected_stitched.csv'
    # Name pattern to store images
    Name_Output_SummaryTable='lc_summary.csv'
    
    # TIC numbers to process
    # TICs = ['139369718', '140511383', '140528827']
    TICs = 'all'

    cat = pd.read_csv('../TICv8.S-CVZ.OBAFcandidates.csv')
    
    # Start the program
    time1 = time.perf_counter()
    summary_table(TICs,
                  InputDir=InputDir,
                  NamePattern_Input_PickledFiles=NamePattern_Input_PickledFiles,
                  OutputDir=OutputDir,
                  NamePattern_Output_StitchedLC=NamePattern_Output_StitchedLC,
                  Name_Output_SummaryTable=Name_Output_SummaryTable,
                  nThreads=1)
    time2 = time.perf_counter()
    print(f'Seconds: {time2-time1}')
