#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path
import pickle, re, functools, time
from joblib import Parallel, delayed
import ivs.timeseries.freqanalyse as fa
from ivs.timeseries import pergrams

##########################################################
### Utility functions
##########################################################

def Normalize_lc(flux):
    '''Function applied to light curves of individual TESS sectors before stitching them'''
    flux -= np.median(flux)
    return flux


def check_if_iterable(x,
                      msg_if_not='It is not iterable',
                      raise_exception=True,
                      return_boolean=False):
    '''Check if `x` is an iterable'''

    # Try to create an iterator from the variable
    try:
        _ = iter(x)
        del _
        if return_boolean:
            return True
    # If not iterable, raise an exception or return False
    except TypeError:
        if raise_exception:
            raise TypeError(msg_if_not)
        elif return_boolean:
            return False

def make_outputdir(OutputDir):
    '''Create the output directory if needed'''
    if OutputDir.exists():
        if not OutputDir.is_dir():
            raise ValueError('The outputdir exist but is not a directory. It must be a directory')
    else:
        OutputDir.mkdir()


##########################################################
### Main function
##########################################################

def extract_stitched_lcs_single(TIC,
                                InputDir=Path.cwd(),
                                NamePattern_InputFile='tess{TIC}_allsectors_corrected.pickled',
                                OutputDir=Path.cwd(),
                                NamePattern_Output_StitchedLC='lc_tess{TIC}_corrected_stitched.csv'):
    '''Function extracts light curves from a single pickled file'''

    ##########################################################
    ### Checking functions
    ##########################################################

    def TIC_is_str(TIC):
        '''Check `TIC` is a string instance'''
        if not isinstance(TIC,str):
            raise TypeError('TIC must be a string instance. Ex: TIC="349092922"')

    def InputDir_is_Path(InputDir):
        '''Check `InputDir` is a Path instance'''
        if not isinstance(InputDir,Path):
            raise TypeError('InputDir must be a Path instance. Ex: outputdir=pathlib.Path.cwd()')
    
    def OnputDir_is_Path(OutputDir):
        '''Check `OutputDir` is a Path instance'''
        if not isinstance(OutputDir,Path):
            raise TypeError('outputdir must be a Path instance. Ex: outputdir=pathlib.Path.cwd()')
    
    def NamePattern_InputFile_is_str(NamePattern_InputFile):
        '''Check `NamePattern_InputFile` is a string  instance and contains characters {TIC} and ends with .pickled'''
        if not isinstance(NamePattern_InputFile,str):
            raise TypeError('NamePattern_InputFile must be a string instance containing the characters {TIC} and ending with ".pickled". Ex: "tess{TIC}_allsectors_corrected.pickled"')
    def NamePattern_InputFile_is_has_TIC_characters(NamePattern_InputFile):
        '''Check `NamePattern_InputFile` contains characters {TIC}'''
        if not '{TIC}' in NamePattern_InputFile:
            raise TypeError('NamePattern_InputFile must be a string instance containing the characters {TIC} and ending with ".pickled" Ex: "tess{TIC}_allsectors_corrected.pickled"')
    def NamePattern_InputFile_ends_in_pickled_characters(NamePattern_InputFile):
        '''Check `NamePattern_InputFile` ends with .pickled'''
        if not NamePattern_InputFile.endswith('.pickled'):
            raise TypeError('NamePattern_InputFile must be a string instance containing the characters {TIC} and ending with ".pickled" Ex: "tess{TIC}_allsectors_corrected.pickled"')
 
    def NamePattern_Output_StitchedLC_is_str(NamePattern_Output_StitchedLC):
        '''Check `NamePattern_Output_StitchedLC` is a string instance and contains characters {TIC} and ends with .csv'''
        if not isinstance(NamePattern_Output_StitchedLC,str):
            raise TypeError('NamePattern_Output_StitchedLC must be a string instance containing the characters {TIC} and ending with ".csv". Ex: "lc_tess{TIC}_corrected_stitched.csv"')
    def NamePattern_Output_StitchedLC_has_TIC_characters(NamePattern_Output_StitchedLC):
        '''Check `NamePattern_Output_StitchedLC` contains characters {TIC}'''
        if not '{TIC}' in NamePattern_Output_StitchedLC:
            raise TypeError('NamePattern_Output_StitchedLC must be a string instance containing the characters {TIC} and ending with ".csv". Ex: "lc_tess{TIC}_corrected_stitched.csv"')
    def NamePattern_Output_StitchedLC_ends_in_csv_characters(NamePattern_Output_StitchedLC):
        '''Check `NamePattern_InputFile` ends with .csv'''
        if not NamePattern_Output_StitchedLC.endswith('.csv'):
            raise TypeError('NamePattern_Output_StitchedLC must be a string instance containing the characters {TIC} and ending with ".csv". Ex: "lc_tess{TIC}_corrected_stitched.csv"')

    ##########################################################
    ### Main code
    ##########################################################

    # Check the arguments are correct
    TIC_is_str(TIC)
    InputDir_is_Path(InputDir)
    OnputDir_is_Path(OutputDir)
    NamePattern_InputFile_is_str(NamePattern_InputFile)
    NamePattern_InputFile_is_has_TIC_characters(NamePattern_InputFile)
    NamePattern_InputFile_ends_in_pickled_characters(NamePattern_InputFile)
    NamePattern_Output_StitchedLC_is_str(NamePattern_Output_StitchedLC)
    NamePattern_Output_StitchedLC_has_TIC_characters(NamePattern_Output_StitchedLC)
    NamePattern_Output_StitchedLC_ends_in_csv_characters(NamePattern_InputFile)

    # Create the output directory if needed
    make_outputdir(OutputDir)

    # Pickled file containing LC of all sectors
    file = InputDir/Path(NamePattern_InputFile.format(TIC=TIC))
  
    # Read pickled file
    with open(file.as_posix(), 'rb') as tmp:
        results = pickle.load(tmp)

    # Lists to store the results
    flux = []
    time = []

    # Read the results
    nsecs = 0
    for result in results:
        if result['tag'] == 'OK': 
            nsecs += 1
            time.append(result['lc_regressed_notoutlier']['time'])
            flux.append(Normalize_lc(result['lc_regressed_notoutlier']['flux'])) 

    # If no OK sectors 
    if nsecs ==0:
        return None

    # Get TIC value from the pickled file
    tic  = result['tic']

    # Stitch light curve
    flux = np.concatenate(flux)
    time = np.concatenate(time)
    lc = pd.DataFrame({'flux':flux,
                       'time':time})

    # Save LC as CSV file
    NameOutput_StitchedLC = Path(NamePattern_Output_StitchedLC.format(TIC=tic))
    lc.to_csv(OutputDir/NameOutput_StitchedLC, index=False)
    

##########################################################
### Wrapping of the main function
##########################################################

def extract_stitched_lcs(TICs,
                         InputDir=Path.cwd(),
                         NamePattern_InputFile='tess{TIC}_allsectors_corrected.pickled',
                         OutputDir=Path.cwd(),
                         nThreads=1,
                         **kwargs):
    '''Function prepares input for parallel run'''

    def run_extract_stitched_lcs_single(TIC,i,n=None,**kwargs ):
        '''Print the progress of the parallel runs and run the single version'''
        print(f'Working on {i+1}/{n}, TIC {TIC}')
        return extract_stitched_lcs_single(TIC, **kwargs)
    
    # Ensure TIC is not a number
    if isinstance(TICs,int):
        raise TypeError('TICs must be a string instance (ex: TIC="349092922") or an iterable of strings (ex: TICs=["349092922","55852823"])')

    # Search for TIC number in all files that match NamePattern_InputFile in the InputDir
    if TICs == 'all':
        return_TIC = lambda name: re.match(NamePattern_InputFile.format(TIC='(\d+)'),name).group(1)
        TICs = [ return_TIC(file.name) for file in InputDir.glob(NamePattern_InputFile.format(TIC='*'))]

    # If TICs is a plain string, run the single version of this call
    if isinstance(TICs,str):
        extract_stitched_lcs_single(TICs,
                                    InputDir=InputDir,
                                    NamePattern_InputFile=NamePattern_InputFile,
                                    OutputDir=OutputDir,
                                    **kwargs)
        df = pd.DataFrame(table)
    # If TICs is not a plain string, ensure TICs is iterable
    else:
        check_if_iterable(TICs, msg_if_not='TICs has to be an iterable of strings. Ex: TICs=["349092922","55852823"]')
        # Run the parallelized version 
        Parallel(n_jobs=nThreads)( delayed(run_extract_stitched_lcs_single)(TIC,
                                                                            i,
                                                                            n=len(TICs),
                                                                            InputDir=InputDir,
                                                                            NamePattern_InputFile=NamePattern_InputFile,
                                                                            OutputDir=OutputDir,
                                                                            **kwargs) for i,TIC in enumerate(TICs) )


##########################################################
### Use examples
##########################################################
    
if __name__ == '__main__':
    

    # Use unbuffer print as default
    import functools
    print = functools.partial(print, flush=True)

    # Example of a custom run:
    
    # Directory where to search by TIC the pickled files containing groups of corrected LCs 
    InputDir = Path('grouped')
    # Name pattern to read the pickled files
    NamePattern_InputFile='tess{TIC}_allsectors_corrected.pickled'
    
    # Directory to store stitched LCs
    OutputDir = Path('lcs_test')
    # Name pattern to store images
    NamePattern_Output_StitchedLC='lc_tess{TIC}_corrected_stitched.csv'
    
    # TIC numbers to process
    #TICs = 'all'
    #TICs = ['38828023', '29830105']
    TICs = pd.read_csv('TICs_stitched_10+sec_4+pix.list', header=None, names=['TIC']).TIC.astype('str').tolist()
    
    # Start the program
    extract_stitched_lcs(TICs,
                         InputDir=InputDir,
                         NamePattern_InputFile=NamePattern_InputFile,
                         OutputDir=OutputDir,
                         NamePattern_Output_StitchedLC=NamePattern_Output_StitchedLC,
                         nThreads=1)
