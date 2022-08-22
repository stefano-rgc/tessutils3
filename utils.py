import re, os
from pathlib import Path
import pandas as pd

def tpf_name_pattern():
    """Default name pattern for TPFs"""
    return 'tess{TIC}_sec{SECTOR}.fits'

def validate_name_pattern(pattern):
    """Ensure pattern containing keywords for TIC and sector"""
    if pattern is None:
        pattern = tpf_name_pattern()
    else:
        try:
            if '{TIC}' in pattern and '{SECTOR}' in pattern:
                pass
        except Exception as e:
            raise ValueError('Pattern must contain keywords {TIC} and {SECTOR}.')
    return pattern

def validate_name(name):
    """Ensure that `name` contains two distinct numbers"""
    if not re.match('.*?(\d+)[^0-9]+(\d+).*?',name):
        raise ValueError('`name` must have two numbers separated by a non-number.')

def return_TIC_and_sector(name, pattern=None):
    """Return TIC and sector numbers from str containing keywords {TIC} and {SECTOR}"""
    validate_name(name)
    pattern = validate_name_pattern(pattern)
    # Substitute the keywords for regular expressions
    _pattern = pattern.format(TIC='(\d+)', SECTOR='\d+')
    TIC = re.match(_pattern,name).group(1)
    _pattern = pattern.format(TIC='\d+', SECTOR='(\d+)')
    SECTOR = re.match(_pattern,name).group(1) 
    # Return as integers
    return int(TIC), int(SECTOR) 

def return_TIC(*args, **kwargs):
    """Return TIC number from str containing keywords {TIC} and {SECTOR}"""
    return return_TIC_and_sector(*args, **kwargs)[0]

def return_sector(*args, **kwargs):
    """Return sector number from str containing keywords {TIC} and {SECTOR}"""
    return return_TIC_and_sector(*args, **kwargs)[1]

def test_return_TIC_and_sector():
    f = 'tess25152923_sec5.fits'
    r = return_sector(f)
    print(r)

def make_softlink_to_tpfs(otherdir, TICs=None, outputdir=None, pattern=None, outputpattern=None):
    """
    Recycle target TPFs already downloaded in other directory via soft links
    """   
    
    pattern = validate_name_pattern(pattern)
    outputpattern = validate_name_pattern(outputpattern)
    outputdir = Path('tpfs') if outputdir is None else Path(outputdir)
    if not outputdir.exists():
        outputdir.mkdir(parents=True)
    for filepath in Path(otherdir).glob(pattern.format(TIC='*', SECTOR='*')):
        name = filepath.name
        TIC = return_TIC(name)
        sector = return_sector(name)
        # Skip TIC if not in TICs list
        if TICs and not TIC in TICs:
            continue
        # Create the soft links for target TICs
        outputname = outputpattern.format(TIC=TIC, SECTOR=sector)
        command = f'ln -s {filepath} {outputdir/outputname}'
        os.system(command)   
    
def test_make_softlink_to_tpfs():
    TICs = [293270956,
            32150270,
            349835272]
    # TICs = None
    otherdir = '/Users/stefano/Work/IvS/lc/tutorial/ster/work/catalogs/TICv8_S-CVZ_OBAFcandidates/tpfs_paper'
    outputdir = 'tpfs_test'
    # outputpattern = '{SECTOR}_TIC{TIC}.fits'
    make_softlink_to_tpfs(otherdir, TICs=TICs, outputdir=outputdir)
    
    
def remove_TICs_from_list(TICs, dir, mode='1', pattern=None, nsectors=None, sectors=None):
    
    TICs = pd.Series(TICs, dtype='int')
    pattern = validate_name_pattern(pattern)
    dir = Path(dir)
    files = [f.name for f in dir.glob(pattern.format(TIC='*', SECTOR='*'))]
    
    # Skip TICs already downloaded 
    # (if there is already at least one downloaded sector for this TIC, this TIC is skipped)
    if mode == 1:
        TICs_remove = pd.Series(files, dtype=str).apply(return_TIC, args=(pattern,))
        TICs = pd.concat([TICs, TICs_remove]).drop_duplicates(keep=False)

    # Skip TICs with `n` sectors already downloaded
    if mode == 2:
        if sectors is None and nsectors is None:
            raise ValueError('Either `sectors` or `nsectors` must be specified.')
        df = pd.DataFrame({'name':files})
        df['tic'] = df['name'].apply(return_TIC)
        df['sec'] = df['name'].apply(return_sector)
        if sectors:
            df.query('sec not in @sectors', inplace=True)
        group = df.groupby('tic')    
        if nsectors:
            TICs_remove = pd.Series(group.count().query('sec >= @nsectors').index.tolist(), dtype=int)
        else:
            TICs_remove = pd.Series(group.count().index.tolist(), dtype=int)
        TICs = pd.concat([TICs, TICs_remove]).drop_duplicates(keep=False)

def test_remove_TICs_from_list():
    TICs = [
            293270956,
            32150270,
            349835272
            ]
    dir = 'tpfs_test'
    mode = 2
    sectors = [11]
    nsectors = 12
    remove_TICs_from_list(TICs, dir, mode=mode, sectors=sectors, nsectors=nsectors)

def test():
    # test_return_TIC_and_sector()
    # test_make_softlink_to_tpfs()
    test_remove_TICs_from_list()
    
if __name__ == '__main__':
    test()