import re

def return_TIC(name):
    re.match(name_pattern.format(TIC='(\d+)', SECTOR='\d+'),name).group(1)    
    
return_TIC = lambda name: re.match(name_pattern.format(TIC='(\d+)', SECTOR='\d+'),name).group(1)
return_SEC = lambda name: re.match(name_pattern.format(TIC='\d+', SECTOR='(\d+)'),name).group(1)
