import os
import shutil
import subprocess



for ind0 in range(100):
    subprocess.run(['python', '-u', 'draft00.py']) #block until subprocess finish
    shutil.copy('tbd00.pkl', f'data/tbd00_{ind0}.pkl')
