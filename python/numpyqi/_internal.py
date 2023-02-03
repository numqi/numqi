import os
import json

ROOTDIR = os.path.dirname(__file__)
with open(os.path.join(ROOTDIR, '_package.json'), 'r', encoding='utf-8') as fid:
    _package = json.load(fid)
