#!/bin/bash
dir = 'jsonTiles_s2'
for f in $dir
do

    python taipanPyRouter.py -v 5 -f $f  -o RTile_Test.json
done