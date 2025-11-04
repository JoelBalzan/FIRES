#!/bin/bash

mkdir /mnt/Storage/FIRES/l_frac/PA/N
cp /mnt/Storage/FIRES/PA/n1*/*.pkl /mnt/Storage/FIRES/PA/N

fires -f phase_paper_lead_N -p l_frac --data /mnt/Storage/FIRES/PA/N -s -o . --plot-scale log --figsize 8 6 -v --use-latex --obs-data /mnt/Storage/craft_frb/191001 --obs-params /mnt/Storage/craft_frb/191001/parameters.txt --equal-value-lines 4 --phase-window first --x-measured Vpsi
