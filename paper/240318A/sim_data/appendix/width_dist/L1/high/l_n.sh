#!/bin/bash

firesenv

fires -f lfrac_w20-40_240318A_mg_high -p l_frac -d . -o . -v --obs-data /mnt/Storage/craft_frb/240318A/ --obs-params /mnt/Storage/craft_frb/240318A/parameters.txt --plot-config ./plotparams.toml --phase-window first
