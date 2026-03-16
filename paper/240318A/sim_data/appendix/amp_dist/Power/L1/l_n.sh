#!/bin/fish

firesenv
fires -f lfrac_A1-P2_240318A -p l_frac -d . -o . -v --obs-data /mnt/Storage/craft_frb/240318A/ --obs-params /mnt/Storage/craft_frb/240318A/parameters.txt --plot-config ./plotparams.toml --phase-window first
