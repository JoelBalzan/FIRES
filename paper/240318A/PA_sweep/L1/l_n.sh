#!/bin/bash

fires -f phase_paper_240318A_L1 -p l_frac -d /mnt/Storage/FIRES/240318A/PA/L1/N/ -o . -v --obs-data /mnt/Storage/craft_frb/240318A/ --obs-params /mnt/Storage/craft_frb/240318A/parameters.txt --plot-config ./plotparams.toml --phase-window first
