#!/bin/bash

fires -f 240318A -p lvpa --ncpu 12 --config-dir ~/Documents/GitHub/FIRES/paper/240318A/ -v  --seed 0 --sefd 1.4 --scint --override-param "tau=0.128" -o /home/joel/Documents/GitHub/FIRES/paper/240318A/
fires -f 240318A_nn -p lvpa --ncpu 12 --config-dir ~/Documents/GitHub/FIRES/paper/240318A/ -v  --seed 0 --scint --override-param "tau=0.128" -o /home/joel/Documents/GitHub/FIRES/paper/240318A/
fires -f 240318A_nn_ns -p lvpa --ncpu 12 --config-dir ~/Documents/GitHub/FIRES/paper/240318A/ -v  --seed 0 --override-param "tau=0" -o /home/joel/Documents/GitHub/FIRES/paper/240318A/

fires -f 240318A_sd10 -p pa --ncpu 12 --config-dir ~/Documents/GitHub/FIRES/paper/240318A/ -v  --seed 0 --sefd 1.4 --scint --override-param "tau=0.128" -o /home/joel/Documents/GitHub/FIRES/paper/240318A/
fires -f 240318A_sd10nn -p pa --ncpu 12 --config-dir ~/Documents/GitHub/FIRES/paper/240318A/ -v  --seed 0 --scint --override-param "tau=0.128" -o /home/joel/Documents/GitHub/FIRES/paper/240318A/

fires -f 240318A_sd0 -p pa --ncpu 12 --config-dir ~/Documents/GitHub/FIRES/paper/240318A/ -v  --seed 0 --sefd 1.4 --scint --override-param "tau=0.128" "sd_PA=0" -o /home/joel/Documents/GitHub/FIRES/paper/240318A/
fires -f 240318A_sd0nn -p pa --ncpu 12 --config-dir ~/Documents/GitHub/FIRES/paper/240318A/ -v  --seed 0 --scint --override-param "tau=0.128" "sd_PA=0" -o /home/joel/Documents/GitHub/FIRES/paper/240318A/