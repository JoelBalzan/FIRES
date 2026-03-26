#!/bin/bash

fires -f 191001_nn_ns -p lvpa --config-dir ~/Documents/GitHub/FIRES/examples/20191001A/ -v --override-param "tau=0" "propagation.scintillation.enable=false" "observation.sefd=0" "observation.target_snr=0"
fires -f 191001_nn -p lvpa --config-dir ~/Documents/GitHub/FIRES/examples/20191001A/ -v --override-param  "observation.sefd=0" "observation.target_snr=0"
fires -f 191001 -p lvpa --config-dir ~/Documents/GitHub/FIRES/examples/20191001A/ -v --override-param "observation.sefd=0" "observation.target_snr=194"
