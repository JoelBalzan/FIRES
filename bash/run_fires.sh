#sbatch --job-name=PA0 --export=ALL,PA_sd=0 fires_n_loop.sh
#sbatch --job-name=PA30 --export=ALL,PA_sd=30 fires_n_loop.sh
#sbatch --job-name=PA60 --export=ALL,PA_sd=60 fires_n_loop.sh
#sbatch --job-name=PA90 --export=ALL,PA_sd=90 fires_n_loop.sh

#sbatch --job-name=5 --export=ALL,N=5 fires_n_loop.sh
sbatch --job-name=10 --export=ALL,N=10 fires_n_loop.sh
sbatch --job-name=15 --export=ALL,N=15 fires_n_loop.sh
sbatch --job-name=20 --export=ALL,N=20 fires_n_loop.sh
sbatch --job-name=50 --export=ALL,N=50 fires_n_loop.sh
sbatch --job-name=100 --export=ALL,N=100 fires_n_loop.sh
sbatch --job-name=200 --export=ALL,N=200 fires_n_loop.sh
sbatch --job-name=600 --export=ALL,N=600 fires_n_loop.sh
sbatch --job-name=1000 --export=ALL,N=1000 fires_n_loop.sh

#sbatch --job-name=10 --export=ALL,N=10,mg_width_low=1,mg_width_high=5 fires_n_loop.sh
#sbatch --job-name=100 --export=ALL,N=100,mg_width_low=1,mg_width_high=5 fires_n_loop.sh
#sbatch --job-name=600 --export=ALL,N=600,mg_width_low=1,mg_width_high=5 fires_n_loop.sh
#sbatch --job-name=1000 --export=ALL,N=1000,mg_width_low=1,mg_width_high=5 fires_n_loop.sh

#sbatch --job-name=10 --export=ALL,N=10,mg_width_low=20,mg_width_high=40 fires_n_loop.sh
#sbatch --job-name=100 --export=ALL,N=100,mg_width_low=20,mg_width_high=40 fires_n_loop.sh
#sbatch --job-name=600 --export=ALL,N=600,mg_width_low=20,mg_width_high=40 fires_n_loop.sh
#sbatch --job-name=1000 --export=ALL,N=1000,mg_width_low=20,mg_width_high=40 fires_n_loop.sh
