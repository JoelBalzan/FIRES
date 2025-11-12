fires -f phase_paper_a1 -p pa_var --data . -s -o . --plot-scale log --figsize 8 6 -v --use-latex --compare-windows full:leading full:trailing full:total --weight-x-by width --weight-y-by PA_i

fires -f phase_paper_a1 -p pa_var --data . -s -o . --plot-scale linear --figsize 8 6 -v --use-latex --compare-windows full:leading full:trailing full:total --weight-x-by width --weight-y-by PA_i

fires -f freq_paper_a1 -p pa_var --data . -s -o . --plot-scale log --figsize 8 6 -v --use-latex --compare-windows 1q:total 4q:total full:total --weight-x-by width --weight-y-by PA_i

fires -f freq_paper_a1 -p pa_var --data . -s -o . --plot-scale linear --figsize 8 6 -v --use-latex --compare-windows 1q:total 4q:total full:total --weight-x-by width --weight-y-by PA_i

fires -f freq_phase_paper_a1 -p pa_var --data . -s -o . --plot-scale log --figsize 8 6 -v --use-latex --compare-windows 1q:total 4q:total full:total full:leading full:trailing full:total --weight-x-by width --weight-y-by PA_i
