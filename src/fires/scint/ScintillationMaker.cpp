////////////////////////////////////////////////////////////////////////

// g++ -Wall -O2 -fopenmp -shared -Wl,-soname,ScintillationMaker -o ScintillationMaker.so -fPIC ScintillationMaker.cpp

#include <iostream>
#include <cmath>
#include <vector>

using namespace std;



extern "C"
void  simulate_SimpleScreen (double *E_real, double *E_im, int N_t, int N_nu, int N_th, double *nu, double *t, double *thx, double *thy, double *mu, double *ph )
{
	#define  REAL(i_t,i_nu)  E_real[(i_t)*N_nu + (i_nu)]
	#define  IMAG(i_t,i_nu)  E_im[(i_t)*N_nu + (i_nu)]
    
    // Faulty optimization, for now not used; some bug prevents correct impact of t[i_t]
    //const int N_tau = N_th*N_t;
    //vector<int> delay(N_tau);
    //#define  TAU(i_th,i_t)  delay[(i_t)*N_th + (i_th)]
    //#pragma omp parallel for
    //for (int i_th = 0; i_th < N_th; i_th++){
    //    const double thy_squared = thy[i_th]*thy[i_th];
    //    double thx_moving;
    //    const double v_thx = thx[i_th];
    //    for (int i_t = 0; i_t < N_t; i_t++){
    //        thx_moving = v_thx+t[i_t];
    //        TAU(i_th,i_t) = thx_moving*thx_moving + thy_squared;
    //    }
    //}
	
	#pragma omp parallel for
	for (int i_nu = 0; i_nu < N_nu; i_nu++){
		double Phase;
        const double v_nu = nu[i_nu];
        for (int i_t = 0; i_t < N_t; i_t++){
            double realsum = 0.;
            double imagsum = 0.;
            const double v_t = t[i_t];
            #pragma omp simd reduction(+:realsum,imagsum)
			for (int i_th = 0; i_th < N_th; i_th++){
				//Phase = ph[i_th] + v_nu* TAU(i_th,i_t);
                const double thx_moving = thx[i_th]+v_t;
                const double v_thy = thy[i_th];
                Phase = ph[i_th] + v_nu* (thx_moving*thx_moving + v_thy*v_thy);
                const double v_mu = mu[i_th];
				realsum += v_mu*cos(Phase);
				imagsum += v_mu*sin(Phase);
			}
			REAL(i_t,i_nu) = realsum;
            IMAG(i_t,i_nu) = imagsum;
		}
	}
	
	#undef  REAL
	#undef  IMAG
	#undef  TAU
} 
