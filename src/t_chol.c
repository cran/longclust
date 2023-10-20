#include <stdio.h>
#include <stdlib.h>
#include <R.h>
#include <Rmath.h>
#include "zeroin.h"
//#ifdef __APPLE__
//#include <Accelerate/Accelerate.h>
//#else
#include <Rconfig.h> // for PR18534fixed
//#ifdef PR18534fixed
//# define usePR18534fix 1
//#endif
#include <R_ext/Lapack.h>
#include <R_ext/BLAS.h>
typedef int __CLPK_integer;
//#endif

#define COMMENTS 0  // set this to 1 to print lots of intermediate values.

void copyvec(double *A, int n, double *C) {
  int i;
	for(i=0; i < n; i++)
	  C[i] = A[i];
}

void printmx(double *A, int r, int c) {
    int i, j;
	for(i=0; i < r; i++) {
		for(j=0; j < c; j++)
			Rprintf("%G ", A[i + r*j]);
		Rprintf("\n");
	}
	Rprintf("\n");
}


void copymx(double *A, int r, int c, int lda, double *C) {
  int i,j;
  for(i=0; i < r; i++) {
	for(j=0; j < c; j++)
		C[i + j*r] = A[i + j*lda];
  }
}

// compute the reverse condition number of the matrix A.
int determinant(double *A, __CLPK_integer k, __CLPK_integer lda, double *res) {
	int i;
	__CLPK_integer *ipiv;
	double *C;
	__CLPK_integer info=0;
	
	C = (double*)malloc(sizeof(double)*k*k);
	copymx(A, k, k, lda, C);
	
	ipiv = (__CLPK_integer *)malloc(sizeof(__CLPK_integer)*k);
	
	dgetrf_(&k, &k, C, &k, ipiv, &info);
	
	if(COMMENTS && info != 0) 
		Rprintf("Failed in computing matrix determinant.\n");
	
	*res=1.0;
	for(i=0; i < k*k; i++) {
	  if(i%k == i/k) 
		  (*res) *= C[i];
	}

	if(*res < 0)	// computing absolute value of the determinant
	  *res = -(*res);
	 
	free(ipiv);
	free(C);
	
	return info;
}

// compute the reverse condition number of the matrix A.
int rcond(double *A, __CLPK_integer k, __CLPK_integer lda, double *res) {
	__CLPK_integer *ipiv, *iwork; 
	double *work, *C;
	double anorm, result;
	char norm = '1';
	__CLPK_integer info=0;
	
	C = (double*)malloc(sizeof(double)*k*k);
	copymx(A, k, k, lda, C);
	
	ipiv = (__CLPK_integer *)malloc(sizeof(__CLPK_integer)*k);
	iwork = (__CLPK_integer *)malloc(sizeof(__CLPK_integer)*k);
	work = (double*)malloc(sizeof(double)*4*k);

	anorm = dlange_(&norm, &k, &k, C, &k, work,1);
	dgetrf_(&k, &k, C, &k, ipiv, &info);
	
	if(COMMENTS && info != 0) 
		Rprintf("Failed in computing matrix factorization.\n");
	else {
	  dgecon_(&norm, &k, C, &k, &anorm, &result, work, iwork, &info,1);
	
	  if(COMMENTS && info != 0) 
		  Rprintf("Failed in computing the reciprocal of the condition number.\n");
		else
			*res = result;
	}
	
	free(ipiv);
	free(work);
	free(iwork);
	free(C);
	
	return info;
}

// compute the pseudo-inverse of the matrix A and store it in B
int ginv(__CLPK_integer n, __CLPK_integer lda, double *A, double *B) {
	int i;
	__CLPK_integer info;
	__CLPK_integer NRHS = n;
	__CLPK_integer LDWORK = -1;
	__CLPK_integer LDB= n;
	__CLPK_integer IRANK = -1;
	double RCOND = -1.0f;
	double *WORK = (double *)malloc(sizeof(double));
	double *sing = (double *)malloc(sizeof(double)*n);
	double *C = (double *) malloc(sizeof(double)*n*n);
	
	// initialize B
	for(i=0; i < n*n; i++) {
		if(i/n == i%n )
			B[i] = 1.0;
		else
			B[i] = 0.0;
	}
	
	copymx(A, n, n, lda, C);
	
	dgelss_(&n, &n, &NRHS, C, &n, B, &LDB, sing, &RCOND, &IRANK, WORK, &LDWORK, &info);
	if(COMMENTS && info != 0) {
		Rprintf("Failed in computing pseudo inverse.\n");
	} else {
		LDWORK = WORK[0];
	  free(WORK);
	  WORK = (double *)malloc(sizeof(double)*LDWORK);
	  dgelss_(&n, &n, &NRHS, C, &n, B, &LDB, sing, &RCOND, &IRANK, WORK, &LDWORK, &info);
	
	  if(COMMENTS && info != 0) {
		  Rprintf("Failed in computing pseudo inverse.\n");
	  }
	}
	free(WORK);
	free(sing);
	free(C);
	
	return info;
}

int isNan(long double *A, int r, int c) {
 int i;
 for(i=0; i < r*c; i++) {
   if(A[i] != A[i])
	   return 1;
 }
 return 0;
}

void printlongdoublemx(long double *A, int r, int c) {
    int i, j;
	for(i=0; i < r; i++) {
		for(j=0; j < c; j++)
			Rprintf("%Lf ", A[i + r*j]);
		Rprintf("\n");
	}
	Rprintf("\n");
}


int cholest(__CLPK_integer M, double *Sig, double *That, double *Dhat, int isotropic) {
	int r, i, j;
	int flag = 0, error = 0;
	double *B = (double *)malloc(sizeof(double)*M*M);
	double tol_sing = 1.490116/100000000;
	double alpha = 1.0, bt=0.0;
	double mean = 0.0, rcnumber;
//#ifndef __APPLE__
	char notrans = 'N';
	char trans = 'T';
//#endif
	
	for(i=0; i < M*M; i++) { // initialize 
		if( i/M == i%M)
			That[i] = 1.0;
		else
			That[i] = 0.0;
	}
	
	for(r = 2; r <= M; r++) {
		if(r > 2) {
			if(rcond(Sig, r-1, M, &rcnumber) !=0 || rcnumber < tol_sing) {
			  if(COMMENTS)
			    Rprintf("Failed to compute rcond number for Sig[0:%d][0:%d]\n", r-1, r-1);
			  flag = 1;
			}
		}
		if(flag == 0) {
		    ginv(r-1, M, Sig, B);			// what if ginv returns an error ?
			
//			Rprintf("ginv(Sig[1:%d,1:%d])\n", r-1, r-1);
//			printmx(B, 2);
			// That[r-1, 0:(r-2)] = ginv(Sig, r-1, M) %*% Sig[r-1, 0:(r-2)].
			for(i = 0; i <= r-2; i++) {
				That[r-1 + i*M] = 0.0;
				for(j = 0; j <= r-2; j++) {
					That[r-1 +i*M] -= B[i+j*(r-1)]* Sig[r-1 + j*M];  // Assuming that Sig has M rows..
				}
			}
		}
	}
	
	// B is free.
/*#ifdef __APPLE__
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, M, M, alpha, That, M, Sig, M, bt, B, M); // B = That * Sig.
	cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, M, M, M, alpha, B, M, That, M, bt, Dhat, M); // Dhat = B *t(That).
#else*/
  dgemm_(&notrans, &notrans, &M, &M, &M, &alpha, That, &M, Sig, &M, &bt, B, &M,1,1); // B = That * Sig.
  dgemm_(&notrans, &trans, &M, &M, &M, &alpha, B, &M, That, &M, &bt, Dhat, &M,1,1); // Dhat = B *t(That).
//#endif
	
	// compute mean of the diagonal.
	for(i=0; i < M*M; i++) {
		if(i/M == i%M)
			mean += Dhat[i];
	}
	mean /= M;
	
	//compute Dhat
	for(i=0; i < M*M; i++) {
		if((i/M == i%M)) { 
			if(isotropic) 
				Dhat[i] = mean;
		} else {
			Dhat[i] = 0.0;
		}
	}
	
	// B and Sig are free. First compute B=ginv(Dhat), then B = t(That)*B*That, then Sig = ginv(B).
	ginv(M, M, Dhat, B);
	
/*#ifdef __APPLE__
	cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, M, M, M, alpha, That, M, B, M, bt, Sig, M);
	cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, M, M, alpha, Sig, M, That, M, bt, B, M);
#else*/
  dgemm_(&trans, &notrans, &M, &M, &M, &alpha, That, &M, B, &M, &bt, Sig, &M,1,1);
  dgemm_(&notrans, &notrans, &M, &M, &M, &alpha, Sig, &M, That, &M, &bt, B, &M,1,1);
//#endif
	ginv(M, M, B, Sig);
	
	if(flag == 1 || rcond(B, M, M, &rcnumber) != 0 || rcnumber < tol_sing )
		error = 1;
	
	free(B);
	return error;
}

int cholest2(__CLPK_integer M, double **K, double *Sig, double *That, double *Dhat, int isotropic) {
	int r, i, j;
	int flag = 0, error = 0;
	double *B = (double *)malloc(sizeof(double)*M*M);
	double tol_sing = 1.490116/100000000;
	double alpha = 1.0, bt=0.0;
	double mean = 0.0, rcnumber;
//#ifndef __APPLE__
	char notrans = 'N';
	char trans = 'T';
//#endif
	
	for(i=0; i < M*M; i++) { // initialize 
		if( i/M == i%M)
			That[i] = 1.0;
		else
			That[i] = 0.0;
	}
	
	for(r = 2; r <= M; r++) {
		if(r > 2) {
			if(rcond(K[r-1], r-1, M, &rcnumber) !=0 || rcnumber < tol_sing)
			  flag = 1;
		}
		if(flag == 0) {
		    ginv(r-1, M, K[r-1], B);			// what if ginv returns an error ?
			
			// That[r-1, 0:(r-2)] = ginv(Sig, r-1, M) %*% Sig[r-1, 0:(r-2)].
			for(i = 0; i <= r-2; i++) {
				That[r-1 + i*M] = 0.0;
				for(j = 0; j <= r-2; j++) {
					That[r-1 +i*M] -= B[i+j*(r-1)]* K[r-1][r-1 + j*M];  // Assuming that Sig has M rows..
				}
			}
		}
	}
	
	// B is free.
/*#ifdef __APPLE__
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, M, M, alpha, That, M, Sig, M, bt, B, M); // B = That * Sig.
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, M, M, M, alpha, B, M, That, M, bt, Dhat, M); // Dhat = B *t(That).
#else*/
  dgemm_(&notrans, &notrans, &M, &M, &M, &alpha, That, &M, Sig, &M, &bt, B, &M,1,1); // B = That * Sig.
  dgemm_(&notrans, &trans, &M, &M, &M, &alpha, B, &M, That, &M, &bt, Dhat, &M,1,1); // Dhat = B *t(That).
//#endif
	
	// compute mean of the diagonal.
	for(i=0; i < M*M; i++) {
		if(i/M == i%M)
			mean += Dhat[i];
	}
	mean /= M;
	
	//compute Dhat
	for(i=0; i < M*M; i++) {
		if((i/M == i%M)) { 
			if(isotropic) 
				Dhat[i] = mean;
		} else {
			Dhat[i] = 0.0;
		}
	}
	
	// B and Sig are free. First compute B=ginv(Dhat), then B = t(That)*B*That, then Sig = ginv(B).
	ginv(M, M, Dhat, B);
/*#ifdef __APPLE__
  cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, M, M, M, alpha, That, M, B, M, bt, Sig, M);
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, M, M, alpha, Sig, M, That, M, bt, B, M);
#else*/
  dgemm_(&trans, &notrans, &M, &M, &M, &alpha, That, &M, B, &M, &bt, Sig, &M,1,1);
  dgemm_(&notrans, &notrans, &M, &M, &M, &alpha, Sig, &M, That, &M, &bt, B, &M,1,1);
//#endif
	ginv(M, M, B, Sig);
	
	if(flag == 1 || rcond(B, M, M, &rcnumber) != 0 || rcnumber < tol_sing )
		error = 1;
	
	free(B);
	return error;
}

void mahalanobis(int g, int N, int p, double *x, int G, double *mu, double *cov, double *delta) {
	int i, j, n;
	double sum, insum;
  double *inv = (double *) malloc(sizeof(double)*p*p);
	ginv(p, p, cov, inv);

	for(n=0; n < N; n++) {
		sum = 0.0;
	  for(j=0; j < p; j++) {
			insum = 0.0;
			for(i=0; i < p; i++)
				insum += (x[n+ N*i]-mu[g+ G*i])*inv[i+j*p];
			sum += insum*(x[n+ N*j]- mu[g +G*j]);
	  }
    delta[n+ N*g] = sum;
	}
	free(inv);
}

void computeWeightedCovariance(int N, int p, int G, double *x, double *z, double *W, double *mu, int g, int wflag, double *sampcov) {
  int j,k,i;
	double sum, sum2, wsum, wsum2;
	double fac = 1.0;
	double *normWt = (double*)malloc(sizeof(double)*N);

  sum=0; 
	for(i=0; i < N; i++) {
	  if(wflag)
		  fac = W[i+N*g];
		normWt[i] = fac*z[i + N*g];
		sum  += normWt[i];
	}
	for(i=0; i < N; i++) 
	  normWt[i] /= sum;

	for(j=0; j < p; j++) {
	  for(k=0; k < p; k++) {
	  	sum  = 0.0;
	  	sum2 = 0.0;
			wsum = 0.0;
			wsum2 = 0.0;
			sampcov[j + k*p] = 0;
	  	for(i=0; i < N; i++) {
	    	sampcov[j + k*p] += normWt[i]*(x[i+N*j]-mu[g+G*j])*(x[i+N*k]-mu[g+G*k]);
		    wsum += normWt[i];
		    wsum2 += normWt[i]*normWt[i];
				if(wflag) {
		      sum += z[i + N*g];
		      sum2 += z[i + N*g]*W[i + N*g];
				}
		  }
			sampcov[j + k*p] *= wsum;
			sampcov[j + k*p] /= (wsum*wsum - wsum2);
			if(wflag) {
		    sampcov[j + k*p] *= sum2;
		    sampcov[j + k*p] /= sum;
			}
		}
  }
	free(normWt);
}

double maximum(double *A, int size) {
  int i;
	double max = A[0];
	for(i=0; i < size; i++) {
	  if(max < A[i])
		  max= A[i];
	}
	return max;
}

void t_common(int domeans, double *x, int *classify, int N, int p, int G_min, int G_max, int df_flag, int gaussian_flag, int *models_selected, double TOL, int flag_kmeans, double *kmeansz, double *time1, int *Gbest, int *Gbest_icl, double *zbest, double *zbest_icl, double *nubest, double *nubest_icl, double *mubest, double *mubest_icl, double *Tbest, double *Tbest_icl, double *Dbest, double *Dbest_icl, double *llres, double *bicres, double *iclres, double *betabest, double *betabest_icl);

void t_chol(double *x, int *classify, int N, int p, int G_min, int G_max, int df_flag, int gaussian_flag, int *models_selected, double TOL, int flag_kmeans, double *kmeansz, int *Gbest, int *Gbest_icl, double *zbest, double *zbest_icl, double *nubest, double *nubest_icl, double *mubest, double *mubest_icl, double *Tbest, double *Tbest_icl, double *Dbest, double *Dbest_icl, double *llres, double *bicres, double *iclres) {

	t_common(0, x, classify, N, p, G_min, G_max, df_flag, gaussian_flag, models_selected, TOL, flag_kmeans, kmeansz, NULL, Gbest, Gbest_icl, zbest, zbest_icl, nubest, nubest_icl, mubest, mubest_icl, Tbest, Tbest_icl, Dbest, Dbest_icl, llres,bicres, iclres, NULL, NULL);
}

void t_chol_means(double *x, int *classify, int N, int p, double *time1, int G_min, int G_max, int df_flag, int gaussian_flag, int *models_selected, double TOL, int flag_kmeans, double *kmeansz, int *Gbest, int *Gbest_icl, double *zbest, double *zbest_icl, double *nubest, double *nubest_icl, double *mubest, double *mubest_icl, double *Tbest, double *Tbest_icl, double *Dbest, double *Dbest_icl, double *llres, double *bicres, double *iclres) {

	double *betabest  = (double *)malloc(sizeof(double)*G_max*2);
	double *betabest_icl  = (double *)malloc(sizeof(double)*G_max*2);
	
	t_common(1, x, classify, N, p, G_min, G_max,df_flag, gaussian_flag, models_selected, TOL, flag_kmeans, kmeansz, time1, Gbest, Gbest_icl,  zbest, zbest_icl, nubest, nubest_icl, mubest, mubest_icl, Tbest, Tbest_icl, Dbest, Dbest_icl,llres,bicres, iclres, betabest, betabest_icl);

	free(betabest);
	free(betabest_icl);
}

void t_common(int domeans, double *x, int *classify, int N, int p, int G_min, int G_max, int df_flag, int gaussian_flag, int *models_selected, double TOL, int flag_kmeans, double *kmeansz, double *time1, int *Gbest, int *Gbest_icl, double *zbest, double *zbest_icl, double *nubest, double *nubest_icl, double *mubest, double *mubest_icl, double *Tbest, double *Tbest_icl, double *Dbest, double *Dbest_icl, double *llres, double *bicres, double *iclres, double *betabest, double *betabest_icl) {
	double tol_sing = 1.490116/100000000;
	double alpha = 1.0, bt=0.0;
  double MYPI = 3.141593;
//#ifndef __APPLE__
	char notrans = 'N';
	char trans = 'T';
//#endif

	int params[8];
	int extra = 1, fac;
	double mindouble = -(double)(1 << 30); 
	double maxdouble =  (double)(1 << 30); 

	int i,j,k,G;
	int iso, classify_flag; // flag.
	double prob, max, dfnew, dfold, ak, linf, bicval, iclval, penalty, rcnumber, detvalue;
	long double sum, sum2;

	int modelsLength = 8, it_max = 10000, it, conv, mIndex, g;
  char *models[] = {"VVI", "VVA", "EEI", "EEA", "VEI", "VEA", "EVA", "EVI"};
	char *modelName;
  
	double *logl       = (double *)malloc(sizeof(double)*it_max);
  double *df         = (double *)malloc(sizeof(double)*G_max);
	double *n          = (double *)malloc(sizeof(double)*G_max);
	double *pi         = (double *)malloc(sizeof(double)*G_max);


	double *z          = (double *)malloc(sizeof(double)*N*G_max);

	double *mu         = (double *)malloc(sizeof(double)*G_max*p);

	long double *num     = (long double *)malloc(sizeof(long double)*N*G_max);
	double *log_num = (double *)malloc(sizeof(double)*N*G_max);
	double *W       = (double *)malloc(sizeof(double)*N*G_max);
	double *delta   = (double *)malloc(sizeof(double)*N*G_max);
	double *MAPz    = (double *)malloc(sizeof(double)*N*G_max);
	double *avg     = (double *)malloc(sizeof(double)*p*p);
	double *beta    = (double *)malloc(sizeof(double)*G_max*2);


	double **Sigma   = (double**)malloc(sizeof(double*)*G_max);
	double **sampcov = (double**)malloc(sizeof(double*)*G_max);

	double **T         = (double**)malloc(sizeof(double*)*G_max);

	double **D         = (double**)malloc(sizeof(double*)*G_max);

	double **kappa   = (double**)malloc(sizeof(double*)*p);

	double *time2   = (double *)malloc(sizeof(double)*2*p);
	double *dummy1  = (double *)malloc(sizeof(double)*2*2);
	double *dummy2  = (double *)malloc(sizeof(double)*2*2);
	double *dummy   = (double *)malloc(sizeof(double)*p*p);

	*Gbest = 0;
	*Gbest_icl = 0;

  if(COMMENTS)
    Rprintf("Inside the main function!\n");

	for(g=0; g < G_max; g++) {
	  Sigma[g]     = (double *)malloc(sizeof(double)*p*p);
	  sampcov[g]   = (double *)malloc(sizeof(double)*p*p);
	  T[g]         = (double *)malloc(sizeof(double)*p*p);
	  D[g]         = (double *)malloc(sizeof(double)*p*p);
	}

	for(i=0; i < p; i++)
	  kappa[i] = (double *)malloc(sizeof(double)*p*p);

	for(i=0; i < modelsLength; i++) {
		for(j=0; j < G_max; j++) {
			llres[i + j*modelsLength] = mindouble;
			bicres[i + j*modelsLength] = mindouble;
			iclres[i + j*modelsLength] = mindouble;
		}
	}

  classify_flag = 0;
	for(i=0; i < N; i++) {
	  if(classify[i] != 0)
		  classify_flag = 1;
	}

	GetRNGstate();
  for(G = G_min; G < G_max; G++) {
		  if(COMMENTS)
		    Rprintf("G = %d\n",G);
			if(df_flag)
			  extra=1;
			else
			  extra = G;
			if(domeans == 1)
			  fac = 2;
			else
			  fac = p;
      params[0] = extra + G*fac+(G-1)+G*p*(p-1)/2+G;
      params[1] = extra + G*fac+(G-1)+G*p*(p-1)/2+G*p;
      params[2] = extra + G*fac+(G-1)+p*(p-1)/2+1;
      params[3] = extra + G*fac+(G-1)+p*(p-1)/2+p;
      params[4] = extra + G*fac+(G-1)+G*p*(p-1)/2+1;
      params[5] = extra + G*fac+(G-1)+G*p*(p-1)/2+p;
      params[6] = extra + G*fac+(G-1)+p*(p-1)/2+G*p;
      params[7] = extra + G*fac+(G-1)+p*(p-1)/2+G;

      if(COMMENTS) {
			  Rprintf("\nx:");
			  printmx(x, N, p);
			}

	    for(mIndex =0; mIndex < modelsLength-2; mIndex++) {
		    if(!models_selected[mIndex]) // If this model is not selected, skip it.
				  continue;
	      prob = 0;
	    	modelName = models[mIndex];
				if(COMMENTS) {
  	      Rprintf("######################################");
  	      Rprintf("######################################");
  			  Rprintf("mIndex = %d, modelName=%s\n", mIndex, modelName);
  	      Rprintf("######################################");
  	      Rprintf("######################################");
				}
	    	if(modelName[2] == 'I')
		    	iso = 1;
	    	else
		    	iso = 0;

        for(g=0; g < G; g++)
	    	  n[g] = 0.0;

		    for(i=0; i < N*G; i++)
		      z[i] = 0;
    
				if(flag_kmeans == 1) {
	        g = G-G_min;
					for(i=0; i < N; i++) {
						j = kmeansz[i + g*N]-1;
						z[i +j*N] = 1;
					}
				} 
/*			else if(domeans && start == 3) {
	          g = G-G_min;
            t_common(0, x, classify, N, p, G, G+1, df_flag, models_selected, TOL,  0, NULL, NULL, Gbest, Gbest_icl, z, zbest_icl, nubest, nubest_icl, mubest, mubest_icl, Tbest, Tbest_icl, Dbest, Dbest_icl, llres,bicres, iclres, NULL, NULL);
				} */
				else if(classify_flag == 0) {
  	    	for(i=0; i < N; i++) {
		        sum = 0.0;
			      for(g=0; g < G; g++) {
	    		    z[i + g*N] = unif_rand();
	    			  sum += z[i + g*N];
  	    		}
  	    		for(g=0; g < G; g++)
  		    	  z[i + g*N]/=sum;
  	    	}
				}
				else {
				  for(i=0; i < N; i++) {
					  g = classify[i];
						if(g > 0) {
						  g--;
							z[i + g*N] = 1.0;
							continue;
						}
						for(g=0; g < G; g++)
						  z[i + g*N] = 1.0/(double)G;
					}
				}

				for(g=0; g < G; g++) {
				  n[g] = 0;
					for(i=0; i < N; i++)
					  n[g] += z[i + g*N];
				}

	    	for(g=0; g < G; g++)
	    		pi[g] = n[g]/N;
	  
	    	for(i=0; i < G*p; i++)
		      mu[i] = 0.0;
				for(i=0; i < G*2; i++)
					beta[i] = 0.0;

	    	for(i=0; i < N*G; i++) {
	    		num[i]     = 0.0;
	    		log_num[i] = 0.0;
	    		W[i]       = 0.0;
	    		delta[i]   = 0.0;
	    	}

	    	for(g=0; g < G; g++) {
          for(i=0; i < p*p; i++) {
	    		  Sigma[g][i] = 0.0;
	    			sampcov[g][i] = 0.0;
	    			T[g][i] = 0.0;
	    			D[g][i] = 0.0;
	    			if(i/p == i%p)
	    			  D[g][i] = 1.0;
	    		}
	    	}

	    	for(i=0; i < it_max; i++)
	    	  logl[i] = 0;
	    	for(i=0; i < G; i++)
	    	  df[i] = 50;

        it = 1;
	    	conv = 0;

				for(g=0; g < G; g++) {
				  for(j=0; j < p; j++) {
						mu[g + j*G] = 0.0;
						for(i=0; i < N; i++)
							mu[g + j*G] += x[i + j*N]*z[i + g*N];
						mu[g + j*G] /= n[g];
					}
				}


				for(g=0; g < G; g++) {
          computeWeightedCovariance(N, p, G, x, z, W, mu, g, 0, sampcov[g]);
					mahalanobis(g, N, p, x, G, mu, sampcov[g], delta);
					for(i=0; i < N; i++)
						W[i +N*g] = (df[g]+p)/(df[g]+delta[i +N*g]);
				}

				for(i=0; i < N; i++)
					for(j=0; j < G; j++)
								MAPz[i + N*j] = 0;

        // The Algorithms
        if(COMMENTS) {
				  Rprintf("About to enter the main while loop \n");
			    Rprintf("\nz:\n");
				  printmx(z, N, G);
  			  Rprintf("\nW:\n");
	  			printmx(W, N, G);
		  		Rprintf("\nmu:\n");
			  	printmx(mu, G, p);
				}
				while(conv != 1) {
						  
          for(g=0; g < G; g++)
	    	    n[g] = 0.0;
					for(i=0; i < N; i++)
					  for(g=0; g < G; g++)
						  n[g] += z[i + N*g];
					for(g=0; g < G; g++)
					  pi[g] = n[g]/N;

          // update mu.
					for(g =0; g < G; g++) { 
					  sum2=0;
						for(i=0; i < N; i++)
						  sum2 += z[i + N*g]*W[i + N*g];
						for(j=0; j < p; j++) {
						  sum=0; 
							for(i=0; i < N; i++)
							  sum += z[i +N*g]*W[i +N*g]*x[i +N*j];
							mu[g + G*j] = sum/sum2;
						}
						if(domeans) {
  						k=2;
//#ifdef __APPLE__
//  				  cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, k, k, p, alpha, time1, p, time1, p, bt, dummy1, k);
//#else
              dgemm_(&trans, &notrans, &k, &k, &p, &alpha, time1, &p, time1, &p, &bt, dummy1, &k,1,1);
//#endif
	  					ginv(k, k, dummy1, dummy2);
//#ifdef __APPLE__
//	  					cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, k, p, k, alpha, dummy2, k, time1, p, bt, time2, k); 
//#else
              dgemm_(&notrans, &trans, &k, &p, &k, &alpha, dummy2, &k, time1, &p, &bt, time2, &k,1,1);
//#endif
	  					for(i=0; i < k; i++) {
	  						beta[g+G*i] = 0;
	  						for(j=0; j < p; j++)
	  							beta[g+G*i] += time2[i+j*k]*mu[g+G*j];
	  					}
  
  						for(i=0; i < p; i++) {
  							mu[g+G*i] =0;
  							for(j=0; j < k; j++)
  							  mu[g+G*i] += time1[i+j*p]*beta[g+G*j];
  						}
  					}
					}


					if(COMMENTS) {
					  Rprintf("\n mu: \n");
					  printmx(mu, G, p);
					}

          // nu update
          if(!gaussian_flag) {
					  if(df_flag) {
				      dfold = df[0];
						  // compute sum(z*(log(W)-W))
						  sum = 0;
						  for(i=0; i < N; i++)
						    for(g=0; g < G; g++)
							    sum += z[i + N*g]*(log(W[i + N*g])-W[i + N*g]);
						   if(findRoot((double)1.0/N, sum, (dfold+p)/2, 0.0001, 1000, &dfnew) == 0) {
						  	for(g =0; g < G; g++)
							    df[g] = dfnew;
						   } else {
						     prob = 1;
						   }
              // Rprintf("%f\n\n", dfnew);
					  } else {
					    for(g = 0; g < G; g++) {
						    dfold = df[g];
						  	//compute sum(z[,g]*(log(W[,g]-W[,g]))
						   	sum = 0;
							  for(i=0; i < N; i++)
							    sum += z[i + N*g]*(log(W[i + N*g])-W[i+N*g]);

							  if( findRoot((double)1.0/n[g], sum, (dfold+p)/2, 0.1, 1000, &dfnew) == 0)
							    df[g] = dfnew;
							  else {
									if(COMMENTS)
										Rprintf("*** SETTING prob to 1 after findRoot ***\n");
							    prob = 1;
								}
                // Rprintf("%f\n\n", dfnew);
						  }
					  } // end nu update
          }

          if(COMMENTS) {
					  Rprintf("\n df: \n");
					  printmx(df, 1, G);
					}

          for(g =0; g < G; g++) {
            computeWeightedCovariance(N, p, G, x, z, W, mu, g, 1, sampcov[g]);
						if(COMMENTS) {
						  Rprintf("\nsampcov[%d]\n",g);
						  printmx(sampcov[g], p, p);
					  }
					}

					if(modelName[0] == 'E' && modelName[1] == 'E') {
					  for(i=0; i < p*p; i++)
						  avg[i] = 0.0;
						for(g=0; g < G; g++)
						  for(i=0; i < p*p; i++)
							  avg[i] = avg[i] + pi[g]*sampcov[g][i];
						for(g=0; g < G; g++) {
						  for(i=0; i < p*p; i++)
						   sampcov[g][i] = avg[i];
						}
					}

					// update T and D.
					for(g =0; g < G; g++) {
						for(i=0; i < p*p; i++)
							Sigma[g][i] = sampcov[g][i];
						if(cholest(p, Sigma[g], T[g], D[g], iso) != 0) {
							if(COMMENTS) 
							  Rprintf("*** SETTING prob to 1 after cholest***\n");
							prob = 1;
						}
					}
				
					if(modelName[0] == 'V' && modelName[1] == 'E') {
						for(i=0; i < p*p; i++)
							avg[i] = 0.0;
						for(g=0; g < G; g++)
							for(i=0; i < p*p; i++)
								avg[i] = avg[i] + pi[g]*D[g][i];
						for(g=0; g < G; g++)
							for(i=0; i < p*p; i++)
								D[g][i] = avg[i];

            for(g=0; g < G; g++) {
    					// dummy = ginv(D[g]).
		    			if(ginv(p,p, D[g], dummy) != 0) {
							  prob = 1;
								break;
							}
/*#ifdef __APPLE__
					    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, p, p, p, alpha, T[g], p, dummy, p, bt, avg, p); // avg = t(T[g])*dummy.
							cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, p, p, p, alpha, avg, p, T[g], p, bt, dummy, p); // dummy = avg*T[g].
#else*/
              dgemm_(&trans, &notrans, &p, &p, &p, &alpha, T[g], &p, dummy, &p, &bt, avg, &p,1,1); // avg = t(T[g])*dummy.
              dgemm_(&notrans, &notrans, &p, &p, &p, &alpha, avg, &p, T[g], &p, &bt, dummy, &p,1,1); // dummy = avg*T[g].
//#endif
              if(rcond(dummy, p, p, &rcnumber) != 0 || rcnumber < tol_sing ) {
							  prob = 1;
								break;
							}
							ginv(p, p, dummy, Sigma[g]);
						}
				  } // End of T and D update

				  // E-step: z and w
				  for(g = 0; g < G; g++) {
					  mahalanobis(g, N, p, x, G, mu, Sigma[g], delta);
					  for(i=0; i < N; i++)
					 	  W[i +N*g] = (df[g]+p)/(df[g]+delta[i +N*g]);
				  }
			    //Rprintf("\nW:\n");
				  //printmx(W, N, G);
					for(g = 0; g < G; g++) {
					  for(i = 0; i < N; i++) {
						  determinant(D[g], p, p, &detvalue);
					    log_num[i + N*g] = log(pi[g]) + lgamma((df[g]+p)/2) - log(detvalue)/2 - (log(MYPI)+log(df[g]))*p/2 - 
							                   lgamma(df[g]/2) - ((df[g]+p)/2)*log(1+(delta[i + N*g]/df[g]));
							num[i + N*g] = (long double)exp((long double)log_num[i + N*g]);
						}
					}

					//Rprintf("\n num: \n");
					//printlongdoublemx(num, N, G);
					if(COMMENTS) {
					  Rprintf("\n log_num: \n");
					  printmx(log_num, N, G);
					}
          
					// NAN detection.
          if(prob != 1 && isNan(num, N, G)) {
					  //Rprintf("Nan detected!\n");
					  prob = 1;
					}

					if(prob != 1) {
					  // z <- num / rowSums(num).
						sum2 = 0;
						for(i=0; i < N; i++) {
						  if(classify[i]  > 0)
							  continue;
						  sum = 0;
							for(g=0; g < G; g++)
							  sum += num[i + N*g];
							sum2 += log(sum);
              
							for(g=0; g < G; g++) {
							  if(sum == 0)
								  z[i + N*g] = maxdouble;
								else {
								  z[i + N*g] = num[i + N*g]/sum;
								}
							}
						}
						if(COMMENTS) {
							Rprintf("prob = %d\n", prob);
  						Rprintf("\n Updated z: \n");
	  					printmx(z, N, G);
						}
						// check for convergence
						logl[it] = sum2;
						if(COMMENTS) {
						  Rprintf("\n log[it] = %Lf\n", sum2);
						}
						if(it > 3) {
						  ak = (logl[it] - logl[it-1])/(logl[it-1]-logl[it-2]);
							linf = logl[it-1] + (logl[it]-logl[it-1])/(1-ak);
							if(fabs(linf-logl[it-1]) < TOL)
							  conv = 1;
							if(((logl[it]-logl[it-1])<0) && (conv == 0)) {
							  prob = 1;
								if(COMMENTS)
									Rprintf("****  SETTING prob to 1 ******\n");
							}
							if(it == it_max)
							  conv = 1;
						}
					}
					if(prob == 1)
					  conv = 1;
					it++;
	      } // end of while
				if(COMMENTS) {
				  Rprintf("Out of while\n");
	        Rprintf("----------------------------------\n");
					Rprintf("prob = %d\n", prob);
				}
				

				if(prob != 1) {
				  // compute BIC
					bicval = 2*logl[it-1] - log(N)*(double)params[mIndex]; 
					if(bicval == bicval) { // check whether bicval is 'Nan'.


		        //Rprintf("bicres:\n");
		        //printmx(bicres, modelsLength, G_max);
				    max=maximum(bicres, modelsLength*G_max);
				    if(bicval >= max) { // resume here.
				  	  *Gbest = G;
					    copyvec(df, G_max, nubest);
              copymx(z, N, G_max, N, zbest);
              copymx(mu, G, p, G, mubest);
							if(domeans)
  							copymx(beta, G, 2, G, betabest);

              for(g=0; g < G; g++) {
						    copymx(T[g], p, p, p, Tbest + g*p*p);
						    copymx(D[g], p, p, p, Dbest + g*p*p);
					    }
				    }
					  if(bicval > bicres[mIndex + modelsLength*G]) {
  					  llres[mIndex + modelsLength*G] = logl[it-1];
	  				  bicres[mIndex + modelsLength*G] = bicval;
					  }
					}
					for(i=0; i < N; i++) {
					  max = z[i + N*0];
						for(g=0; g < G; g++) {
						  if(max < z[ i + N*g])
							  max = z[i + N*g];
						}
						for(g=0; g < G; g++) {
						  if(max == z[i+ N*g])
							  MAPz[i + N*g] = 1;
							else
							  MAPz[i + N*g] = 0;
						}
					}

					penalty=0;
					for(i=0; i < N ; i++) 
					  for(g=0; g < G ; g++) 
						  if(MAPz[i + N*g] == 1) {
							    penalty += log(z[i+N*g]);
							}
					iclval = bicval + 2*penalty;
					if(iclval == iclval) {
						max = maximum(iclres, modelsLength*G_max);
						if(iclval >= max) {
						  *Gbest_icl = G;
							copyvec(df, G_max, nubest_icl);
							copymx(z, N, G_max, N, zbest_icl);
              copymx(mu, G, p, G, mubest_icl);
							if(domeans)
  							copymx(beta, G, 2, G, betabest_icl);

              for(g=0; g < G; g++) {
						    copymx(T[g], p, p, p, Tbest_icl + g*p*p);
						    copymx(D[g], p, p, p, Dbest_icl + g*p*p);
					    }
						}
						if(iclval > iclres[mIndex + modelsLength*G])
						  iclres[mIndex + modelsLength*G] = iclval;
          }   
				} else {
				  if(params[mIndex]< 0) {
				    bicval = maxdouble;
						iclval = maxdouble;
					}
					else {
					  bicval = mindouble;
						iclval = mindouble;
					}
			  }
		  } // end for model

			// This is for the two 'funny' ones: EVA and EVI
			for(mIndex = modelsLength-2; mIndex < modelsLength; mIndex++) {
		    if(!models_selected[mIndex])  // If this model is not selected, skip it.
				  continue;
			  prob = 0;
	    	modelName = models[mIndex];

        if(COMMENTS) {
  	      Rprintf("######################################");
  	      Rprintf("######################################");
  			  Rprintf("mIndex = %d, modelName=%s\n", mIndex, modelName);
  	      Rprintf("######################################");
  	      Rprintf("######################################");
				}

				if(modelName[2] == 'I')
				  iso=1;
				else
				  iso=0;

        for(g=0; g < G; g++)
	    	  n[g] = 0.0;
		    for(i=0; i < N*G; i++)
		      z[i] = 0;
    
				if(flag_kmeans == 1) {
	        g = G-G_min;
					for(i=0; i < N; i++) {
					  j = kmeansz[i + g*N]-1;
						z[i +j*N] = 1;
					}
				} else if(classify_flag == 0) { 
		  		for(i=0; i < N; i++) {
			  	  sum=0;
				  	for(g=0; g < G; g++) {
					    z[i + N*g] = unif_rand(); 
			  			sum += z[i + N*g];
			  		}
			  		for(g=0; g < G; g++)
			  		  z[i + N*g] /= sum;
			  	}
				} else {
          for(i=0; i < N; i++) {
					  g = classify[i];
						if(g != 0) {
						  g--;
							z[i + g*N] = 1.0;
							continue;
						}
						for(g=0; g < G; g++)
						  z[i + g*N] = 1.0/(double)G;
					}
				}
				  
				for(g=0; g < G; g++) {
				  n[g] = 0;
					for(i=0; i < N; i++)
					  n[g] += z[i + g*N];
				}

				for(g=0; g < G; g++)
				  pi[g] = n[g]/N;

				for(g=0; g < G; g++) {
				  for(i=0; i < p*p; i++) {
					  Sigma[g][i] = 0;
						sampcov[g][i] = 0;
						T[g][i] = 0;
						D[g][i] = 0;
						if(i/p == i%p)
						  D[g][i] = 1;
					}
				}

				for(k=0; k < p; k++)
				  for(i=0; i < p*p; i++)
					  kappa[k][i] = 0;

	    	for(i=0; i < it_max; i++)
	    	  logl[i] = 0;
	    	for(i=0; i < G; i++)
	    	  df[i] = 50;
	    	for(i=0; i < N*G; i++) {
	    		num[i]     = 0.0;
	    		log_num[i] = 0.0;
	    		W[i]       = 0.0;
	    		delta[i]   = 0.0;
	    	}
				it = 1;
				conv = 0;

				for(g=0; g < G; g++) { 
				  for(j=0; j < p; j++) {
						mu[g + j*G] = 0.0;
						for(i=0; i < N; i++)
							mu[g + j*G] += x[i + j*N]*z[i + g*N];
						mu[g + j*G] /= n[g];
					}
				}
				for(i=0; i < G*2; i++)
					beta[i] = 0.0;

				for(g=0; g < G; g++) {
          computeWeightedCovariance(N, p, G, x, z, W, mu, g, 0, sampcov[g]);
					mahalanobis(g, N, p, x, G, mu, sampcov[g], delta);
					for(i=0; i < N; i++)
						W[i +N*g] = (df[g]+p)/(df[g]+delta[i +N*g]);
				}
				if(COMMENTS) {
  				Rprintf("About to enter the main while loop \n");
  			  Rprintf("\nz:\n");
  				printmx(z, N, G);
  			  Rprintf("\nW:\n");
  				printmx(W, N, G);
  				Rprintf("\nmu:\n");
  				printmx(mu, G, p);
				}
        
				while (conv != 1) {
          for(g=0; g < G; g++)
	    	    n[g] = 0.0;
					for(i=0; i < N; i++)
					  for(g=0; g < G; g++)
						  n[g] += z[i + N*g];
					for(g=0; g < G; g++)
					  pi[g] = n[g]/N;
          
					// update mu.
					for(g =0; g < G; g++) { 
					  sum2=0;
						for(i=0; i < N; i++)
						  sum2 += z[i + N*g]*W[i + N*g];
						for(j=0; j < p; j++) {
						  sum=0; 
							for(i=0; i < N; i++)
							  sum += z[i +N*g]*W[i +N*g]*x[i +N*j];
							mu[g + G*j] = sum/sum2;
						}
						if(domeans) {
  						k=2;
/*#ifdef __APPLE__
	  				  cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, k, k, p, alpha, time1, p, time1, p, bt, dummy1, k);
#else*/
              dgemm_(&trans, &notrans, &k, &k, &p, &alpha, time1, &p, time1, &p, &bt, dummy1, &k,1,1);
//#endif
	  					ginv(k, k, dummy1, dummy2);
/*#ifdef __APPLE__
	  					cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, k, p, k, alpha, dummy2, k, time1, p, bt, time2, k); 
#else*/
              dgemm_(&notrans, &trans, &k, &p, &k, &alpha, dummy2, &k, time1, &p, &bt, time2, &k,1,1);
//#endif
	  					for(i=0; i < k; i++) {
	  						beta[g+G*i] = 0;
	  						for(j=0; j < p; j++)
	  							beta[g+G*i] += time2[i+j*k]*mu[g+G*j];
	  					}
  
  						for(i=0; i < p; i++) {
  							mu[g+G*i] =0;
  							for(j=0; j < k; j++)
  							  mu[g+G*i] += time1[i+j*p]*beta[g+G*j];
  						}
  					}
					}

          // nu update
          if(!gaussian_flag) {
					  if(df_flag) {
				      dfold = df[0];
					  	// compute sum(z*(log(W)-W))
					  	sum = 0;
					  	for(i=0; i < N; i++)
					  	  for(g=0; g < G; g++)
					  		  sum += z[i + N*g]*(log(W[i + N*g])-W[i + N*g]);
					  	if(findRoot((double)1.0/N, sum, (dfold+p)/2, 0.0001, 1000, &dfnew) == 0) {
					  		for(g =0; g < G; g++)
					  		  df[g] = dfnew;
					  	} else {
					  	  prob = 1;
					  	}
					  } else {
					    for(g = 0; g < G; g++) {
						    dfold = df[g];
						  	//compute sum(z[,g]*(log(W[,g]-W[,g]))
						  	sum = 0;
						  	for(i=0; i < N; i++)
						  	  sum += z[i + N*g]*(log(W[i + N*g])-W[i+N*g]);
						  	if(findRoot((double)1.0/n[g], sum, (dfold+p)/2, 0.1, 1000, &dfnew) == 0)
						  	  df[g] = dfnew;
						  	else
						  	  prob = 1;
					  	}
					  } // end nu update
          }
		     
				  if(COMMENTS) { 
	  				Rprintf("\n df: \n");
		  			printmx(df, 1, G);
					}

					for(g =0; g < G; g++) {
            computeWeightedCovariance(N, p, G, x, z, W, mu, g, 1, sampcov[g]);
					//	Rprintf("sampcov[g]:\n");
					//	printmx(sampcov[g], p, p);
					}
					
					if(iso) { // For EVI
						for(k=1; k < p; k++)
							for(i=0; i < p; i++) 
								for(j=0; j < p; j++) {
									kappa[k][i + j*p] = 0;
									for(g=0; g < G; g++)
										kappa[k][i+ j*p] += n[g]*sampcov[g][i+j*p]/D[g][k + k*p];
								}
					} else { // for EVA
						for(k=1; k < p; k++) {
							for(i=0; i < p; i++)
								for(j=0; j < p; j++) {
									kappa[k][i+j*p] = 0;
									for(g=0; g < G; g++)
										kappa[k][i+ j*p] += pi[g]*sampcov[g][i+p*j]/D[g][k+p*k];
								}
						}
					}	

				  // update T and D.
					for(g =0; g < G; g++) {
						for(i=0; i < p*p; i++)
							Sigma[g][i] = sampcov[g][i];
						if(cholest2(p, kappa, Sigma[g], T[g], D[g], iso) != 0)
							prob = 1;
					} // End T and D update
					// E-step: z and w
					for(g=0; g < G; g++) {
  					mahalanobis(g, N, p, x, G, mu, Sigma[g], delta);
	  				for(i=0; i < N; i++)
		  				W[i +N*g] = (df[g]+p)/(df[g]+delta[i +N*g]);									
					}
					if(COMMENTS) {
  					Rprintf("W:\n"); 
	  				printmx(W, N, G);
          }
					
					for(g = 0; g < G; g++) {
					  for(i = 0; i < N; i++) {
						  determinant(D[g], p, p, &detvalue);
					    log_num[i + N*g] = log(pi[g]) + lgamma((df[g]+p)/2) - log(detvalue)/2 - (log(MYPI)+log(df[g]))*p/2 - 
							                   lgamma(df[g]/2) - ((df[g]+p)/2)*log(1+(delta[i + N*g]/df[g]));
							num[i + N*g] = (long double) exp((long double)log_num[i + N*g]);
						}
					}					
		
		      if(COMMENTS) {
				  	Rprintf("log_num:\n");
					  printmx(log_num, N, G);
					}


					// NAN detection.
          if(prob != 1 && isNan(num, N, G)) {
					  //Rprintf("Nan detected!\n");
					  prob = 1;
					}
					if(prob != 1) {
					  // z <- num / rowSums(num).
						sum2 = 0;
						for(i=0; i < N; i++) {
							if(classify[i] > 0) 
								continue;
						  sum = 0;
							for(g=0; g < G; g++)
							  sum += num[i + N*g];
							sum2 += log(sum);
							for(g=0; g < G; g++) {
							  if(sum == 0)
								  z[i + N*g] = maxdouble;
								else {
								  z[i + N*g] = num[i + N*g]/sum;
								}
							}
						}

						if(COMMENTS) {
						  Rprintf("updated z:\n");
						  printmx(z, N, G);
						}

						// check for convergence
						logl[it] = sum2;
						if(COMMENTS) {
						  Rprintf("logl[it] = %Lf\n", sum2);
						}
						if(it > 3) {
						  ak = (logl[it] - logl[it-1])/(logl[it-1]-logl[it-2]);
							linf = logl[it-1] + (logl[it]-logl[it-1])/(1-ak);
							if(fabs(linf-logl[it-1]) < TOL)
							  conv = 1;
							if(((logl[it]-logl[it-1])<0) && (conv == 0)) {
							  prob = 1;
								if(COMMENTS)
									Rprintf("****  SETTING prob to 1 ******\n");
							}
							if(it == it_max)
							  conv = 1;
						}
					}
					if(prob == 1)
					  conv = 1;
					it++;
	      } // end of while					
				if(COMMENTS) {
  				Rprintf("<--------- OUT OF WHILE ---- >\n");
	  			Rprintf("<--------- OUT OF WHILE ---- >\n");
				}

				if(prob != 1) {
				  // compute BIC
					bicval = 2*logl[it-1] - log(N)*params[mIndex]; 
					
          if(COMMENTS)
					  Rprintf("\nparam = %d, bicval= %G\n", params[mIndex], bicval);
					if(bicval == bicval) { // check whether bicval is 'Nan'.
				    max=maximum(bicres, modelsLength*G_max);
				    if(bicval >= max) { // resume here.
				    	*Gbest = G;
					    copyvec(df, G_max, nubest);
              copymx(z, N, G_max, N, zbest);
              copymx(mu, G, p, G, mubest);
							if(domeans)
  							copymx(beta, G, 2, G, betabest);

              for(g=0; g < G; g++) {
  						  copymx(T[g], p, p, p, Tbest + g*p*p);
  						  copymx(D[g], p, p, p, Dbest + g*p*p);
  					  }
  				  }
					  if(bicval > bicres[mIndex + modelsLength*G]) {
  					  llres[mIndex + modelsLength*G] = logl[it-1];
	  				  bicres[mIndex + modelsLength*G] = bicval;
			  		}
					}
					for(i=0; i < N; i++) {
					  max = z[i + N*0];
						for(g=0; g < G; g++) {
						  if(max < z[ i + N*g])
							  max = z[i + N*g];
						}
						for(g=0; g < G; g++) {
						  if(max == z[i+ N*g])
							  MAPz[i + N*g] = 1;
							else
							  MAPz[i + N*g] = 0;
						}
					}

					penalty=0;
					for(i=0; i < N; i++)
					  for(g=0; g < G; g++)
						  if(MAPz[i + N*g] == 1) {
							  penalty += log(z[i+N*g]);
							}
					iclval = bicval + 2*penalty;
					if(iclval == iclval) {   // iclval is not nan.
						max = maximum(iclres, modelsLength*G_max);
						if(iclval >= max) {
						  *Gbest_icl = G;
							copyvec(df, G_max, nubest_icl);
							copymx(z, N, G_max, N, zbest_icl);
              copymx(mu, G, p, G, mubest_icl);
							if(domeans)
							  copymx(beta, G, 2, G, betabest_icl);

              for(g=0; g < G; g++) {
						    copymx(T[g], p, p, p, Tbest_icl + g*p*p);
						    copymx(D[g], p, p, p, Dbest_icl + g*p*p);
					    }
						}
						if(iclval > iclres[mIndex + modelsLength*G])
						  iclres[mIndex + modelsLength*G] = iclval;
					}
				} else {
				  if(params[mIndex]< 0) {
				    bicval = maxdouble;
						iclval = maxdouble;
					}
					else {
					  bicval = mindouble;
						iclval = mindouble;
					}
			  }
			}
    }
		if(COMMENTS) {
      Rprintf("llres:\n");
  		printmx(llres, modelsLength, G_max);
  		Rprintf("bicres:\n");
  		printmx(bicres, modelsLength, G_max);
  		Rprintf("Gbest = %d\n",*Gbest);
  		Rprintf("nubest:\n");
  		printmx(nubest, 1, *Gbest);
  		Rprintf("zbest:\n");
  		printmx(zbest, N, *Gbest);
		}
		// exit(1);

	PutRNGstate();

	free(logl);
  free(df);
	free(n);
	free(pi);


	free(z);

	free(mu);

	free(num);
	free(log_num);
	free(W);
	free(delta);
	free(MAPz);
	free(avg);
	free(dummy);

	for(g=0; g < G_max; g++) {
  	free(Sigma[g]);
  	free(sampcov[g]);
	  free(T[g]);
  	free(D[g]);
	}
	free(Sigma);
	free(sampcov);
	free(T);
	free(D);


	for(i=0; i < p; i++)
	  free(kappa[i]);
	free(kappa);
}


void tchol_entry(double *x, int *classify, int *N, int *p, int *G_min, int *G_max, int *df_flag, int *gaussian_flag, int *models_selected, int *flag_kmeans, double *kmeansz, int *Gbest, int *Gbest_icl, double *zbest, double *zbest_icl, double *nubest, double *nubest_icl, double *mubest, double *mubest_icl, double *Tbest, double *Tbest_icl, double *Dbest, double *Dbest_icl, double *llres, double *bicres, double *iclres) {
  t_chol(x, classify, *N, *p, *G_min, *G_max+1, *df_flag, *gaussian_flag, models_selected, 0.1, *flag_kmeans, kmeansz, Gbest, Gbest_icl, zbest, zbest_icl, nubest, nubest_icl, mubest, mubest_icl, Tbest, Tbest_icl, Dbest, Dbest_icl, llres, bicres, iclres);
}

void tcholmeans_entry(double *x, int *classify, double *time1, int *N, int *p, int *G_min, int *G_max, int *df_flag, int *gaussian_flag, int *models_selected, int *flag_kmeans, double *kmeansz, int *Gbest, int *Gbest_icl, double *zbest, double *zbest_icl, double *nubest, double *nubest_icl, double *mubest, double *mubest_icl, double *Tbest, double *Tbest_icl, double *Dbest, double *Dbest_icl, double *llres, double *bicres, double *iclres) {

  t_chol_means(x, classify, *N, *p, time1, *G_min, *G_max+1, *df_flag, *gaussian_flag, models_selected, 0.1, *flag_kmeans, kmeansz, Gbest, Gbest_icl, zbest, zbest_icl, nubest, nubest_icl, mubest, mubest_icl, Tbest, Tbest_icl, Dbest, Dbest_icl, llres, bicres, iclres);
}
