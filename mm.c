#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <sys/time.h>
#include <gsl/gsl_cblas.h>

#define N 5000;

void mm_bruteforce_ijk(double *a, double *b, double *c, int I, int K, int J) {
    for(int i = 0; i < I; i++) {
        for(int j = 0; j < J; j++) {
            for(int k = 0; k < K; k++) {
                c[i * J + j] += a[i * K + k] * b[k * J + j];
            }
        }
    }
}

void mm_bruteforce_ikj(double *a, double *b, double *c, int I, int K, int J) {
    for(int i = 0; i < I; i++) {
        for(int k = 0; k < K; k++) {
            double dv = a[i * K + k];
            for(int j = 0; j < J; j++) {
                c[i * J + j] += dv * b[k * J + j];
            }
        }
    }
}

void mm_omp(double *a, double *b, double *c, int I, int K, int J) {
    #pragma omp parallel for 
    for(int i = 0; i < I; i++) {
        for(int k = 0; k < K; k++) {
            register double dv = a[i * K + k];
            for(int j = 0; j < J; j++) {
                c[i * J + j] += dv * b[k * J + j];
            }
        }
    }
}

void mm_cblas_dgemm(double *a, double *b, double *c, int p, int q, int r) {
    int l = p;
    int m = q;
    int n = r;

    int lda = m;
    int ldb = n;
    int ldc = n;

    double alpha = 1.0;
    double beta = 0.0;

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            l, n, m, alpha, a, lda, b, ldb, beta, c, ldc);
}

void init_arange(double *mat, int a, int b) {
    for(int i = 0; i < (a*b); i++) {
        mat[i] = i + 1;
    }
}

void init_value(double *mat, int a, int b, double value) {
    for(int i = 0; i < (a*b); i++) {
        mat[i] = value;
    }
}

void init_zero(double *mat, int a, int b) {
    init_value(mat, a, b, 0);
}

void init_one(double *mat, int a, int b) {
    init_value(mat, a, b, 1);
}

void timer_start(struct timeval *pstv) {
    gettimeofday(pstv, NULL);
}

void timer_end(struct timeval *petv) {
    gettimeofday(petv, NULL);
}

void timer_print(struct timeval *pstv, struct timeval *petv) {
    time_t sec;
    suseconds_t usec;
    sec = petv->tv_sec - pstv->tv_sec;
    usec = petv->tv_usec - pstv->tv_usec;
    if(usec < 0) {
        sec--;
        usec += 1000000;
    }
    printf("elapsed time : %ld.%ld\n", sec, usec);
}

typedef void (*fptr_mm)(double *a, double *b, double *c, int I, int K, int J);

void check_etime_mm(double *a, double *b, double *c, int I, int K, int J, fptr_mm mm) {
    struct timeval stv;
    struct timeval etv;
    init_zero(c, I, J);
    timer_start(&stv);
    mm(a, b, c, I, K, J);
    timer_end(&etv);
    timer_print(&stv, &etv);
}

int main(int argc, char *argv[]) {
    int I, K, J;
    I = K = J = N;

    double *a = (double*)malloc(sizeof(double) * I * K);
    double *b = (double*)malloc(sizeof(double) * K * J);
    double *c = (double*)malloc(sizeof(double) * I * J);

    init_arange(a, I, K);
    init_arange(b, K, J);

    check_etime_mm(a, b, c, I, K, J, mm_bruteforce_ijk);
    check_etime_mm(a, b, c, I, K, J, mm_bruteforce_ikj);
    check_etime_mm(a, b, c, I, K, J, mm_omp);
    check_etime_mm(a, b, c, I, K, J, mm_cblas_dgemm);

    return 0;
}
