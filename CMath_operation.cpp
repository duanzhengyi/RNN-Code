#include "stdafx.h"
#define ZERO 1.0E-30


CMath_operation::CMath_operation()
{
}


CMath_operation::~CMath_operation()
{
}

void CMath_operation::GetMatrix(double * A, int M, int N)
{
	PM = M;
	PN = N;
	for (int i = 0; i < PM; i++) {
		PA[i] = A + i * PN;
	}
}

double CMath_operation::GetMainElement(int k)
{
	int FLAG;
	double temp, MEAV;
	FLAG = k;
	MEAV = fabs(PA[k][k]);
	for (int i = k + 1; i < PM; i++) {
		temp = fabs(PA[i][k]);
		if (temp <= MEAV) {
			continue;
		}
		FLAG = i;
		MEAV = temp;
	}
	for (int j = 0; j < PN; j++) {
		temp = PA[k][j];
		PA[k][j] = PA[FLAG][j];
		PA[FLAG][j] = temp;
	}
	return MEAV;
}

int CMath_operation::Jordan_G()
{
	double x, test = 0;
	for (int k = 0; k < PM; k++) {
		test = GetMainElement(k);
		if (test < ZERO) {
			return -11;
		}
		x = PA[k][k];
		for (int j = 0; j < PN; j++) {
			PA[k][j] = PA[k][j] / x;
		}
		for (int i = 0; i < PM; i++) {
			if (i == k) {
				continue;
			}
			x = PA[i][k];
			for (int j = k; j < PN; j++) {
				PA[i][j] = PA[i][j] - x * PA[k][j];
			}
		}
	}

	return 0;
}

void CMath_operation::matrix_cross(double *A, double *B, double *C)
{
	double Xu, Yu, Zu;
	double Xv, Yv, Zv;
	double Xn, Yn, Zn;

	Xu = A[0]; Yu = A[1]; Zu = A[2];
	Xv = B[0]; Yv = B[1]; Zv = B[2];

	Xn = Yu * Zv - Zu * Yv;
	Yn = Zu * Xv - Xu * Zv;
	Zn = Xu * Yv - Yu * Xv;
	C[0] = Xn;
	C[1] = Yn;
	C[2] = Zn;
}

int CMath_operation::matrixInversion(double *A, double *B, int n)
{
	int ErrCode = 0;
	if (n > 25) {
		return -1;
	}
	PM = n; PN = PM * 2;
	PBUF = (double*)malloc(PM*PN * sizeof(double));
	if (!PBUF) {
		return -2;
	}
	for (int k = 0; k < PM; k++) {
		PW[k] = A + k * PM;
		PA[k] = PBUF + k * PN;
	}
	for (int i = 0; i < PM; i++) {
		for (int j = 0; j < PM; j++) {
			PA[i][j] = PW[i][j];
			PA[i][j + PM] = 0.0;
		}
		PA[i][i + PM] = 1.0;
	}
	ErrCode = Jordan_G();
	for (int i = 0; i < PM; i++) {
		PW[i] = B + i * PM;
		for (int j = 0; j < PM; j++) {
			PW[i][j] = PA[i][j + PM];
		}
	}
	free(PBUF);
	return ErrCode;

}

void CMath_operation::matrix_add(double * A, double * B, double * C, int PM, int PN)
{
	for (int i = 0; i < PM; i++) {
		PA[i] = A + i * PN;
		PW[i] = B + i * PN;
	}
	for (int i = 0; i < PM; i++) {
		for (int j = 0; j < PN; j++) {
			C[i*PN + j] = PA[i][j] + PW[i][j];
		}
	}
}

void CMath_operation::matrix_minus(double *A, double *B, double *C, int PM, int PN)
{
	for (int i = 0; i < PM; i++) {
		PA[i] = A + i * PN;
		PW[i] = B + i * PN;
	}
	for (int i = 0; i < PM; i++) {
		for (int j = 0; j < PN; j++) {
			C[i*PN + j] = PA[i][j] - PW[i][j];
		}
	}
}

//¾ØÕóÏà¼õ
void CMath_operation::matrix_multiply(double *A, double *B, double *C, int PM, int PN, int PQ)
{
	for (int i = 0; i < PM; i++) {
		PA[i] = A + i * PN;
	}
	for (int i = 0; i < PN; i++) {
		PW[i] = B + i * PQ;
	}
	for (int i = 0; i < PM; i++) {
		for (int j = 0; j < PQ; j++) {
			C[i*PQ + j] = 0.0;
			for (int k = 0; k < PN; k++) {
				C[i*PQ + j] += PA[i][k] * PW[k][j];
			}
		}
	}

}

void CMath_operation::matrix_transpose(double *A, double *B, int M, int N)
{
	GetMatrix(A, M, N);
	for (int i = 0; i < PM; i++) {
		for (int j = 0; j < PN; j++) {
			B[j*PM + i] = PA[i][j];
		}
	}
}
