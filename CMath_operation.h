#pragma once
class CMath_operation
{
public:
	CMath_operation();
	~CMath_operation();

public:
	double *PBUF, *PA[25], *PW[25];
	int PM, PN;

public:
	void GetMatrix(double *A, int M, int N);
	double GetMainElement(int k);
	int Jordan_G();

public:
	void matrix_cross(double *A, double *B, double *C);//C=A X B向量差乘
	int  matrixInversion(double *A, double *B, int n);//矩阵求逆B=A(-1) ---n:维数
	void matrix_add(double *A, double *B, double *C, int PM, int PN);//矩阵相加C=A+B ---M:行数 N：列数
	void matrix_minus(double *A, double *B, double *C, int PM, int PN);//矩阵相减C=A-B ---M:行数 N：列数
	void matrix_multiply(double *A, double *B, double *C, int PM, int PN, int PQ);//矩阵乘法C=A*B  ---C[PM][PQ]=A[PM][PN]*B[PN][PQ]
	void matrix_transpose(double *A, double *B, int M, int N);//矩阵求转置B=AT  ---M:行数 N：列数


};

