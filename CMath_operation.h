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
	void matrix_cross(double *A, double *B, double *C);//C=A X B�������
	int  matrixInversion(double *A, double *B, int n);//��������B=A(-1) ---n:ά��
	void matrix_add(double *A, double *B, double *C, int PM, int PN);//�������C=A+B ---M:���� N������
	void matrix_minus(double *A, double *B, double *C, int PM, int PN);//�������C=A-B ---M:���� N������
	void matrix_multiply(double *A, double *B, double *C, int PM, int PN, int PQ);//����˷�C=A*B  ---C[PM][PQ]=A[PM][PN]*B[PN][PQ]
	void matrix_transpose(double *A, double *B, int M, int N);//������ת��B=AT  ---M:���� N������


};

