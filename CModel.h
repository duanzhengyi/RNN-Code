#pragma once
class CModel
{
public:
	CModel();
	~CModel();

public:
	vector<vector<double>> state_vector;
	vector<vector<double>> control_vector;
	vector<vector<double>> output_vector;

	int State_Quantity;
	int Control_Quantity;
	int Output_Quantity;
	int Predict_Step;

public:
	void Initial_Data(int p_s, int s_q, int c_q, int out_q,vector<double> state_ve, vector<vector<double>> control_u);//���ó�ֵ
	void Model_State_Function();//ϵͳ״̬����
	void Model_Output_Function();//ϵͳ�������
	void Model_Operation();//ģ������
};

