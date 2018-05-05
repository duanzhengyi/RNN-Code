#include "stdafx.h"


CModel::CModel()
{
}

CModel::~CModel()
{
}

void CModel::Initial_Data(int p_s, int s_q, int c_q, int out_q,vector<double> state_ve, vector<vector<double>> control_u)
{
	Predict_Step = p_s;

	State_Quantity = s_q;
	Control_Quantity = c_q;
	Output_Quantity = out_q;

	state_vector.resize(Predict_Step + 1);
	control_vector.resize(Predict_Step);
	output_vector.resize(Predict_Step);

	state_vector[0].resize(state_ve.size());
	for (int i=0;i<int(state_ve.size());i++)
	{
		state_vector[0][i] = state_ve[i];
	}

	control_vector = control_u;
}

void CModel::Model_State_Function()
{
	for (int i=0;i<Predict_Step;i++)
	{
		//状态方程
		double new_state_one = state_vector[i][0]+ control_vector[i][0];
		double new_state_two = 2*state_vector[i][1];

		vector<double> new_state_ve;
		new_state_ve.push_back(new_state_one);
		new_state_ve.push_back(new_state_two);
		state_vector[i+1]=new_state_ve;
	}
}

void CModel::Model_Output_Function()
{
	for (int i = 0; i<Predict_Step; i++)
	{
		//输出方程
		double new_output_one = state_vector[i+1][0];
		double new_output_two = state_vector[i+1][1];

		vector<double> new_output_ve;
		new_output_ve.push_back(new_output_one);
		new_output_ve.push_back(new_output_two);
		output_vector[i] = new_output_ve;
	}
}

void CModel::Model_Operation()
{
	Model_State_Function();
	Model_Output_Function();
}
