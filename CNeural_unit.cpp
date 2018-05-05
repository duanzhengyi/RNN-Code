#include "stdafx.h"
#include "CNeural_unit.h"


CNeural_unit::CNeural_unit(int InputNode_Quantity, vector<double> InputNode_Value, int OutputNode_Quantity, int Term_Number, Nerual_role NN_Role)
{
	this->InputNode_Quantity = InputNode_Quantity;
	this->InputNode_Value = InputNode_Value;
	this->OutputNode_Quantity = OutputNode_Quantity;
	this->Term_Number = Term_Number;
	this->NN_Role = NN_Role;
	
	Initial_NeuralNetwork();
	
}

CNeural_unit::CNeural_unit()
{

}

CNeural_unit::~CNeural_unit()
{

}

void CNeural_unit::Initial_NeuralNetwork()
{
	Calculate_SetQuantity();//每个次项的数量
	Calculate_WeightQuantity();//权值赋值
	Generate_StateString();//状态变量的字符串
	Calculate_LineIndex();//输入节点与隐节点连线序号
	Inital_OutputSize();

}

void CNeural_unit::Calculate_SetQuantity()
{
	//常数项为1个，不计入
	
	vector<int> oneVector;
	vector<int> twoVector;

	for (int i=0;i<Term_Number;i++)
	{
		if (i>0)
		{
			twoVector = Term_NumVector.at(i - 1);
		}

		int sum_one=0;
		for (int k=0;k< InputNode_Quantity;k++)
		{
			if (i ==0)
			{
				oneVector.push_back(1);
				sum_one += 1;
			}
			else {
				int sum=0;

				for (int j=k;j< InputNode_Quantity;j++)
				{
					sum += twoVector[j];
				}

				oneVector.push_back(sum);
				sum_one += sum;
			}
		}

		SetQuantity.push_back(sum_one);

		Term_NumVector.push_back(oneVector);
		oneVector.clear();//清零
		twoVector.clear();

	}

}

void CNeural_unit::Generate_StateString()
{

	string str = "abcdefghijklmn";

	for (int i = 0; i < InputNode_Quantity; i++)
	{
		State_StringVector.push_back(str.substr(i, 1));
	}
}

void CNeural_unit::Calculate_WeightQuantity()
{
	HiddenNode_Quantity = 0;
	for (int i=0;i<int(SetQuantity.size());i++)
	{
		HiddenNode_Quantity += SetQuantity[i];
	}

	vector<double> one_Vector;
	srand((unsigned)time(NULL));
	for (int i=0;i<HiddenNode_Quantity;i++)
	{	
		if (i==0)//常数项权值
		{
			for (int j = 0; j < OutputNode_Quantity; j++)
			{
				double rand_number = rand() % 100 / (double)101;//0-1随机数
				one_Vector.push_back(rand_number);
			}
			Neural_Weight.push_back(one_Vector);
			one_Vector.clear();
		}
		
		for (int j = 0; j < OutputNode_Quantity; j++)
		{
			double rand_number = rand() % 100 / (double)101;//0-1随机数
			one_Vector.push_back(rand_number);
		}
		Neural_Weight.push_back(one_Vector);
		one_Vector.clear();
	}

}

void CNeural_unit::Calculate_LineIndex()
{
	//创建隐节点的string字符串
	for (int term_num=1; term_num<=Term_Number; term_num++)
	{
		vector<int> one;
		Recuurent_Produce(term_num,0,one);
	}

	for (int i=0;i<InputNode_Quantity;i++)
	{
		vector<int> vector_one;
		for (int j=0;j<int(HiddenState_StringVector.size());j++)
		{
			string str_one = HiddenState_StringVector[j];
			string::size_type idx;
			idx = str_one.find(State_StringVector[i]);
			if (idx == string::npos)//不存在
			{
			}
			else {
				vector_one.push_back(j);
			}
		}
		InputNode_LineIndex.push_back(vector_one);
		vector_one.clear();
	}
	
}

void CNeural_unit::Recuurent_Produce(int index, int start, vector<int> vector_one)
{
	if (index==1)//最后一个变量
	{
		for (int i=start;i<InputNode_Quantity;i++)
		{
			string sum_string="";
			for (int j=0;j<int(vector_one.size());j++)
			{
				string state_s=State_StringVector[vector_one[j]];
				sum_string += state_s;
			}
			sum_string += State_StringVector[i];
			HiddenState_StringVector.push_back(sum_string);
		}
		return;
	}
	else {
		for (int i = start; i < InputNode_Quantity; i++)
		{
			int index_next = index - 1;
			vector<int> new_vector = vector_one;
			new_vector.push_back(i);
			Recuurent_Produce(index_next, i, new_vector);
		}
	}
}

void CNeural_unit::Data_FeedForward()
{
	//隐节点的输出
	u_value.clear();
	for (int i=0;i<int(HiddenState_StringVector.size());i++)
	{
		double sum = 0;
		for (int j=0;j<InputNode_Quantity;j++)
		{
			string str_one = HiddenState_StringVector[i];
			string::size_type idx;
			idx = str_one.find(State_StringVector[j]);
			if (idx == string::npos)//不存在
			{
			}
			else {
				sum += InputNode_Value[j];
			}
		}
		u_value.push_back(sum);
	}

	s_value.clear();
	s_value = u_value;

	//NN的输出
	OutputNode_Value.clear();
	for (int i=0;i<OutputNode_Quantity;i++)
	{
		double sum = 0;
		//常数项
		sum = sum + 1 * Neural_Weight[0][i];

		//非常数项
		for (int j=0;j<int(u_value.size());j++)
		{
			sum = sum + u_value[j] * Neural_Weight[1 + j][i];
		}
		OutputNode_Value.push_back(sum);
	}

}

void CNeural_unit::Inital_OutputSize()
{
	OutputNode_Value.resize(OutputNode_Quantity);
}

