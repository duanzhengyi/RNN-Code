#pragma once
class CNeural_unit
{
public:
	CNeural_unit(int InputNode_Quantity, vector<double> InputNode_Value, int OutputNode_Quantity, int Term_Number, Nerual_role NN_Role);
	CNeural_unit();
	~CNeural_unit();

public:
	int InputNode_Quantity;//����ڵ�����
	vector<double> InputNode_Value;//����ڵ���ֵ

	int OutputNode_Quantity;//����ڵ�����
	
public:
	int Term_Number;//MTN���������
	Nerual_role NN_Role;//NN�䵱�Ľ�ɫ
	
public:
	vector<double> u_value;//��ؽڵ��������ֵ
	vector<double> s_value;
	vector<double> v_value;

	vector<int> SetQuantity;//ÿһ�������
	vector<vector<int>> Term_NumVector;//ÿһ������ڵ��Ӧÿһ��ĸ���

public:
	vector<string> State_StringVector;//״̬�����ַ���ʾ
	vector<string> HiddenState_StringVector;//���ڵ���ַ���ʾ
	
	vector<double> OutputNode_Value;//����ڵ���ֵ
	int HiddenNode_Quantity;//���ڵ�����
	vector<vector<double>> Neural_Weight;//Ȩֵ
	int NeuralNetwork_Index;//��NN��unfold�е����

public:
	vector<vector<int>> InputNode_LineIndex;//����ڵ�Ķ�Ӧ�������

public:
	void Initial_NeuralNetwork();
	void Calculate_SetQuantity();
	void Calculate_WeightQuantity();
	void Calculate_LineIndex();
	void Generate_StateString();
	void Recuurent_Produce(int index,int start,vector<int> vector_one);
	void Data_FeedForward();//��Ϣ��ǰ����
	void Inital_OutputSize();//��ʼ����������ռ��С

};

