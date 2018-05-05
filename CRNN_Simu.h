#pragma once
class CRNN_Simu
{
public:
	CRNN_Simu();
	~CRNN_Simu();

public:
	void simulate();

public:
	void Simulate_Error_Backpropogation();

public:
	int Predictive_Step;//Ԥ�ⲽ��

	vector<vector<double>> errorDelta_2_Out;
	vector<vector<double>> errorDelta_2_Hid;

	vector<vector<double>> errorDelta_1_Out;
	vector<vector<double>> errorDelta_1_Hid;

	vector<vector<double>> gradient_vector_StateSpace;
	vector<vector<double>> gradient_vector_OutputSpace;

	vector<vector<double>> Target_Output;

public:
	//State-Space NN
	int Quantity_InputNode_SS;//�����
	int Quantity_OutputNode_SS;//�����
	int Term_Num_SS;//��

	//Output-Space NN
	int Quantity_InputNode_OS;//�����
	int Quantity_OutputNode_OS;//�����
	int Term_Num_OS;//��

	vector<double> InputNode_Value_vector;//State-Space NN�ĳ�ʼֵ
	vector<double> Input_Control_vector;//ÿһ��ʱ������Ŀ���u��ֵ

	vector<CNeural_unit> State_Space_NN;
	vector<CNeural_unit> Output_Space_NN;

	CNeural_unit State_Space_NN_Real;
	CNeural_unit Output_Space_NN_Real;

	vector<vector<double>> State_Space_Neural_Weight_Reserved;//������һ��NNȨֵSS
	vector<vector<double>> Output_Space_Neural_Weight_Reserved;//������һ��NNȨֵOS

	vector<vector<double>> gradient_vector_StateSpace_Reserved;//������һ��NN�ݶ�SS
	vector<vector<double>> gradient_vector_OutputSpace_Reserved;//������һ��NN�ݶ�SS

	double Learning_rate_min;//��Сѧϰ��
	double Learning_Rate_NN;//ѧϰ��

	double Error_Function_Value;//������ֵ
	double Error_Function_Value_Temp;//��ʱ������ֵ
	double Error_Accuracy;//����
	int Max_Episode;//������ѭ������
	int Episode_num;//�������´���

	CModel m_model;

	FILE * file_gradient_ss;//�ݶ������ļ�
	FILE * file_gradient_os;

	FILE * file_errorDelta_1_Out;//�в�
	FILE * file_errorDelta_1_Hid;
	FILE * file_errorDelta_2_Out;
	FILE * file_errorDelta_2_Hid;

	FILE * file_ss_nn_weight;
	FILE * file_os_nn_weight;

	FILE * file_nn_output;

public:
	void Initial_Simu();
	void Initial_Data();
	void Initial_NeuralNetwork();
	void Create_Real_NeuralNetwork();

	void Calculate_Error();
	void NN_1_OutputLayer_CalError(int index_step);
	void NN_1_HiddenLayer_CalError(int index_step);

	void Update_Weight();//����Ȩֵ
	void Cal_Gradient();//�����ݶ�
	double Cal_Error_Value_Temp();//��ʱ����������
	void Cal_Error_Function_Value();//����������ֵ
	void Unfold_Neural_Network();//չ��������
	void Data_Feedforward();//������ǰ����
	void Initial_Unfolding_Neural_Network();//չ��RNN
	void File_Operate();//�ļ�����
	void File_Initail();//�ļ���ʼ��
	void Recurrent_Initial_NN_Weight();//��ʼ��Ȩֵ
	void Update_learning_rate(int step);//����ѧϰ��

};

