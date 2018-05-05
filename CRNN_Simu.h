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
	int Predictive_Step;//预测步长

	vector<vector<double>> errorDelta_2_Out;
	vector<vector<double>> errorDelta_2_Hid;

	vector<vector<double>> errorDelta_1_Out;
	vector<vector<double>> errorDelta_1_Hid;

	vector<vector<double>> gradient_vector_StateSpace;
	vector<vector<double>> gradient_vector_OutputSpace;

	vector<vector<double>> Target_Output;

public:
	//State-Space NN
	int Quantity_InputNode_SS;//输入端
	int Quantity_OutputNode_SS;//输出端
	int Term_Num_SS;//次

	//Output-Space NN
	int Quantity_InputNode_OS;//输入端
	int Quantity_OutputNode_OS;//输出端
	int Term_Num_OS;//次

	vector<double> InputNode_Value_vector;//State-Space NN的初始值
	vector<double> Input_Control_vector;//每一个时刻输入的控制u的值

	vector<CNeural_unit> State_Space_NN;
	vector<CNeural_unit> Output_Space_NN;

	CNeural_unit State_Space_NN_Real;
	CNeural_unit Output_Space_NN_Real;

	vector<vector<double>> State_Space_Neural_Weight_Reserved;//保存上一步NN权值SS
	vector<vector<double>> Output_Space_Neural_Weight_Reserved;//保存上一步NN权值OS

	vector<vector<double>> gradient_vector_StateSpace_Reserved;//保存上一步NN梯度SS
	vector<vector<double>> gradient_vector_OutputSpace_Reserved;//保存上一步NN梯度SS

	double Learning_rate_min;//最小学习率
	double Learning_Rate_NN;//学习率

	double Error_Function_Value;//误差函数数值
	double Error_Function_Value_Temp;//暂时误差函数数值
	double Error_Accuracy;//误差精度
	int Max_Episode;//最大更新循环次数
	int Episode_num;//迭代更新次数

	CModel m_model;

	FILE * file_gradient_ss;//梯度数据文件
	FILE * file_gradient_os;

	FILE * file_errorDelta_1_Out;//残差
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

	void Update_Weight();//更新权值
	void Cal_Gradient();//计算梯度
	double Cal_Error_Value_Temp();//暂时计算输出误差
	void Cal_Error_Function_Value();//计算误差函数数值
	void Unfold_Neural_Network();//展开神经网络
	void Data_Feedforward();//数据向前传递
	void Initial_Unfolding_Neural_Network();//展开RNN
	void File_Operate();//文件操作
	void File_Initail();//文件初始化
	void Recurrent_Initial_NN_Weight();//初始化权值
	void Update_learning_rate(int step);//更新学习率

};

