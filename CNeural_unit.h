#pragma once
class CNeural_unit
{
public:
	CNeural_unit(int InputNode_Quantity, vector<double> InputNode_Value, int OutputNode_Quantity, int Term_Number, Nerual_role NN_Role);
	CNeural_unit();
	~CNeural_unit();

public:
	int InputNode_Quantity;//输入节点数量
	vector<double> InputNode_Value;//输入节点数值

	int OutputNode_Quantity;//输出节点数量
	
public:
	int Term_Number;//MTN几次项拟合
	Nerual_role NN_Role;//NN充当的角色
	
public:
	vector<double> u_value;//相关节点输出的数值
	vector<double> s_value;
	vector<double> v_value;

	vector<int> SetQuantity;//每一项的数量
	vector<vector<int>> Term_NumVector;//每一个输入节点对应每一项的个数

public:
	vector<string> State_StringVector;//状态量的字符表示
	vector<string> HiddenState_StringVector;//隐节点的字符表示
	
	vector<double> OutputNode_Value;//输出节点数值
	int HiddenNode_Quantity;//隐节点数量
	vector<vector<double>> Neural_Weight;//权值
	int NeuralNetwork_Index;//该NN在unfold中的序号

public:
	vector<vector<int>> InputNode_LineIndex;//输入节点的对应连线序号

public:
	void Initial_NeuralNetwork();
	void Calculate_SetQuantity();
	void Calculate_WeightQuantity();
	void Calculate_LineIndex();
	void Generate_StateString();
	void Recuurent_Produce(int index,int start,vector<int> vector_one);
	void Data_FeedForward();//信息向前传递
	void Inital_OutputSize();//初始化输出变量空间大小

};

