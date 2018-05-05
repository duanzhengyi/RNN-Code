#include "stdafx.h"


CRNN_Simu::CRNN_Simu()
{
	
}


CRNN_Simu::~CRNN_Simu()
{
}

void CRNN_Simu::simulate()
{
	Initial_Simu();
	Recurrent_Initial_NN_Weight();//利用一步预测初始化N步预测的NN权值
	Simulate_Error_Backpropogation();
}

void CRNN_Simu::Recurrent_Initial_NN_Weight()
{
	//暂时设定一步预测步长
	int temp_predictive_step = Predictive_Step;
	Learning_Rate_NN = 0.005;
	for (int i=0;i<temp_predictive_step -1;i++)
	{
		Predictive_Step = i+1;
		while (Error_Function_Value > Error_Accuracy && Episode_num < Max_Episode)
		{
			Unfold_Neural_Network();
			Cal_Error_Function_Value();//计算误差
			Calculate_Error();//计算残差
			Cal_Gradient();//计算梯度
			Update_learning_rate(Episode_num);//更新学习率
			Update_Weight();//更新权值
			Episode_num += 1;
		}
		Episode_num = 0;
		Error_Function_Value = Error_Accuracy + 1;
	}
	
	Predictive_Step = temp_predictive_step;//恢复原始预测步长
	
}

void CRNN_Simu::Update_learning_rate(int step)
{
	if (step > 0)
	{
		//权值范数
		//State-Space
		double sum_nn = 0;
		for (int i = 0; i<int(State_Space_NN_Real.Neural_Weight.size()); i++)
		{
			for (int j = 0; j<int(State_Space_NN_Real.Neural_Weight[i].size()); j++)
			{
				sum_nn = sum_nn + pow(State_Space_Neural_Weight_Reserved[i][j] - State_Space_NN_Real.Neural_Weight[i][j], 2);
			}
		}

		//Output-Space
		for (int i = 0; i<int(Output_Space_NN_Real.Neural_Weight.size()); i++)
		{
			for (int j = 0; j<int(Output_Space_NN_Real.Neural_Weight[i].size()); j++)
			{
				sum_nn = sum_nn + pow(Output_Space_Neural_Weight_Reserved[i][j] - Output_Space_NN_Real.Neural_Weight[i][j], 2);
			}
		}
		double weight_norm = sqrt(sum_nn);

		//梯度范数
		//State-Space
		double sum_gradient = 0;
		for (int i = 0; i<int(gradient_vector_StateSpace_Reserved.size()); i++)
		{
			for (int j = 0; j<int(gradient_vector_StateSpace_Reserved[i].size()); j++)
			{
				sum_gradient = sum_gradient + pow(gradient_vector_StateSpace_Reserved[i][j] - gradient_vector_StateSpace[i][j], 2);
			}
		}

		//Output-Space
		for (int i = 0; i<int(gradient_vector_OutputSpace_Reserved.size()); i++)
		{
			for (int j = 0; j<int(gradient_vector_OutputSpace_Reserved[i].size()); j++)
			{
				sum_gradient = sum_gradient + pow(gradient_vector_OutputSpace_Reserved[i][j] - gradient_vector_OutputSpace[i][j], 2);
			}
		}
		double gradient_norm = sqrt(sum_gradient);

		Learning_Rate_NN = 0.5*weight_norm / gradient_norm;
	}
	
// 	while (Learning_Rate_NN < Learning_rate_min)
// 	{
// 		Learning_Rate_NN = Learning_Rate_NN * 2;
// 	}

	//梯度范数
	double gradient_sum_one = 0;
	for (int i=0;i<int(gradient_vector_OutputSpace.size());i++)
	{
		for (int j=0;j<int(gradient_vector_OutputSpace[i].size());j++)
		{
			gradient_sum_one = gradient_sum_one + pow(gradient_vector_OutputSpace[i][j], 2);
		}
	}

	while ((Cal_Error_Value_Temp()-Error_Function_Value)>(-0.5*Learning_Rate_NN*gradient_sum_one))
	{
		double ee = Cal_Error_Value_Temp();
		Learning_Rate_NN = Learning_Rate_NN / 2;
	}

}

void CRNN_Simu::Simulate_Error_Backpropogation()
{
	while (Error_Function_Value > Error_Accuracy && Episode_num < Max_Episode)
	{
		Unfold_Neural_Network();
		Cal_Error_Function_Value();//计算误差
		Calculate_Error();//计算残差
		Cal_Gradient();//计算梯度
//		File_Operate();//输出数据
		Update_learning_rate(Episode_num);//更新学习率
		Update_Weight();//更新权值
		Episode_num += 1;
	}
}

void CRNN_Simu::Initial_Simu()
{
	Initial_Data();
	File_Initail();//初始化文件
	Initial_NeuralNetwork();
}

void CRNN_Simu::Initial_Data()
{
	Predictive_Step = 10;//预测步长

	//State-Space NN
	Quantity_InputNode_SS = 3;//输入端
	Quantity_OutputNode_SS = 2;//输出端
	Term_Num_SS = 2;//次

	//Output-Space NN
	Quantity_InputNode_OS = 2;//输入端
	Quantity_OutputNode_OS = 2;//输出端
	Term_Num_OS = 2;//次

	//State-Space-NN 初始值
	InputNode_Value_vector.push_back(1.0);
	InputNode_Value_vector.push_back(1.0);
	InputNode_Value_vector.push_back(1.0);

	//每一时刻控制u的值，默认为1
	for (int i=0;i<Predictive_Step;i++)
	{
		Input_Control_vector.push_back(1);
	}

	vector<double> Input_Model_State;
	Input_Model_State.push_back(1.0);
	Input_Model_State.push_back(1.0);
	vector<vector<double>> Input_Model_Control;
	vector<double> control_value_temp;
	control_value_temp.push_back(1.0);
	for (int i=0;i<Predictive_Step;i++)
	{
		Input_Model_Control.push_back(control_value_temp);
	}
	m_model.Initial_Data(Predictive_Step, Quantity_OutputNode_SS, Quantity_InputNode_SS-Quantity_OutputNode_SS, Quantity_OutputNode_OS, Input_Model_State, Input_Model_Control);
	m_model.Model_Operation();
	Target_Output.resize(Predictive_Step);
	Target_Output = m_model.output_vector;

	//学习率
	Learning_rate_min = 0.00001;

	//初始迭代次数
	Episode_num = 0;

	//误差精度
	Error_Accuracy = 0.1;
	Error_Function_Value = Error_Accuracy + 1;
	Error_Function_Value_Temp = Error_Accuracy + 1;
	
	//最大更新循环次数
	Max_Episode = 30000;

	//初始化残差vector大小
	errorDelta_2_Out.resize(Predictive_Step);
	errorDelta_2_Hid.resize(Predictive_Step);
	errorDelta_1_Out.resize(Predictive_Step);
	errorDelta_1_Hid.resize(Predictive_Step);

}

void CRNN_Simu::Create_Real_NeuralNetwork()
{
	//State-Space NN
	Nerual_role nn_role;
	nn_role = State_Space;

	State_Space_NN_Real.InputNode_Quantity = Quantity_InputNode_SS;
	State_Space_NN_Real.InputNode_Value = InputNode_Value_vector;
	State_Space_NN_Real.OutputNode_Quantity = Quantity_OutputNode_SS;
	State_Space_NN_Real.Term_Number = Term_Num_SS;
	State_Space_NN_Real.NN_Role = nn_role;
	State_Space_NN_Real.Initial_NeuralNetwork();

	//State_Space_Neural_Weight_Reserved.resize();

	//Output-Space NN
	nn_role = Output_Space;
	vector<double> InputValue_vector_one;
	for (int j = 0; j < Quantity_OutputNode_SS; j++)
	{
		InputValue_vector_one.push_back(State_Space_NN_Real.OutputNode_Value[j]);//上一个NN的输出值
	}

	Output_Space_NN_Real.InputNode_Quantity = Quantity_InputNode_OS;
	Output_Space_NN_Real.InputNode_Value = InputValue_vector_one;
	Output_Space_NN_Real.OutputNode_Quantity = Quantity_OutputNode_OS;
	Output_Space_NN_Real.Term_Number = Term_Num_OS;
	Output_Space_NN_Real.NN_Role = nn_role;
	Output_Space_NN_Real.Initial_NeuralNetwork();

}

void CRNN_Simu::Initial_NeuralNetwork()
{
	//建立真实的NN
	Create_Real_NeuralNetwork();

	//建立unfolding-NN
	//State-Space NN
	for (int i=0;i<Predictive_Step;i++)
	{
		Nerual_role nn_role;
		nn_role = State_Space;

		if (i==0)
		{			
			CNeural_unit m_StateSpace_NN(Quantity_InputNode_SS, InputNode_Value_vector, Quantity_OutputNode_SS, Term_Num_SS, nn_role);
			m_StateSpace_NN.Neural_Weight = State_Space_NN_Real.Neural_Weight;//初始化：用统一的weight
			State_Space_NN.push_back(m_StateSpace_NN);
		}
		else {			
			vector<double> InputValue_vector_one;
			for (int j = 0; j < Quantity_OutputNode_SS; j++)
			{
				InputValue_vector_one.push_back(State_Space_NN[i - 1].OutputNode_Value[j]);//上一个NN的输出值
			}
			InputValue_vector_one.push_back(Input_Control_vector[i]);
			CNeural_unit m_StateSpace_NN(Quantity_InputNode_SS, InputValue_vector_one, Quantity_OutputNode_SS, Term_Num_SS, nn_role);	
			m_StateSpace_NN.Neural_Weight = State_Space_NN_Real.Neural_Weight;//初始化：用统一的weight
			State_Space_NN.push_back(m_StateSpace_NN);
		}		
	}

	//Output-Space NN
	for (int i = 0; i <Predictive_Step;i++)
	{
		Nerual_role nn_role;
		nn_role = Output_Space;
		vector<double> InputValue_vector_one;
		for (int j = 0; j < Quantity_OutputNode_SS; j++)
		{
			InputValue_vector_one.push_back(State_Space_NN[i].OutputNode_Value[j]);//上一个NN的输出值
		}
		CNeural_unit m_OutputSpace_NN(Quantity_InputNode_OS,InputValue_vector_one, Quantity_OutputNode_OS, Term_Num_OS,nn_role);		
		m_OutputSpace_NN.Neural_Weight = Output_Space_NN_Real.Neural_Weight;//初始化：用统一的weight
		Output_Space_NN.push_back(m_OutputSpace_NN);
	}

}

void CRNN_Simu::Calculate_Error()
{
	//Output-Space NN
	//Output-Layer
	for (int i = (Predictive_Step-1); i > -1; i--) {
		vector<double> value_oneVector;
		for (int j=0;j<Output_Space_NN[i].OutputNode_Quantity;j++)
		{
			value_oneVector.push_back(Target_Output[i][j] - Output_Space_NN[i].OutputNode_Value[j]);
		}
		errorDelta_2_Out[i]=value_oneVector;
	}

	//Hidden-Layer
	for (int k=(Predictive_Step-1);k>-1;k--)
	{
		vector<double> value_oneVector;
		for (int i=0;i<Output_Space_NN[k].HiddenNode_Quantity;i++)
		{			
			double sum_one=0;
			for (int j=0;j<Output_Space_NN[k].OutputNode_Quantity;j++)
			{
				sum_one = sum_one + errorDelta_2_Out[k][j]* Output_Space_NN[k].Neural_Weight[i+1][j];//Neural_Weight数组需要先去除常数项
			}
			value_oneVector.push_back(sum_one);
		}
		errorDelta_2_Hid[k]=value_oneVector;
	}

	//State-Space NN
	for (int k=(Predictive_Step-1);k>-1;k--)
	{
		NN_1_OutputLayer_CalError(k);
		NN_1_HiddenLayer_CalError(k);
	}

}

void CRNN_Simu::NN_1_OutputLayer_CalError(int index_step)
{
	int k = index_step;
	//Output-Layer
	vector<double> value_oneVec;

	if (k == Predictive_Step - 1)
	{
		for (int i = 0; i < State_Space_NN[k].OutputNode_Quantity; i++)
		{
			double sum_one = 0;
			double sum_two = 0;
			vector<int> vector_two = Output_Space_NN[k].InputNode_LineIndex[i];
			//k时刻: 2-NN传入的error
			for (int j = 0; j<int(vector_two.size()); j++)
			{
				sum_two = sum_two + errorDelta_2_Hid[k][vector_two[j]];
			}
			sum_one = sum_two;
			value_oneVec.push_back(sum_one);
		}
	}
	else {

		for (int i = 0; i < State_Space_NN[k].OutputNode_Quantity; i++)
		{
			double sum_one = 0;
			double sum_two = 0;
			vector<int> vector_two = Output_Space_NN[k].InputNode_LineIndex[i];
			//k时刻: 2-NN传入的error
			for (int j = 0; j<int(vector_two.size()); j++)
			{
				sum_two = sum_two + errorDelta_2_Hid[k][vector_two[j]];
			}

			//k+1时刻: 1-NN传入的error
			double sum_three = 0;
			vector<int> vector_three = State_Space_NN[k + 1].InputNode_LineIndex[i];
			for (int j = 0; j<int(vector_three.size()); j++)
			{
				sum_three = sum_three + errorDelta_1_Hid[k + 1][vector_three[j]];
			}

			sum_one = sum_two + sum_three;
			value_oneVec.push_back(sum_one);
		}

	}
	errorDelta_1_Out[k] = value_oneVec;
}

void CRNN_Simu::NN_1_HiddenLayer_CalError(int index_step)
{
	int k = index_step;
	vector<double> value_oneVector;
	for (int i = 0; i < State_Space_NN[k].HiddenNode_Quantity; i++)
	{
		double sum_one = 0;
		for (int j = 0; j < State_Space_NN[k].OutputNode_Quantity; j++)
		{
			sum_one = sum_one + errorDelta_1_Out[k][j] * State_Space_NN[k].Neural_Weight[i + 1][j];//Neural_Weight数组需要先去除常数项
		}
		value_oneVector.push_back(sum_one);
	}
	errorDelta_1_Hid[k] = value_oneVector;
}

void CRNN_Simu::Update_Weight()
{	
	//首先保存上一步的权重值
	Output_Space_Neural_Weight_Reserved = Output_Space_NN_Real.Neural_Weight;
	State_Space_Neural_Weight_Reserved = State_Space_NN_Real.Neural_Weight;

	for (int i=0;i<int(gradient_vector_OutputSpace.size());i++)
	{
		for (int j=0;j<int(gradient_vector_OutputSpace[i].size());j++)
		{
			Output_Space_NN_Real.Neural_Weight[i][j] = Output_Space_NN_Real.Neural_Weight[i][j] + Learning_Rate_NN * gradient_vector_OutputSpace[i][j];
		}
	}

	for (int i=0;i<int(gradient_vector_StateSpace.size());i++)
	{
		for (int j=0;j<int(gradient_vector_StateSpace[i].size());j++)
		{
			State_Space_NN_Real.Neural_Weight[i][j] = State_Space_NN_Real.Neural_Weight[i][j] + Learning_Rate_NN * gradient_vector_StateSpace[i][j];
		}
	}
	
}

void CRNN_Simu::Cal_Gradient()
{
	//保存上一步的梯度值
	gradient_vector_StateSpace_Reserved = gradient_vector_StateSpace;
	gradient_vector_OutputSpace_Reserved = gradient_vector_OutputSpace;

	//梯度vector清空
	gradient_vector_StateSpace.clear();
	gradient_vector_OutputSpace.clear();

	//Output-Space weight
	//常数项
	vector<double> one_ve;
	for (int j = 0; j < Output_Space_NN_Real.OutputNode_Quantity; j++)
	{
		double sum_change = 0;
		for (int k = 0; k < Predictive_Step; k++)
		{
			sum_change = sum_change + errorDelta_2_Out[k][j];
		}
		one_ve.push_back(sum_change);
	}
	gradient_vector_OutputSpace.push_back(one_ve);

	//非常数项
	for (int i = 0; i < Output_Space_NN_Real.HiddenNode_Quantity; i++)
	{
		vector<double> one_ve;
		for (int j = 0; j < Output_Space_NN_Real.OutputNode_Quantity; j++)
		{
			double sum_change = 0;
			for (int k = 0; k < Predictive_Step; k++)
			{
				sum_change = sum_change + errorDelta_2_Out[k][j] * Output_Space_NN[k].s_value[i];
			}
			one_ve.push_back(sum_change);			
		}
		gradient_vector_OutputSpace.push_back(one_ve);
	}

	//State-Space weight
	//常数项
	vector<double> two_ve;
	for (int j = 0; j < State_Space_NN_Real.OutputNode_Quantity; j++)
	{
		double sum_change = 0;
		for (int k = 0; k < Predictive_Step; k++)
		{
			sum_change = sum_change + errorDelta_1_Out[k][j];
		}
		two_ve.push_back(sum_change);
	}
	gradient_vector_StateSpace.push_back(two_ve);

	//非常数项
	for (int i = 0; i < State_Space_NN_Real.HiddenNode_Quantity; i++)
	{
		vector<double> one_ve;
		for (int j = 0; j < State_Space_NN_Real.OutputNode_Quantity; j++)
		{
			double sum_change = 0;
			for (int k = 0; k < Predictive_Step; k++)
			{
				sum_change = sum_change + errorDelta_1_Out[k][j] * State_Space_NN[k].s_value[i];
			}
			one_ve.push_back(sum_change);			
		}
		gradient_vector_StateSpace.push_back(one_ve);
	}

}

double CRNN_Simu::Cal_Error_Value_Temp()
{
	vector<vector<double>> State_Space_Neural_Weight_Temp = State_Space_NN_Real.Neural_Weight;//暂时保存NN权值SS
	vector<vector<double>> Output_Space_Neural_Weight_Temp = Output_Space_NN_Real.Neural_Weight;//暂时保存NN权值OS

	//Output-Space weight
	//常数项
	for (int j = 0; j < Output_Space_NN_Real.OutputNode_Quantity; j++)
	{
		double sum_change = 0;
		for (int k = 0; k < Predictive_Step; k++)
		{
			sum_change = sum_change + errorDelta_2_Out[k][j];
		}
		Output_Space_NN_Real.Neural_Weight[0][j] = Output_Space_NN_Real.Neural_Weight[0][j] + Learning_Rate_NN * sum_change;
	}
	
	//非常数项
	for (int i = 0; i < Output_Space_NN_Real.HiddenNode_Quantity; i++)
	{
		for (int j = 0; j < Output_Space_NN_Real.OutputNode_Quantity; j++)
		{
			double sum_change = 0;
			for (int k = 0; k < Predictive_Step; k++)
			{
				sum_change = sum_change + errorDelta_2_Out[k][j] * Output_Space_NN[k].s_value[i];
			}
			Output_Space_NN_Real.Neural_Weight[i + 1][j] = Output_Space_NN_Real.Neural_Weight[i + 1][j] + Learning_Rate_NN * sum_change;
		}		
	}

	//State-Space weight
	//常数项
	for (int j = 0; j < State_Space_NN_Real.OutputNode_Quantity; j++)
	{
		double sum_change = 0;
		for (int k = 0; k < Predictive_Step; k++)
		{
			sum_change = sum_change + errorDelta_1_Out[k][j];
		}
		State_Space_NN_Real.Neural_Weight[0][j] = State_Space_NN_Real.Neural_Weight[0][j] + Learning_Rate_NN * sum_change;
	}

	//非常数项
	for (int i = 0; i < State_Space_NN_Real.HiddenNode_Quantity; i++)
	{
		for (int j = 0; j < State_Space_NN_Real.OutputNode_Quantity; j++)
		{
			double sum_change = 0;
			for (int k = 0; k < Predictive_Step; k++)
			{
				sum_change = sum_change + errorDelta_1_Out[k][j] * State_Space_NN[k].s_value[i];
			}
			State_Space_NN_Real.Neural_Weight[i + 1][j] = State_Space_NN_Real.Neural_Weight[i + 1][j] + Learning_Rate_NN * sum_change;
		}
	}

	Unfold_Neural_Network();

	double Error_one = Error_Function_Value;//暂时保存误差值
	Cal_Error_Function_Value();//计算误差
	Error_Function_Value_Temp = Error_Function_Value;
	
	State_Space_NN_Real.Neural_Weight = State_Space_Neural_Weight_Temp;//恢复NN权值SS
	Output_Space_NN_Real.Neural_Weight = Output_Space_Neural_Weight_Temp;//恢复NN权值OS
	
	Error_Function_Value = Error_one;//恢复系统误差函数值

	return Error_Function_Value_Temp;
}

void CRNN_Simu::Cal_Error_Function_Value()
{
	double sum_error = 0;
	for (int i = 0; i < Predictive_Step; i++) {
		for (int j = 0; j < Output_Space_NN[i].OutputNode_Quantity; j++)
		{
			double one = Target_Output[i][j] - Output_Space_NN[i].OutputNode_Value[j];
			sum_error = sum_error + 0.5*pow(one,2);
		}
	}
	Error_Function_Value = sum_error;
}

void CRNN_Simu::Unfold_Neural_Network()
{
	Initial_Unfolding_Neural_Network();
	Data_Feedforward();
}

void CRNN_Simu::Data_Feedforward()
{
	for (int i = 0; i < Predictive_Step; i++)
	{		
		if (i == 0)
		{
			State_Space_NN[i].InputNode_Value = InputNode_Value_vector;
			State_Space_NN[i].Data_FeedForward();
		}
		else {
			vector<double> InputValue_vector_one;
			for (int j = 0; j < Quantity_OutputNode_SS; j++)
			{
				InputValue_vector_one.push_back(State_Space_NN[i - 1].OutputNode_Value[j]);//上一个NN的输出值
			}
			InputValue_vector_one.push_back(Input_Control_vector[i]);
			State_Space_NN[i].InputNode_Value = InputValue_vector_one;
			State_Space_NN[i].Data_FeedForward();
		}
	}

	//Output-Space NN
	for (int i = 0; i < Predictive_Step; i++)
	{
		vector<double> InputValue_vector_one;
		for (int j = 0; j < Quantity_OutputNode_SS; j++)
		{
			InputValue_vector_one.push_back(State_Space_NN[i].OutputNode_Value[j]);//上一个NN的输出值
		}		
		Output_Space_NN[i].InputNode_Value = InputValue_vector_one;
		Output_Space_NN[i].Data_FeedForward();
	}

}

void CRNN_Simu::Initial_Unfolding_Neural_Network()
{
	for (int i=0;i<int(State_Space_NN.size());i++)
	{
		State_Space_NN[i].Neural_Weight = State_Space_NN_Real.Neural_Weight;
	}

	for (int i=0;i<int(Output_Space_NN.size());i++)
	{
		Output_Space_NN[i].Neural_Weight = Output_Space_NN_Real.Neural_Weight;
	}
}

void CRNN_Simu::File_Operate()
{
	//state-space的梯度
	fprintf(file_gradient_ss, "\n\n第%d步循环\n", Episode_num);
	for (int i=0;i<int(gradient_vector_StateSpace.size());i++)
	{
		for (int j=0;j<int(gradient_vector_StateSpace[i].size());j++)
		{
			fprintf(file_gradient_ss, "[%d][%d]%.1f\t", i,j,gradient_vector_StateSpace[i][j]);
		}
		fprintf(file_gradient_ss, "\n");
	}

	//output-space的梯度
	fprintf(file_gradient_os, "\n\n第%d步循环\n", Episode_num);
	for (int i = 0; i<int(gradient_vector_OutputSpace.size()); i++)
	{
		for (int j = 0; j < int(gradient_vector_OutputSpace[i].size()); j++)
		{
			fprintf(file_gradient_os, "[%d][%d]%.1f\t", i,j,gradient_vector_OutputSpace[i][j]);
		}
		fprintf(file_gradient_os, "\n");
	}

	//state-space的weight
	fprintf(file_ss_nn_weight, "\n\n第%d步循环\n", Episode_num);
	for (int i=0;i<int(State_Space_NN_Real.Neural_Weight.size());i++)
	{
		for (int j=0;j<int(State_Space_NN_Real.Neural_Weight[i].size());j++)
		{
			fprintf(file_ss_nn_weight, "[%d][%d]%.1f\t", i,j,State_Space_NN_Real.Neural_Weight[i][j]);
		}
		fprintf(file_ss_nn_weight, "\n");
	}

	//output-space的weight
	fprintf(file_os_nn_weight, "\n\n第%d步循环\n", Episode_num);
	for (int i = 0; i<int(Output_Space_NN_Real.Neural_Weight.size()); i++)
	{
		for (int j = 0; j<int(Output_Space_NN_Real.Neural_Weight[i].size()); j++)
		{
			fprintf(file_os_nn_weight, "[%d][%d]%.1f\t", i,j,Output_Space_NN_Real.Neural_Weight[i][j]);
		}
		fprintf(file_os_nn_weight, "\n");
	}

	//output
	fprintf(file_nn_output, "\n\n第%d步循环", Episode_num);
	for (int k=0;k<int(Output_Space_NN.size());k++)
	{
		fprintf(file_nn_output, "\n第%d个NN输出\t\n", k);
		for (int i = 0; i<int(Output_Space_NN[k].OutputNode_Value.size());i++)
		{
			fprintf(file_nn_output, "%.1f\t", Output_Space_NN[k].OutputNode_Value[i]);
		}
	}

	//errorDelta
	fprintf(file_errorDelta_1_Hid, "\n\n第%d步循环", Episode_num);
	for (int i=0;i<int(errorDelta_1_Hid.size());i++)
	{
		fprintf(file_errorDelta_1_Hid, "\n第%d个NN输出\t\n", i);
		for (int j=0;j<int(errorDelta_1_Hid[i].size());j++)
		{
			fprintf(file_errorDelta_1_Hid, "[%d][%d]%.1f\t", i,j,errorDelta_1_Hid[i][j]);
		}
	}

	fprintf(file_errorDelta_1_Out, "\n\n第%d步循环", Episode_num);
	for (int i = 0; i<int(errorDelta_1_Out.size()); i++)
	{
		fprintf(file_errorDelta_1_Out, "\n第%d个NN输出\t\n", i);
		for (int j = 0; j<int(errorDelta_1_Out[i].size()); j++)
		{
			fprintf(file_errorDelta_1_Out, "[%d][%d]%.1f\t", i,j,errorDelta_1_Out[i][j]);
		}
	}

	fprintf(file_errorDelta_2_Hid, "\n\n第%d步循环", Episode_num);
	for (int i = 0; i<int(errorDelta_2_Hid.size()); i++)
	{
		fprintf(file_errorDelta_2_Hid, "\n第%d个NN输出\t\n", i);
		for (int j = 0; j<int(errorDelta_2_Hid[i].size()); j++)
		{
			fprintf(file_errorDelta_2_Hid, "[%d][%d]%.1f\t", i,j,errorDelta_2_Hid[i][j]);
		}
	}

	fprintf(file_errorDelta_2_Out, "\n\n第%d步循环", Episode_num);
	for (int i = 0; i<int(errorDelta_2_Out.size()); i++)
	{
		fprintf(file_errorDelta_2_Out, "\n第%d个NN输出\t\n", i);
		for (int j = 0; j<int(errorDelta_2_Out[i].size()); j++)
		{
			fprintf(file_errorDelta_2_Out, "[%d][%d]%.1f\t", i,j,errorDelta_2_Out[i][j]);
		}
	}

}

void CRNN_Simu::File_Initail()
{
	string file_address = "E:\\data_one\\";
	fopen_s(&file_gradient_ss, (file_address + "gradient_state_space.dat").c_str(), "w");
	fopen_s(&file_gradient_os, (file_address + "gradient_output_space.dat").c_str(), "w");
	fopen_s(&file_ss_nn_weight, (file_address + "weight_state_space.dat").c_str(), "w");
	fopen_s(&file_os_nn_weight, (file_address + "weight_output_space.dat").c_str(), "w");
	fopen_s(&file_nn_output, (file_address + "output.dat").c_str(), "w");
	fopen_s(&file_errorDelta_1_Out, (file_address + "errorDelta_1_Out.dat").c_str(), "w");
	fopen_s(&file_errorDelta_1_Hid, (file_address + "errorDelta_1_Hid.dat").c_str(), "w");
	fopen_s(&file_errorDelta_2_Out, (file_address + "errorDelta_2_Out.dat").c_str(), "w");
	fopen_s(&file_errorDelta_2_Hid, (file_address + "errorDelta_2_Hid.dat").c_str(), "w");
}


