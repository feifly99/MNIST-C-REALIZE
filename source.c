#include "baseHeader.h"

#pragma warning (disable: 6011)
#pragma warning (disable: 6387)

void ExFreeMemory(
	PVOID* mem
)
{
	free(*mem);
	*mem = NULL;
	return;
}

double Relu(
	IN double x
)
{
	return (x >= 0) ? x : 0;
}

double nullActivate(
	IN double x
)
{
	return x;
}

void SoftMax(
	IN_OUT double (*array)[OUTPUT_LAYER_CELL_COUNT]
)
{
	double maxVal = (*array)[0];
	double sum = 0.0;

	for (size_t j = 1; j < OUTPUT_LAYER_CELL_COUNT; j++)
	{
		if ((*array)[j] > maxVal)
		{
			maxVal = (*array)[j];
		}
	}

	for (size_t j = 0; j < OUTPUT_LAYER_CELL_COUNT; j++)
	{
		(*array)[j] = exp((*array)[j] - maxVal);
		sum += (*array)[j];
	}

	for (size_t j = 0; j < OUTPUT_LAYER_CELL_COUNT; j++)
	{
		(*array)[j] /= sum;
	}

	return;
}

#include <math.h>

void initializeHiddenLayer0(
	OUT PHIDDEN_LAYER_0_CELL_INFO* hid0
)
{
	*hid0 = (PHIDDEN_LAYER_0_CELL_INFO)malloc(HIDDEN_LAYER_0_CELL_COUNT * sizeof(HIDDEN_LAYER_0_CELL_INFO));
	RtlZeroMemory(*hid0, HIDDEN_LAYER_0_CELL_COUNT * sizeof(HIDDEN_LAYER_0_CELL_INFO));

	for (size_t i = 0; i < HIDDEN_LAYER_0_CELL_COUNT; i++)
	{
		for (SIZE_T j = 0; j < INPUT_SIZE; j++)
		{
			(*hid0)[i].weightsUsedByFront[j] = 1.0 / (double)INPUT_SIZE;
		}
		(*hid0)[i].bias = (double)rand() / (double)RAND_MAX * 0.01; // 偏置初始化为 0
		(*hid0)[i].activate = Relu;
		(*hid0)[i].output = 0.0;
		(*hid0)[i].error = 0.0;
	}
	return;
}

void initializeHiddenLayer1(
	OUT PHIDDEN_LAYER_1_CELL_INFO* hid1
)
{
	*hid1 = (PHIDDEN_LAYER_1_CELL_INFO)malloc(HIDDEN_LAYER_1_CELL_COUNT * sizeof(HIDDEN_LAYER_1_CELL_INFO));
	RtlZeroMemory(*hid1, HIDDEN_LAYER_1_CELL_COUNT * sizeof(HIDDEN_LAYER_1_CELL_INFO));

	for (size_t i = 0; i < HIDDEN_LAYER_1_CELL_COUNT; i++)
	{
		for (SIZE_T j = 0; j < HIDDEN_LAYER_0_CELL_COUNT; j++)
		{
			(*hid1)[i].weightsUsedByFront[j] = 1.0 / (double)HIDDEN_LAYER_0_CELL_COUNT;
		}
		(*hid1)[i].bias = (double)rand() / (double)RAND_MAX * 0.01; // 偏置初始化为 0
		(*hid1)[i].activate = Relu;
		(*hid1)[i].output = 0.0;
		(*hid1)[i].error = 0.0;
	}
	return;
}

void initializeResultLayer(
	OUT PRESULT_LAYER_CELL_INFO* res
)
{
	*res = (PRESULT_LAYER_CELL_INFO)malloc(OUTPUT_LAYER_CELL_COUNT * sizeof(RESULT_LAYER_CELL_INFO));
	RtlZeroMemory(*res, OUTPUT_LAYER_CELL_COUNT * sizeof(RESULT_LAYER_CELL_INFO));

	for (size_t i = 0; i < OUTPUT_LAYER_CELL_COUNT; i++)
	{
		for (SIZE_T j = 0; j < HIDDEN_LAYER_1_CELL_COUNT; j++)
		{
			(*res)[i].weightsUsedByFront[j] = 1.0 / (double)HIDDEN_LAYER_1_CELL_COUNT;
		}
		(*res)[i].bias = (double)rand() / (double)RAND_MAX * 0.01;
		(*res)[i].activate = nullActivate;
		(*res)[i].output = 0.0;
		(*res)[i].error = 0.0;
	}
	return;
}

void forward(
	IN PUCHAR inputVector,
	IN_OUT PHIDDEN_LAYER_0_CELL_INFO* hid0,
	IN_OUT PHIDDEN_LAYER_1_CELL_INFO* hid1,
	IN_OUT PRESULT_LAYER_CELL_INFO* res
)
{
	double normalizedInput[INPUT_SIZE];
	for (size_t j = 0; j < INPUT_SIZE; j++)
	{
		normalizedInput[j] = (double)inputVector[j] / 255.0;
	}

	for (size_t i = 0; i < HIDDEN_LAYER_0_CELL_COUNT; i++)
	{
		double sum = 0.0;
		for (size_t j = 0; j < INPUT_SIZE; j++)
		{
			sum += normalizedInput[j] * (*hid0)[i].weightsUsedByFront[j];
		}
		sum += (*hid0)[i].bias;
		(*hid0)[i].output = (*hid0)[i].activate(sum);
	}

	for (size_t i = 0; i < HIDDEN_LAYER_1_CELL_COUNT; i++)
	{
		double sum = 0.0;
		for (size_t j = 0; j < HIDDEN_LAYER_0_CELL_COUNT; j++)
		{
			sum += (*hid0)[j].output * (*hid1)[i].weightsUsedByFront[j];
		}
		sum += (*hid1)[i].bias;
		(*hid1)[i].output = (*hid1)[i].activate(sum); 
	}

	double outputVector[OUTPUT_LAYER_CELL_COUNT];
	for (size_t i = 0; i < OUTPUT_LAYER_CELL_COUNT; i++)
	{
		double sum = 0.0;
		for (size_t j = 0; j < HIDDEN_LAYER_1_CELL_COUNT; j++)
		{
			sum += (*hid1)[j].output * (*res)[i].weightsUsedByFront[j];
		}
		sum += (*res)[i].bias;
		(*res)[i].output = (*res)[i].activate(sum);
	}

	for (size_t j = 0; j < OUTPUT_LAYER_CELL_COUNT; j++)
	{
		outputVector[j] = (*res)[j].output;
	}

	SoftMax(&outputVector);

	for (size_t i = 0; i < OUTPUT_LAYER_CELL_COUNT; i++)
	{
		(*res)[i].output = outputVector[i];
	}
}

void calculateError(
	IN_OUT PRESULT_LAYER_CELL_INFO* res,
	IN UCHAR teacherData 
)
{
	for (size_t i = 0; i < OUTPUT_LAYER_CELL_COUNT; i++)
	{
		double predicted = (*res)[i].output;
		double target = (i == teacherData) ? 1.0 : 0.0;
		(*res)[i].error = predicted - target;
	}
}

void backward(
	IN const double* normalizedInput, 
	IN_OUT PHIDDEN_LAYER_1_CELL_INFO* hid1,
	IN_OUT PHIDDEN_LAYER_0_CELL_INFO* hid0,
	IN_OUT PRESULT_LAYER_CELL_INFO* res,
	IN double learningRate
)
{
	for (size_t i = 0; i < HIDDEN_LAYER_1_CELL_COUNT; i++)
	{
		double errorSum = 0.0;

		for (size_t j = 0; j < OUTPUT_LAYER_CELL_COUNT; j++)
		{
			errorSum += (*res)[j].error * (*res)[j].weightsUsedByFront[i];
		}

		errorSum *= ((*hid1)[i].output > 0) ? 1.0 : 0.0;

		(*hid1)[i].error = errorSum;
	}

	for (size_t i = 0; i < HIDDEN_LAYER_0_CELL_COUNT; i++)
	{
		double errorSum = 0.0;

		for (size_t j = 0; j < HIDDEN_LAYER_1_CELL_COUNT; j++)
		{
			errorSum += (*hid1)[j].error * (*hid1)[j].weightsUsedByFront[i];
		}

		errorSum *= ((*hid0)[i].output > 0) ? 1.0 : 0.0;

		(*hid0)[i].error = errorSum;
	}

	for (size_t i = 0; i < OUTPUT_LAYER_CELL_COUNT; i++)
	{
		for (size_t j = 0; j < HIDDEN_LAYER_1_CELL_COUNT; j++)
		{
			double gradient = (*res)[i].error * (*hid1)[j].output;
			(*res)[i].weightsUsedByFront[j] -= learningRate * gradient;
		}
		(*res)[i].bias -= learningRate * (*res)[i].error;
	}

	for (size_t i = 0; i < HIDDEN_LAYER_1_CELL_COUNT; i++)
	{
		for (size_t j = 0; j < HIDDEN_LAYER_0_CELL_COUNT; j++)
		{
			double gradient = (*hid1)[i].error * (*hid0)[j].output;
			(*hid1)[i].weightsUsedByFront[j] -= learningRate * gradient;
		}
		(*hid1)[i].bias -= learningRate * (*hid1)[i].error;
	}

	for (size_t i = 0; i < HIDDEN_LAYER_0_CELL_COUNT; i++)
	{
		for (size_t j = 0; j < INPUT_SIZE; j++)
		{
			double gradient = (*hid0)[i].error * normalizedInput[j];
			(*hid0)[i].weightsUsedByFront[j] -= learningRate * gradient;
		}
		(*hid0)[i].bias -= learningRate * (*hid0)[i].error;
	}
}

UCHAR* load_mnist_labels(
	const char* file_path
)
{
	FILE* file = fopen(file_path, "rb");
	if (!file)
	{
		fprintf(stderr, "无法打开文件: %s\n", file_path);
		exit(EXIT_FAILURE);
	}

	fseek(file, 8, SEEK_SET);

	UCHAR* labels = (UCHAR*)malloc(10000 * sizeof(UCHAR));
	if (!labels)
	{
		fprintf(stderr, "内存分配失败\n");
		fclose(file);
		exit(EXIT_FAILURE);
	}

	size_t readSize = fread(labels, sizeof(UCHAR), 10000, file);
	if (readSize != 10000)
	{
		fprintf(stderr, "读取标签数据失败\n");
		fclose(file);
		free(labels);
		exit(EXIT_FAILURE);
	}

	fclose(file);
	return labels;
}

void train(
	IN CONST CHAR* trainSet,           
	IN CONST CHAR* trainLabels,         
	IN size_t epoch,                    
	IN_OUT PHIDDEN_LAYER_0_CELL_INFO* hid0,
	IN_OUT PHIDDEN_LAYER_1_CELL_INFO* hid1,
	IN_OUT PRESULT_LAYER_CELL_INFO* res
)
{
	FILE* trainFile = fopen(trainSet, "rb");
	FILE* labelFile = fopen(trainLabels, "rb");
	if (!trainFile || !labelFile)
	{
		fprintf(stderr, "无法打开训练集或标签文件\n");
		exit(EXIT_FAILURE);
	}

	fseek(trainFile, 16, SEEK_SET);

	fseek(labelFile, 8, SEEK_SET);

	UCHAR mnistVector[INPUT_SIZE] = { 0 };
	UCHAR label = 0;

	for (size_t e = 0; e < epoch; e++)
	{
		printf("Epoch %zu/%zu\n", e + 1, epoch);

		double totalError = 0.0;

		for (size_t k = 0; k < TRAIN_IMAGE_COUNT; k++)
		{ 
			size_t readSize = fread(mnistVector, sizeof(UCHAR), INPUT_SIZE, trainFile);
			if (readSize != INPUT_SIZE)
			{
				fprintf(stderr, "读取图片数据失败\n");
				fclose(trainFile);
				fclose(labelFile);
				exit(EXIT_FAILURE);
			}

			readSize = fread(&label, sizeof(UCHAR), 1, labelFile);
			if (readSize != 1)
			{
				fprintf(stderr, "读取标签数据失败\n");
				fclose(trainFile);
				fclose(labelFile);
				exit(EXIT_FAILURE);
			}

			forward(mnistVector, hid0, hid1, res);

			calculateError(res, label);

			for (size_t i = 0; i < OUTPUT_LAYER_CELL_COUNT; i++)
			{
				totalError += fabs((*res)[i].error); 
			}

			double normalizedInput[INPUT_SIZE];
			for (size_t j = 0; j < INPUT_SIZE; j++)
			{
				normalizedInput[j] = (double)mnistVector[j] / 255.0;
			}

			backward(normalizedInput, hid1, hid0, res, LEARNING_RATE);

			if (k % 1000 == 0)
			{
				printf("已训练 %zu 张图片\n", k);
			}
		}

		double averageError = totalError / (TRAIN_IMAGE_COUNT * OUTPUT_LAYER_CELL_COUNT);
		printf("Epoch %zu 平均误差: %.6f\n", e + 1, averageError);

		fseek(trainFile, 16, SEEK_SET);
		fseek(labelFile, 8, SEEK_SET);
	}

	fclose(trainFile);
	fclose(labelFile);

	return;
}

int detect(
	IN CONST CHAR* testSet,           
	IN size_t k,                    
	IN_OUT PHIDDEN_LAYER_0_CELL_INFO* hid0,
	IN_OUT PHIDDEN_LAYER_1_CELL_INFO* hid1,
	IN_OUT PRESULT_LAYER_CELL_INFO* res
)
{
	FILE* testFile = fopen(testSet, "rb");
	if (!testFile)
	{
		fprintf(stderr, "无法打开测试集文件\n");
		exit(EXIT_FAILURE);
	}

	fseek(testFile, 16, SEEK_SET);

	size_t offset = k * INPUT_SIZE; 
	fseek(testFile, (LONG)offset, SEEK_CUR);

	UCHAR mnistVector[INPUT_SIZE] = { 0 };
	size_t readSize = fread(mnistVector, sizeof(UCHAR), INPUT_SIZE, testFile);
	if (readSize != INPUT_SIZE)
	{
		fprintf(stderr, "读取图片数据失败\n");
		fclose(testFile);
		exit(EXIT_FAILURE);
	}

	fclose(testFile);

	forward(mnistVector, hid0, hid1, res);

	int predictedClass = 0;
	double maxProbability = (*res)[0].output;
	for (size_t i = 1; i < OUTPUT_LAYER_CELL_COUNT; i++)
	{
		if ((*res)[i].output > maxProbability)
		{
			maxProbability = (*res)[i].output;
			predictedClass = (INT)i;
		}
	}

	return predictedClass;
}