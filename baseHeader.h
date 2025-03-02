#ifndef __BASE_HEADER__
#define __BASE_HEADER__

#include <stdio.h>
#include <Windows.h>
#include <math.h>
#include <time.h>

#define IN
#define OUT
#define IN_OUT

#define INPUT_SIZE (784)

#define HIDDEN_LAYER_0_CELL_COUNT (1024)

#define HIDDEN_LAYER_1_CELL_COUNT (512)

#define OUTPUT_LAYER_CELL_COUNT (10)

#define LEARNING_RATE (0.01)

#define TRAIN_IMAGE_COUNT (60000)

#define EPOCH (10)

#define ckDouble(sen) printf("%s -> %lf\n", #sen, (sen));

typedef struct _cell_layer_0
{
	double weightsUsedByFront[INPUT_SIZE];
	double bias;
	double (*activate)(double);
	double output;
	double error;
}HIDDEN_LAYER_0_CELL_INFO, * PHIDDEN_LAYER_0_CELL_INFO;

typedef struct _cell_layer_1
{
	double weightsUsedByFront[HIDDEN_LAYER_0_CELL_COUNT];
	double bias;
	double (*activate)(double);
	double output;
	double error;
}HIDDEN_LAYER_1_CELL_INFO, * PHIDDEN_LAYER_1_CELL_INFO;

typedef struct _result_layer
{
	double weightsUsedByFront[HIDDEN_LAYER_1_CELL_COUNT];
	double bias;
	double (*activate)(double);
	double output;
	double error;
}RESULT_LAYER_CELL_INFO, * PRESULT_LAYER_CELL_INFO;

void ExFreeMemory(
	PVOID* mem
);

double Relu(
	IN double x
);

double nullActivate(
	IN double x
);

void SoftMax(
	IN_OUT double (*array)[OUTPUT_LAYER_CELL_COUNT]
);

void initializeHiddenLayer0(
	OUT PHIDDEN_LAYER_0_CELL_INFO* hid0
);

void initializeHiddenLayer1(
	OUT PHIDDEN_LAYER_1_CELL_INFO* hid1
);

void initializeResultLayer(
	OUT PRESULT_LAYER_CELL_INFO* res
);

void forward(
	IN PUCHAR inputVector, // 输入向量（原始像素值，0-255）
	IN_OUT PHIDDEN_LAYER_0_CELL_INFO* hid0,
	IN_OUT PHIDDEN_LAYER_1_CELL_INFO* hid1,
	IN_OUT PRESULT_LAYER_CELL_INFO* res
);

void calculateError(
	IN_OUT PRESULT_LAYER_CELL_INFO* res,
	IN UCHAR teacherData // 真实标签 (0~9)
);

void backward(
	IN const double* normalizedInput, // 归一化后的输入数据
	IN_OUT PHIDDEN_LAYER_1_CELL_INFO* hid1,
	IN_OUT PHIDDEN_LAYER_0_CELL_INFO* hid0,
	IN_OUT PRESULT_LAYER_CELL_INFO* res,
	IN double learningRate
);

UCHAR* load_mnist_labels(
	const char* file_path
);

void train(
	IN CONST CHAR* trainSet,            // 训练集文件路径
	IN CONST CHAR* trainLabels,         // 训练集标签文件路径
	IN size_t epoch,                    // 训练轮次
	IN_OUT PHIDDEN_LAYER_0_CELL_INFO* hid0,
	IN_OUT PHIDDEN_LAYER_1_CELL_INFO* hid1,
	IN_OUT PRESULT_LAYER_CELL_INFO* res
);

int detect(
	IN CONST CHAR* testSet,             // 测试集文件路径
	IN size_t k,                        // 第 k 张图片（从 0 开始）
	IN_OUT PHIDDEN_LAYER_0_CELL_INFO* hid0,
	IN_OUT PHIDDEN_LAYER_1_CELL_INFO* hid1,
	IN_OUT PRESULT_LAYER_CELL_INFO* res
);

#endif