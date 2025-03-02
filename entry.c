#include "baseHeader.h"

int main()
{
	PHIDDEN_LAYER_0_CELL_INFO hid0 = NULL;
	initializeHiddenLayer0(&hid0);
	PHIDDEN_LAYER_1_CELL_INFO hid1 = NULL;
	initializeHiddenLayer1(&hid1);
	PRESULT_LAYER_CELL_INFO res = NULL;
	initializeResultLayer(&res);

	CONST CHAR* trainSet = "E:\\desk\\MNIST_data\\train-images.idx3-ubyte";
	CONST CHAR* trainLabels = "E:\\desk\\MNIST_data\\train-labels.idx1-ubyte";

	train(trainSet, trainLabels, EPOCH, &hid0, &hid1, &res);

	CONST CHAR* testSet = "E:\\desk\\MNIST_data\\t10k-images.idx3-ubyte";
	CONST CHAR* testLabels = "E:\\desk\\MNIST_data\\t10k-labels.idx1-ubyte";

	UCHAR* trueLabels = load_mnist_labels(testLabels);

	for (size_t j = 0; j < 2000; j++)
	{
		int predictedClass = detect(testSet, j, &hid0, &hid1, &res);
		printf("第 %zu 张测试图片 - 预测值: %d, 真实值: %d \t\t 差值: %d\n", j, predictedClass, trueLabels[j], predictedClass - trueLabels[j]);
	}

	ExFreeMemory((PVOID*)&trueLabels);
	ExFreeMemory((PVOID*)&res);
	ExFreeMemory((PVOID*)&hid1);
	ExFreeMemory((PVOID*)&hid0);

	return 0;
}