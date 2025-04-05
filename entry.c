#include <stdio.h>
#include <math.h>
#include <time.h>
#include <Windows.h>

#define LEARNING_RATE 0.02f
#define VERIFY_BATCH 10000
short mark[VERIFY_BATCH] = { 0 };

typedef struct _hid1
{
    short eachCellInputSize;
    short cellsCount;
    float w[256][784];
    float bias[256];
    void (*activate)(IN float[], IN SIZE_T, IN OUT float (*)[]);
    float z[256];
    float out[256];
    float error[256];
}HID1, * PHID1;

typedef struct _hid2
{
    short eachCellInputSize;
    short cellsCount;
    float w[512][256];
    float bias[512];
    void (*activate)(IN float[], IN SIZE_T, IN OUT float (*)[]);
    float z[512];
    float out[512];
    float error[512];
}HID2, * PHID2;

typedef struct _hid3
{
    short eachCellInputSize;
    short cellsCount;
    float w[16][512];
    float bias[16];
    void (*activate)(IN float[], IN SIZE_T, IN OUT float (*)[]);
    float z[16];
    float out[16];
    float error[16];
}HID3, * PHID3;

typedef struct _out
{
    short eachCellInputSize;
    short cellsCount;
    float w[10][16];
    float bias[10];
    void (*activate)(IN float[], IN SIZE_T, IN OUT float (*)[]);
    float z[10];
    float out[10];
    float error[10];
}OUTL, * POUTL;

void allocateZeroMemory(
    IN OUT PVOID* mem,
    IN SIZE_T size
)
{
    *mem = malloc(size);
    RtlZeroMemory(*mem, size);
    return;
}

void ExFreeMemory(
    IN OUT PVOID* mem
)
{
    free(*mem);
    *mem = NULL;

    return;
}

void ReLU(
    IN float z[],
    IN SIZE_T size,
    IN OUT float (*out)[]
)
{
    for (size_t j = 0; j < size; j++)
    {
        (*out)[j] = (z[j] >= 0.0f) ? z[j] : 0.0f;
    }

    return;
}

void lekyReLU(
    IN float z[],
    IN SIZE_T size,
    IN OUT float (*out)[]
)
{
    for (size_t j = 0; j < size; j++)
    {
        (*out)[j] = (z[j] >= 0.0f) ? z[j] : -0.001f * z[j];
    }

    return;
}

void SoftMax(
    IN float z[],
    IN SIZE_T size,
    IN OUT float (*out)[]
)
{
    float max = z[0];
    float sum = 0.0f;

    for (size_t i = 1; i < size; i++)
    {
        if (z[i] > max)
        {
            max = z[i];
        }
    }

    for (size_t i = 0; i < size; i++)
    {
        (*out)[i] = expf(z[i] - max);
        sum += (*out)[i];
    }

    for (size_t i = 0; i < size; i++)
    {
        (*out)[i] /= sum;
    }

    return;
}

void initializeNet(
    OUT PHID1* hid1,
    OUT PHID2* hid2,
    OUT PHID3* hid3,
    OUT POUTL* out
)
{
    srand((unsigned int)time(NULL));

    (*hid1)->eachCellInputSize = 784;
    (*hid1)->cellsCount = 256;
    (*hid1)->activate = &lekyReLU;
    for (size_t j = 0; j < (*hid1)->cellsCount; j++)
    {
        for (size_t i = 0; i < (*hid1)->eachCellInputSize; i++)
        {
            (*hid1)->w[j][i] = (float)(rand() % 21 - 10) / 1000.0f;
        }
        (*hid1)->bias[j] = (float)(rand() % 21 - 10) / 1000.0f;
    }

    (*hid2)->eachCellInputSize = (*hid1)->cellsCount;
    (*hid2)->cellsCount = 512;
    (*hid2)->activate = &lekyReLU;
    for (size_t j = 0; j < (*hid2)->cellsCount; j++)
    {
        for (size_t i = 0; i < (*hid2)->eachCellInputSize; i++)
        {
            (*hid2)->w[j][i] = (float)(rand() % 21 - 10) / 1000.0f;
        }
        (*hid2)->bias[j] = (float)(rand() % 21 - 10) / 1000.0f;
    }

    (*hid3)->eachCellInputSize = (*hid2)->cellsCount;
    (*hid3)->cellsCount = 16;
    (*hid3)->activate = &lekyReLU;
    for (size_t j = 0; j < (*hid3)->cellsCount; j++)
    {
        for (size_t i = 0; i < (*hid3)->eachCellInputSize; i++)
        {
            (*hid3)->w[j][i] = (float)(rand() % 21 - 10) / 1000.0f;
        }
        (*hid3)->bias[j] = (float)(rand() % 21 - 10) / 1000.0f;
    }

    (*out)->eachCellInputSize = (*hid3)->cellsCount;
    (*out)->cellsCount = 10;
    (*out)->activate = &SoftMax;
    for (size_t j = 0; j < (*out)->cellsCount; j++)
    {
        for (size_t i = 0; i < (*out)->eachCellInputSize; i++)
        {
            (*out)->w[j][i] = (float)(rand() % 21 - 10) / 1000.0f;
        }
        (*out)->bias[j] = (float)(rand() % 21 - 10) / 1000.0f;
    }

    return;
}

void forward(
    IN OUT PHID1* hid1,
    IN OUT PHID2* hid2,
    IN OUT PHID3* hid3,
    IN OUT POUTL* out,
    IN float input[]
)
{
    for (size_t j = 0; j < (*hid1)->cellsCount; j++)
    {
        (*hid1)->z[j] = 0.0;
        for (size_t i = 0; i < (*hid1)->eachCellInputSize; i++)
        {
            (*hid1)->z[j] += input[i] * (*hid1)->w[j][i];
        }
        (*hid1)->z[j] += (*hid1)->bias[j];
    }

    (*hid1)->activate((*hid1)->z, (*hid1)->cellsCount, &(*hid1)->out);

    for (size_t j = 0; j < (*hid2)->cellsCount; j++)
    {
        (*hid2)->z[j] = 0.0;
        for (size_t i = 0; i < (*hid2)->eachCellInputSize; i++)
        {
            (*hid2)->z[j] += (*hid1)->out[i] * (*hid2)->w[j][i];
        }
        (*hid2)->z[j] += (*hid2)->bias[j];
    }

    (*hid2)->activate((*hid2)->z, (*hid2)->cellsCount, &(*hid2)->out);

    for (size_t j = 0; j < (*hid3)->cellsCount; j++)
    {
        (*hid3)->z[j] = 0.0;
        for (size_t i = 0; i < (*hid3)->eachCellInputSize; i++)
        {
            (*hid3)->z[j] += (*hid2)->out[i] * (*hid3)->w[j][i];
        }
        (*hid3)->z[j] += (*hid3)->bias[j];
    }

    (*hid3)->activate((*hid3)->z, (*hid3)->cellsCount, &(*hid3)->out);

    for (size_t j = 0; j < (*out)->cellsCount; j++)
    {
        (*out)->z[j] = 0.0;
        for (size_t i = 0; i < (*out)->eachCellInputSize; i++)
        {
            (*out)->z[j] += (*hid3)->out[i] * (*out)->w[j][i];
        }
        (*out)->z[j] += (*out)->bias[j];
    }

    (*out)->activate((*out)->z, (*out)->cellsCount, &(*out)->out);

    return;
}

void backward(
    IN OUT PHID1* hid1,
    IN OUT PHID2* hid2,
    IN OUT PHID3* hid3,
    IN OUT POUTL* out,
    IN SIZE_T epoch,
    IN float input[],
    IN UCHAR label
)
{
    int predictedLabel = -1;

    double loss = 0.0; 
    for (size_t j = 0; j < (*out)->cellsCount; j++)
    {
        (*out)->error[j] = (*out)->out[j] - ((SIZE_T)label == j ? 1.0f : 0.0f);
    }
    for (size_t j = 0; j < (*out)->cellsCount; j++)
    {
        loss += (*out)->error[j] * (*out)->error[j];
    }

    float maxOutput = -99999.0f;
    for (size_t j = 0; j < (*out)->cellsCount; j++)
    {
        if ((*out)->out[j] > maxOutput)
        {
            maxOutput = (*out)->out[j];
            predictedLabel = (int)j;
        }
    }

    if (predictedLabel == (int)label)
    {
        mark[epoch % VERIFY_BATCH] = 1;
    }
    else
    {
        mark[epoch % VERIFY_BATCH] = 0;
    }
    printf("Epoch: %zu\tLabel: %d\tPredicted: %d\tLoss: %.6f\tDiffer: %d\tAI Confidence: %f%%\n", epoch, (int)label, predictedLabel, loss, ((int)label - (int)predictedLabel) >= 0?(-((int)label - (int)predictedLabel)): (int)label - (int)predictedLabel, loss < 1.0f ? (1.0f - loss) * 100.0f : 0.0f);

    if (epoch % VERIFY_BATCH == 0)
    {
        int good = 0;
        for (size_t j = 0; j < VERIFY_BATCH; j++)
        {
            if (mark[j] == 1)
            {
                good++;
            }
        }
        if (epoch <= 50000)
        {
            printf("》》》》》》》》》》》》%zu次预测正确率：%lf%%\n", (size_t)VERIFY_BATCH, ((double)good / (double)VERIFY_BATCH) * 100);
        }
        else
        {
            printf("》》》》》》》》》》》》%zu次预测正确率：%lf%%\n", (size_t)VERIFY_BATCH, ((double)good / (double)VERIFY_BATCH) * 100);
            Sleep(50);
        }
    }

    for (size_t j = 0; j < (*hid3)->cellsCount; j++)
    {
        float temp = 0.0;
        for (size_t i = 0; i < (*out)->cellsCount; i++)
        {
            temp += (*out)->error[i] * (*out)->w[i][j];
        }
        (*hid3)->error[j] = ((*hid3)->z[j] >= 0.0f) ? 1.0f * temp : -0.001f * temp;
        //printf("(*hid3)->error[%zu]: %f\n", j, (*hid3)->error[j]);
    }

    for (size_t j = 0; j < (*hid2)->cellsCount; j++)
    {
        float temp = 0.0;
        for (size_t i = 0; i < (*hid3)->cellsCount; i++)
        {
            temp += (*hid3)->error[i] * (*hid3)->w[i][j];
        }
        (*hid2)->error[j] = ((*hid2)->z[j] >= 0.0f) ? 1.0f * temp : -0.001f * temp;
        //printf("(*hid2)->error[%zu]: %f\n", j, (*hid2)->error[j]);
    }

    for (size_t j = 0; j < (*hid1)->cellsCount; j++)
    {
        float temp = 0.0;
        for (size_t i = 0; i < (*hid2)->cellsCount; i++)
        {
            temp += (*hid2)->error[i] * (*hid2)->w[i][j];
        }
        (*hid1)->error[j] = ((*hid1)->z[j] >= 0.0f) ? 1.0f * temp : -0.001f * temp;
        //printf("(*hid1)->error[%zu]: %f\n", j, (*hid1)->error[j]);
    }

    for (size_t j = 0; j < (*out)->cellsCount; j++)
    {
        for (size_t i = 0; i < (*out)->eachCellInputSize; i++)
        {
            (*out)->w[j][i] -= LEARNING_RATE * (*out)->error[j] * (*hid3)->out[i];
        }
        (*out)->bias[j] -= LEARNING_RATE * (*out)->error[j] * 1.0f;
    }

    for (size_t j = 0; j < (*hid3)->cellsCount; j++)
    {
        for (size_t i = 0; i < (*hid3)->eachCellInputSize; i++)
        {
            (*hid3)->w[j][i] -= LEARNING_RATE * (*hid3)->error[j] * (*hid2)->out[i];
        }
        (*hid3)->bias[j] -= LEARNING_RATE * (*hid3)->error[j] * 1.0f;
    }

    for (size_t j = 0; j < (*hid2)->cellsCount; j++)
    {
        for (size_t i = 0; i < (*hid2)->eachCellInputSize; i++)
        {
            (*hid2)->w[j][i] -= LEARNING_RATE * (*hid2)->error[j] * (*hid1)->out[i];
        }
        (*hid2)->bias[j] -= LEARNING_RATE * (*hid2)->error[j] * 1.0f;
    }

    for (size_t j = 0; j < (*hid1)->cellsCount; j++)
    {
        for (size_t i = 0; i < (*hid1)->eachCellInputSize; i++)
        {
            (*hid1)->w[j][i] -= LEARNING_RATE * (*hid1)->error[j] * input[i];
        }
        (*hid1)->bias[j] -= LEARNING_RATE * (*hid1)->error[j] * 1.0f;
    }

    return;
}

void loadMnistLabels(
    IN CONST CHAR* trainSetPath,
    IN CONST CHAR* trainLabelPath,
    IN CONST INT trainNums,
    IN OUT float*** trainSetData,
    IN OUT UCHAR** trainLabelData
)
{
    FILE* imageFile = fopen(trainSetPath, "rb");
    FILE* labelFile = fopen(trainLabelPath, "rb");

    fseek(imageFile, 16, SEEK_SET); 
    fseek(labelFile, 8, SEEK_SET);  

    for (int i = 0; i < trainNums; ++i)
    {
        for (int j = 0; j < 784; ++j)
        {
            UCHAR pixel = 0;
            fread(&pixel, 1, 1, imageFile);
            (*trainSetData)[i][j] = (float)pixel / 255.0f; 
        }

        UCHAR label = 0;
        fread(&label, 1, 1, labelFile);
        (*trainLabelData)[i] = label;
    }

    fclose(imageFile);
    fclose(labelFile);

    return;
}

int main(void)
{
    CONST CHAR* trainSetPath = "E:\\desk\\MNIST_data\\train-images.idx3-ubyte";
    CONST CHAR* trainLabelPath = "E:\\desk\\MNIST_data\\train-labels.idx1-ubyte";
    CONST INT trainNums = 10000;

    float** trainSetData = NULL; allocateZeroMemory((PVOID*)&trainSetData, trainNums * sizeof(float*));
    for (size_t j = 0; j < trainNums; j++)
    {
        allocateZeroMemory((PVOID*)&trainSetData[j], 784 * sizeof(float));
    }
    UCHAR* trainLabelData = NULL; allocateZeroMemory((PVOID*)&trainLabelData, trainNums * sizeof(UCHAR));

    loadMnistLabels(trainSetPath, trainLabelPath, trainNums, &trainSetData, &trainLabelData);

    PHID1 hid1 = NULL; allocateZeroMemory((PVOID*)&hid1, sizeof(HID1));
    PHID2 hid2 = NULL; allocateZeroMemory((PVOID*)&hid2, sizeof(HID2));
    PHID3 hid3 = NULL; allocateZeroMemory((PVOID*)&hid3, sizeof(HID3));
    POUTL outl = NULL; allocateZeroMemory((PVOID*)&outl, sizeof(OUTL));    

    initializeNet(&hid1, &hid2, &hid3, &outl);

    for (size_t epoch = 0; epoch < trainNums * 20; epoch++)
    {
        forward(&hid1, &hid2, &hid3, &outl, trainSetData[epoch % trainNums]);
        backward(&hid1, &hid2, &hid3, &outl, epoch, trainSetData[epoch % trainNums], trainLabelData[epoch % trainNums]);
    }

    ExFreeMemory((PVOID*)&outl);
    ExFreeMemory((PVOID*)&hid3);
    ExFreeMemory((PVOID*)&hid2);
    ExFreeMemory((PVOID*)&hid1);

    ExFreeMemory((PVOID*)&trainLabelData);

    for (size_t j = 0; j < trainNums; j++)
    {
        ExFreeMemory((PVOID*)&trainSetData[j]);
    }
    ExFreeMemory((PVOID*)&trainSetData);

    return 0;
}
