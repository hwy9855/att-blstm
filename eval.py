import torch
from att_blstm import AttBLSTM
import data_helpers
import numpy as np
import pickle as pk

trainFile = 'SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT'
testFile = 'SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT'
gloveFile = 'glove/glove.6B.50d.txt'
embedding_size = 50
seq_length = 85
batch_size = 10
label_size = 19

def poss2label(poss):
    labels = []
    for i in range(batch_size):
        label = 0
        tmp = 0
        for j in range(19):
            if poss[i][j] > tmp:
                label = j
                tmp = poss[i][j]
        labels.append(label)
    return labels

def count(predict, y):

    labels = []
    for i in range(len(predict)):
        tmp = -1
        for j in range(19):
            if y[i][j] == 1:
                tmp = j
                break
        labels.append(tmp)

    for i in range(len(predict)):
        if labels[i] != -1:
            if labels[i] == predict[i]:
                TP[predict[i]] += 1
            else:
                FP[predict[i]] += 1
                FN[labels[i]] += 1

def text2embed(corpus):
    x_train = torch.zeros(batch_size, seq_length, embedding_size)
    for i, sentense in enumerate(corpus):
        words = sentense.split(' ')
        for j, word in enumerate(words):
            if word in word_embedding:
                x_train[i][j] = torch.from_numpy(word_embedding[word])
            else:
                x_train[i][j] = torch.randn(embedding_size)
    return x_train.cuda()

def pad_y(y):
    if y.shape[0] != batch_size:
        tmp = np.zeros([batch_size - y.shape[0], label_size]).astype(np.float32)
        y = np.concatenate((y, tmp), axis=0)
    return y


if __name__ == '__main__':
    TP = np.zeros(19)
    FP = np.zeros(19)
    FN = np.zeros(19)

    word_embedding = data_helpers.load_glove(gloveFile, embedding_size)
    test_text, y_test = data_helpers.load_data_and_labels(testFile)
    model_file = open('Att-BLSTM.pk', 'rb')
    model = pk.load(model_file)
    model.eval()
    for data_batch in data_helpers.batch_iter((test_text, y_test), batch_size, 1):
        (_, data_batch) = data_batch
        x_batch = data_batch[:,0]
        x_batch = text2embed(x_batch)
        y_batch = data_batch[:,1:].astype(np.float32)
        y_batch = pad_y(y_batch)
        y_batch = torch.from_numpy(y_batch).view(batch_size, -1)
        res = model.forward(x_batch)
        predict = poss2label(res)
        count(predict, y_batch)

    precision = np.zeros(19)
    recall = np.zeros(19)
    F1 = np.zeros(19)
    print(TP, FP, FN)
    for i in range(19):
        precision[i] = TP[i] / (TP[i] + FP[i] + 0.0001)
        recall[i] = TP[i] / (TP[i] + FN[i] + 0.0001)
        F1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i] + 0.0001)
    print(F1, F1.mean())
    print(TP.sum() / len(test_text))
