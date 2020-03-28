import torch
from att_blstm import AttBLSTM
import data_helpers
import pickle as pk
import numpy as np

trainFile = 'SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT'
testFile = 'SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT'
gloveFile = 'glove/glove.6B.50d.txt'
embedding_size = 50
label_size = 19
seq_length = 85
batch_size = 10
layer_num = 1
Epoch_size = 1000
LR = 1.0

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
    word_embedding = data_helpers.load_glove(gloveFile, embedding_size)
    train_text, y_train = data_helpers.load_data_and_labels(trainFile)
    model = AttBLSTM(embedding_size, label_size, batch_size).cuda()
    opt = torch.optim.Adadelta(model.parameters(), lr=LR, weight_decay=0.00001)

    prev_epoch = -1
    losses = []
    for data_batch in data_helpers.batch_iter((train_text, y_train), batch_size, Epoch_size):
        (epoch, data_batch) = data_batch
        if epoch != prev_epoch and epoch != 0:
            prev_epoch = epoch
            print('Epoch#' + str(epoch-1) + ':\t' + str(np.mean(losses)))
            losses = []
        x_batch = data_batch[:,0]
        x_batch = text2embed(x_batch)
        y_batch = data_batch[:,1:].astype(np.float32)
        y_batch = pad_y(y_batch)
        y_batch = torch.from_numpy(y_batch).cuda()
        opt.zero_grad()
        res = model.forward(x_batch)
        loss = model.loss(res, y_batch)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    print('Epoch#' + str(Epoch_size-1) + ':\t' + str(np.mean(losses)))
    model_file = open('Att-BLSTM.pk', 'wb')
    pk.dump(model, model_file)