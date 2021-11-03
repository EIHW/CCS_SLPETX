import os
import sys
sys.path.append(os.path.join(sys.path[0], '../'))
sys.path.insert(1, '/home/yantianh/tianhao/lib/python3.7/site-packages')
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import recall_score as recall
from sklearn.metrics import confusion_matrix,classification_report
import seaborn as sns
# from torchvision import models
from torchsummaryX import summary


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def load_data():
    f = open('/home/yantianh/icassp/Compare2021_s300_40fu.pkl', 'rb')
    train_data, train_label, test_data, Test_label, valid_data, Valid_label, test_label, valid_label, pernums_test, pernums_dev = pickle.load(
        f)
    return train_data, train_label, test_data, Test_label, valid_data, Valid_label, test_label, valid_label, pernums_test, pernums_dev

def compute_uar(pred, gold):
    reca = recall(gold, pred, 'macro')

    return reca


def get_CI(data, bstrap):
    """

    :param data: [pred, groundtruth]
    :param bstrap:
    :return:
    """

    uars = []
    for _ in range(bstrap):
        idx = np.random.choice(range(len(data)), len(data), replace=True)
        samples = [data[i] for i in idx]
        sample_pred = [x[0] for x in samples]
        sample_groundtruth = [x[1] for x in samples]
        # sample_pred, sample_groundtruth = [data[i] for i in idx]
        uar = compute_uar(sample_pred, sample_groundtruth)
        uars.append(uar)

    lower_boundary_uar = pd.DataFrame(np.array(uars)).quantile(0.025)[0]
    higher_boundary_uar = pd.DataFrame(np.array(uars)).quantile(1 - 0.025)[0]

    return (higher_boundary_uar - lower_boundary_uar) / 2


class Attention(nn.Module):
    def __init__(self, units=128):
        super(Attention, self).__init__()
        self.units = units
        # self.fc1 = nn.Linear(640, 640)
        self.softmax = nn.Softmax(dim=1)
        self.fc2 = nn.Linear(256, self.units)

    def forward(self, inputs):
        hidden_states = inputs  # b_s, time_step,features      #shape=(?, ?, 128), dtype=float32)
        # hidden_states = torch.Tensor(hidden_states)
        # print(hidden_states.shape())
        # hidden_size = torch.Tensor(hidden_states.size(2))  # features    #128
        hidden_size = torch.einsum("ntf, ntf->f", [hidden_states, hidden_states])   # features    #128
        # score_first_part = F.relu(self.fc1(hidden_states))  # b_s, time_step, features
        # (batch_size, time_steps, hidden_size) dot (hidden_size, hidden_size) => (batch_size, time_steps, hidden_size)
        # W is the trainable weight matrix of attention Luong's multiplicative style score
        score_first_part = torch.einsum("ntf, f->ntf", [hidden_states, hidden_size])  # b_s, time_step, features   # shape=(?, 150, 128), dtype=float32)
        h_t = hidden_states[:, -1, :]  # b_s, features                                         #shape=(?, 128), dtype=float32)
        score = torch.einsum("nf, ntf->nt", [h_t, score_first_part])                       #shape=(?, 150), dtype=float32)
        attention_weights = self.softmax(score)  # b_s, t                            #shape=(?, 150), dtype=float32)
        context_vector = torch.einsum("ntf, nt->nf", [hidden_states, attention_weights])  # b_s, features  #shape=(?, 128), dtype=float32)
        pre_activation = torch.cat((context_vector, h_t), dim=1)  # b_s, 256            #shape=(?, 256), dtype=float32)
        attention_vector = F.tanh(self.fc2(pre_activation))  # b_s,128
        return attention_vector


class AttentionConvLSTM(nn.Module):
    def __init__(self,):
        super(AttentionConvLSTM,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(5, 5), stride=1, padding=2)
        # self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=1, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), stride=1, padding=2)
        # self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.conv5 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1, padding=1)
        # self.conv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1)

        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 4), stride=(2, 4))

        # self.bn1 = nn.BatchNorm2d(32)
        # self.bn2 = nn.BatchNorm2d(64)
        # self.bn3 = nn.BatchNorm2d(64)
        # LSTM block
        hidden_size = 128
        self.lstm = nn.LSTM(input_size=640, hidden_size=128, bidirectional= True, batch_first=True)
        # self.attention =
        self.fc1 = nn.Linear(640, 256)
        self.fc2 = nn.Linear(256,64)
        self.fc3 = nn.Linear(64, 2)
        self.drop_out = nn.Dropout(0.2)
        self.out_softmax = nn.Softmax(dim=1)
        self.attention_linear = nn.Linear(hidden_size*2,1)
        # self.out_linear = nn.Linear(hidden_size + 128, 2)
        # self.attention = Attention(units=128)

    def forward(self, x):      #(b_s, 3, 300, 40)
        x7 = F.relu(self.conv1(x))
        # x = self.conv1(x)

        x1 = self.maxpool1(x7)


        x2 = F.relu(self.conv3(x1))
        # x2 = self.bn2(x2)
        x2 = self.maxpool2(x2)

        x3 = F.relu(self.conv5(x7))
        # x3 = self.bn3(x3)
        x3 = self.maxpool3(x3)

        x4 = x2 + x3  # b_s, channel, t, n_mels    #(b_s, 64, 150, 10)
        x4 = x4.permute(0, 2, 3, 1)
        x4 = torch.flatten(x4, start_dim=2)    #(b_s, 150, 640)
        x4, (h,c) = self.lstm(x4)
        batch_size, T, _ = x4.shape
        attention_weights = [None] * T
        for t in range(T):
            embedding = x4[:, t, :]
            attention_weights[t] = self.attention_linear(embedding)
        attention_weights_norm = nn.functional.softmax(torch.stack(attention_weights, -1), -1)
        attention = torch.bmm(attention_weights_norm, x4)  #(B*1*T) * (B, T, hidden_size) = (B, 1, hidden_size)
        attention = torch.squeeze(attention, 1)

        # print("nanshou", x4.size())
        # x4 = self.attention(x4)
        x4 = self.drop_out(F.relu(self.fc2(attention)))
        x4 = F.relu(self.fc3(x4))
        # out = self.out_softmax(x4)
        return x4, x4


def loss_fnc(predictions, targets):
    return nn.CrossEntropyLoss()(input=predictions, target=targets)
    # return nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0]).to(device))(input=predictions, target=targets)



def make_train_step(model, loss_fnc, optimizer):
    def train_step(X, Y):
        # set model to train mode
        model.train()
        # forward pass
        output_logits, output_softmax = model(X)
        predictions = torch.argmax(output_softmax, dim=1)
        accuracy = torch.sum(Y == predictions) / float(len(Y))
        # compute loss
        loss = loss_fnc(output_logits, Y)
        # compute gradients
        loss.backward()
        # update parameters and zero gradients
        optimizer.step()
        optimizer.zero_grad()
        return loss.item(), accuracy * 100

    return train_step


def make_validate_fnc(model, loss_fnc):
    def validate(X, Y):
        with torch.no_grad():
            model.eval()
            output_logits, output_softmax = model(X)
            predictions = torch.argmax(output_softmax, dim=1)
            accuracy = torch.sum(Y == predictions) / float(len(Y))
            loss = loss_fnc(output_logits, Y)
        return loss.item(), accuracy * 100, predictions

    return validate


if __name__ == '__main__':

    train_data, train_label, test_data, Test_label, valid_data, Valid_label, test_label, valid_label, pernums_test, pernums_dev = load_data()

    train_data = np.transpose(train_data, (0, 3, 1, 2))
    test_data = np.transpose(test_data, (0, 3, 1, 2))
    valid_data = np.transpose(valid_data, (0, 3, 1, 2))

    index = np.arange(len(train_data))
    np.random.shuffle(index)

    train_data = train_data[index]
    train_label = train_label[index]

    # Valid_label = dense_to_one_hot(Valid_label, 2)
    # Test_label = dense_to_one_hot(Test_label, 2)
    # train_label = dense_to_one_hot(train_label, 2)

    # train_data = torch.Tensor(train_data)
    # train_label = torch.Tensor(train_label)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    model = AttentionConvLSTM().to(device)
    # model = AttentionConvLSTM()
    # print(model)
    # inputs = torch.zeros(1,3,150,40)
    # summary(model,inputs)
    print('Number of trainable params: ', sum(p.numel() for p in model.parameters()))
    OPTIMIZER = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-3, momentum=0.8)
    # OPTIMIZER = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

    train_step = make_train_step(model, loss_fnc, optimizer=OPTIMIZER)
    validate = make_validate_fnc(model, loss_fnc)
    losses = []
    val_losses = []
    EPOCHS = 50
    DATASET_SIZE = train_data.shape[0]
    BATCH_SIZE = 8
    best_acc = 0
    criterion = nn.CrossEntropyLoss()
    for epoch in range(EPOCHS):
        ind = np.random.permutation(DATASET_SIZE)
        train_data = train_data[ind, :, :, :]
        train_label = train_label[ind]
        epoch_acc = 0
        epoch_loss = 0
        iters = int(DATASET_SIZE / BATCH_SIZE)
        for i in range(iters):
            batch_start = i * BATCH_SIZE
            batch_end = min(batch_start + BATCH_SIZE, DATASET_SIZE)
            actual_batch_size = batch_end - batch_start
            X = train_data[batch_start:batch_end, :, :, :]
            Y = train_label[batch_start:batch_end]
            X_tensor = torch.tensor(X, device=device).float()
            Y_tensor = torch.tensor(Y, dtype=torch.long, device=device)
            loss, acc = train_step(X_tensor, Y_tensor)
            epoch_acc += acc * actual_batch_size / DATASET_SIZE
            epoch_loss += loss * actual_batch_size / DATASET_SIZE
            print(f"\r Epoch {epoch}: iteration {i}/{iters}", end='')
        X_val_tensor = torch.tensor(valid_data, device=device).float()
        Y_val_tensor = torch.tensor(Valid_label, dtype=torch.long, device=device)
        val_loss, val_acc, predictions = validate(X_val_tensor, Y_val_tensor)
        losses.append(epoch_loss)
        val_losses.append(val_loss)
        print(f"Epoch {epoch} --> loss:{epoch_loss:.4f},acc:{epoch_acc:.2f}%, val_loss:{val_loss:.4f}, val_acc:{val_acc:.2f}%")
    #
        if val_acc > best_acc:
            best_acc = val_acc
            SAVE_PATH = os.path.join(os.getcwd(), "models")
            torch.save(model.state_dict(), os.path.join(SAVE_PATH, 'covid19_crnn66_model' + str(epoch) + '.pt'))
            print('Model is saved to {}'.format(os.path.join(SAVE_PATH, 'covid19_crnn66_model' + str(epoch) + '.pt')))
#.....................................................................................
    # LOAD_PATH = os.path.join(os.getcwd(), 'models')
    # # model = AttentionConvLSTM()
    # model.load_state_dict(torch.load(os.path.join(LOAD_PATH, 'covid19_crnn_model19.pt')))
    # print('Model is loaded from {}'.format(os.path.join(LOAD_PATH, 'covid19_crnn_model19.pt')))
    #
    # X_test_tensor = torch.tensor(test_data, device=device).float()
    # Y_test_tensor = torch.tensor(Test_label, dtype=torch.long, device=device)
    # test_loss, test_acc, predictions = validate(X_test_tensor,Y_test_tensor)
    # print(f'Test loss is {test_loss:.3f}')
    # print(f'Test accuracy is {test_acc:.2f}%')
    # #
    # predictions = predictions.cpu().numpy()
    # test_acc_uw1 = recall(Test_label, predictions, average='macro')
    # print('test_acc_uw1', test_acc_uw1)
    #
    # index = 0
    # pre_results = []
    # true_results = []
    # for idx, per_i in enumerate(pernums_test):
    #     pred_ss = predictions[index:index + per_i]
    #     if float(sum(pred_ss)) / len(pred_ss) >= 0.5:
    #         pred_trunk = 1  # test_label
    #     else:
    #         pred_trunk = 0  # predict
    #     # sample_results.append([pred_trunk,test_label[idx]])
    #     pre_results.append(pred_trunk)
    #     true_results.append(test_label[idx])
    #     index = index + per_i
    # #
    # test_acc_uw1 = recall(true_results, pre_results, average='macro')
    # print('test_acc_uw1', test_acc_uw1)

    # predictions = predictions.cpu().numpy()
    # # predict = np.argmax(predictions, axis=1)
    # # y_gt = np.argmax(Test_label, axis=1)
    # # pd.crosstab(y_gt.reshape(-1),predict,rownames = ['label'], colnames = ['predit'])
    # print(classification_report(Test_label, predictions))
    # con_mat = confusion_matrix(Test_label.reshape(-1), predictions)
    # con_mat_norm = con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis]
    # con_mat_norm = np.around(con_mat_norm, decimals=2)

    # predict = predictions.tolist()
    # threshold = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # for thres in threshold:

    #
    # print(classification_report(true_results, pre_results))
    #
    # cm1 = confusion_matrix(true_results, pre_results)
    # print('Confusion Matrix : \n', cm1)
    #
    # total1 = sum(sum(cm1))
    # accuracy1 = (cm1[0, 0] + cm1[1, 1]) / total1
    # print('Accuracy : ', accuracy1)
    #
    # sensitivity1 = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
    # print('Sensitivity:', sensitivity1)
    #
    # specificity1 = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
    # print('Specificity : ', specificity1)
    #
    # data = [[p, g] for p, g in zip(pre_results, true_results)]
    # # a, b , c ,d = mean_confidence_interval(prediction, confidence=0.95)
    # cis = get_CI(data, 1000)
    # print('confidence_interval', cis)
    #
    # # === plot ===
    # plt.figure(figsize=(4, 4))
    # sns.heatmap(con_mat_norm, annot=True, cmap='Blues')
    #
    # plt.ylim(0, 4)
    # plt.xlabel('Predicted labels')
    # plt.ylabel('True labels')
    #
    # plt.show()
    #
    # loss_values = losses
    # val_loss_values = val_losses
    # epochs = range(1, len(loss_values) + 1)
    #
    # plt.plot(epochs, loss_values, 'bo', label='Training loss')
    # plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
    # plt.title('Training and validation loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend()
    #
    # plt.show()

