import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset
import pickle
from itertools import cycle
import pickle
from sklearn.utils import shuffle
from torch.utils.data import random_split, TensorDataset
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score


BATCH_SIZE = 512  # batch size
EPOCH = 1000  # number of epoch
Num_class = 2
result_path = './K_3/out_put/CL_3'
Num_layers = 1
LAMDA = 0.02 # temperature
LR = 0.001 # learning rate
BN_DIM = 66 # batch normalization dimension

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = torch.device("cuda")

def save_variable(file_name, variable):
    file_object = open(file_name, "wb")
    pickle.dump(variable, file_object)
    file_object.close()




def get_alter_loaders():

    map_file = "./K_3/DNA_map_k_3.txt"

    File_Embed = "./K_3/data_K_3/train_val/steg.txt"
    File_NoEmbed = "./K_3/data_K_3/train_val/raw.txt"


    pklfile_train = './K_3/data_K_3/pklfiles/DNA_train.pkl'
    pklfile_val = './K_3/data_K_3/pklfiles/DNA_val.pkl'

    if not os.path.exists(pklfile_train):

        with open(map_file, 'r') as map_file:
            dna_map = {}
            for line in map_file:
                key, value = line.strip().split()
                dna_map[key] = value

        raw_filepath = File_NoEmbed
        steg_filepath = File_Embed

        with open(raw_filepath, 'r') as raw_file:
            raw_data = raw_file.readlines()
            raw_data_mapped = [' '.join([dna_map[base] for base in line.split()]) for line in raw_data]

        with open(steg_filepath, 'r') as steg_file:
            steg_data = steg_file.readlines()
            steg_data_mapped = [' '.join([dna_map[base] for base in line.split()]) for line in steg_data]

        data = raw_data_mapped + steg_data_mapped
        labels = [0] * len(raw_data) + [1] * len(steg_data)


        data_shuffled,labels_shuffled = shuffle(data,labels)
        print('data长度: ',len(data_shuffled))

        data_shuffled = torch.tensor([list(map(int, sample.split())) for sample in data_shuffled])
        labels_shuffled = torch.tensor(labels_shuffled)
        dataset = TensorDataset(data_shuffled, labels_shuffled)



        total_size = len(dataset)
        train_size = int(0.8 * total_size)
        val_size = total_size - train_size

        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        save_variable(pklfile_train, train_loader)
        save_variable(pklfile_val, val_loader)

        print("训练集长度:", len(train_loader))
        print("验证集长度:", len(val_loader))


    else:  
        train_loader = pickle.load(open(pklfile_train, 'rb'))  
        val_loader = pickle.load(open(pklfile_val, 'rb'))  

        print("训练集长度:", len(train_loader))
        print("验证集长度:", len(val_loader))


    return train_loader,val_loader




def convert_to_loader_CL(x_train, y_train, x_val, y_val, batch_size):

    x_train_tensor = torch.Tensor(x_train)
    y_train_tensor = torch.Tensor(y_train)
    x_val_tensor = torch.Tensor(x_val)
    y_val_tensor = torch.Tensor(y_val)


    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(x_val_tensor, y_val_tensor)

    steg_indices = [i for i, label in enumerate(y_train) if label[1] == 1]

    cover_indices = [i for i, label in enumerate(y_train) if label[1] == 0]


    train_steg_dataset = Subset(train_dataset, steg_indices)

    train_cover_dataset = Subset(train_dataset, cover_indices)

    train_steg_loader = DataLoader(train_steg_dataset, batch_size=batch_size, shuffle=True)
    train_cover_loader = DataLoader(train_cover_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


    return train_steg_loader, train_cover_loader, val_loader


class Model_CL(nn.Module):
    def __init__(self, num_layers):
        super(Model_CL, self).__init__()
        self.embedding = nn.Embedding(BATCH_SIZE, 128)
        self.position_embedding = PositionalEncoding(128)
        self.transformer_encoder_layers = nn.TransformerEncoderLayer(d_model=128, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layers, num_layers=num_layers)
        self.bn = nn.BatchNorm1d(num_features=BN_DIM)
        self.pooling = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):

        x = x.long()
        emb_x = self.embedding(x)
        emb_x += self.position_embedding(emb_x)


        emb_x = emb_x.permute(1, 0, 2)  
        outputs = self.transformer_encoder(emb_x)
        #print(outputs.size())
        outputs = self.bn(outputs.permute(1, 0, 2))
        outputs = self.pooling(outputs.permute(0, 2, 1)).squeeze(2)

        return outputs


class Classifier_CL(nn.Module):
    def __init__(self, num_layers, num_class=Num_class):
        super(Classifier_CL, self).__init__()
        self.model = Model_CL(num_layers)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(128, num_class)

    def forward(self, x):

        x_unsup = self.model(x)
        x_sup_1 = torch.zeros(x_unsup.size(0) // 3, x_unsup.size(1)).to(device)
        x_sup_2 = torch.zeros(x_unsup.size(0) // 3, x_unsup.size(1)).to(device)
        for i in range(x_sup_1.size(0)):
            x_sup_1[i] = x_unsup[3 * i]
            x_sup_2[i] = x_unsup[3 * i + 2]
        x_sup_1 = self.dropout(x_sup_1)
        x_sup_1 = self.fc(x_sup_1)
        x_sup_1 = F.softmax(x_sup_1, dim=1)
        x_sup_2 = self.dropout(x_sup_2)
        x_sup_2 = self.fc(x_sup_2)
        x_sup_2 = F.softmax(x_sup_2, dim=1)

        return x_unsup, x_sup_1, x_sup_2


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1536):
        super(PositionalEncoding, self).__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0),:x.size(1),:x.size(2)]
        return x


def compute_CL_loss(y_pred,lamda=LAMDA):
    row = torch.arange(0,y_pred.shape[0],3,device='cuda')
    col = torch.arange(y_pred.shape[0], device='cuda')
    col = torch.where(col % 3 != 0)[0].cuda()
    y_true = torch.arange(0,len(col),2,device='cuda')
    similarities = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=2)
    

    similarities = torch.index_select(similarities,0,row)
    similarities = torch.index_select(similarities,1,col)
    

    similarities = similarities / lamda
    

    loss = F.cross_entropy(similarities,y_true)
    return torch.mean(loss)


def train_val_model_CL(model, train_steg_loader, train_cover_loader, val_loader, optimizer, loss_fun_sup, num_epochs=EPOCH):

    best_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        total_num = 0
        for (inputs_1, labels_1), (inputs_0, labels_0) in zip(train_steg_loader, cycle(train_cover_loader)):
            inputs_1, labels_1, inputs_0, labels_0 = inputs_1.to(device), labels_1.to(device), inputs_0.to(device), labels_0.to(device)


            labels_1 = torch.eye(2).to(device)[labels_1.unsqueeze(1).long()].squeeze().to(device)
            #labels_0 = labels_0[:, 1]
            labels_0 = torch.eye(2).to(device)[labels_0.unsqueeze(1).long()].squeeze().to(device)

            inputs_size = min(inputs_1.size(0), inputs_0.size(0))
            inputs_1 = inputs_1[:inputs_size]
            inputs_0 = inputs_0[:inputs_size]
            labels_1 = labels_1[:inputs_size]
            labels_0 = labels_0[:inputs_size]

            #print(inputs_1.size())
            input_final = torch.zeros(inputs_size*3, inputs_1.size(1)).to(device)
            for i in range(inputs_size):
                input_final[3*i] = inputs_0[i % inputs_size]
                input_final[3*i+1] = inputs_0[(i+1)  % inputs_size]
                input_final[3*i+2] = inputs_1[i  % inputs_size]

            optimizer.zero_grad()
            outputs_unsup, outputs_sup_1, outputs_sup_2 = model(input_final)
            loss_sup_1 = loss_fun_sup(outputs_sup_1, labels_0)
            loss_sup_2 = loss_fun_sup(outputs_sup_2, labels_1)
            loss_sup = (loss_sup_1 + loss_sup_2) / 2
            loss_unsup = compute_CL_loss(outputs_unsup)
            #loss_unsup = 0
            loss = loss_sup + loss_unsup*0.001
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs_size * 2
            total_num += inputs_size * 2

        epoch_loss = running_loss / total_num
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

        model.eval()
        correct_preds = 0
        num_labels_1 = 0
        num_labels_0 = 0
        num_correct_1_pred_1 = 0
        num_correct_0_pred_1 = 0
        num_correct_1_pred_2 = 0
        num_correct_0_pred_2 = 0
        total_preds = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                input_final = torch.zeros(inputs.size(0) * 3, inputs.size(1)).to(device)
                for i in range(inputs.size(0)):
                    input_final[3 * i] = inputs[i]
                    input_final[3 * i + 1] = inputs[i]
                    input_final[3 * i + 2] = inputs[i]
                _, outputs_sup_1, outputs_sup_2 = model(input_final)
                _, predicted_1 = torch.max(outputs_sup_1, 1)
                _, predicted_2 = torch.max(outputs_sup_2, 1)
                labels = labels.squeeze()
                total_preds += labels.size(0) * 2
                correct_preds += (predicted_1 == labels).sum().item()
                correct_preds += (predicted_2 == labels).sum().item()

                num_labels_1 += torch.sum(labels == 1).item()
                num_labels_0 += torch.sum(labels == 0).item()
                num_correct_1_pred_1 += torch.sum((predicted_1 == labels) & (labels == 1)).item()
                num_correct_0_pred_1 += torch.sum((predicted_1 == labels) & (labels == 0)).item()
                num_correct_1_pred_2 += torch.sum((predicted_2 == labels) & (labels == 1)).item()
                num_correct_0_pred_2 += torch.sum((predicted_2 == labels) & (labels == 0)).item()

        accuracy_1_num = num_correct_1_pred_1 + num_correct_1_pred_2
        accuracy_0_num = num_correct_0_pred_1 + num_correct_0_pred_2
        accuracy_1 = accuracy_1_num / (num_labels_1 * 2)
        accuracy_0 = accuracy_0_num / (num_labels_0 * 2)
        accuracy_bi = accuracy_1/2 + accuracy_0/2
        accuracy = correct_preds / total_preds
        print(f"Validation Accuracy: {accuracy:.4f}")
        print(f"Validation Accuracy_Bi: {accuracy_bi:.4f}")
        print(f"Validation 1 Accuracy: {accuracy_1:.4f}")
        print(f"Validation 0 Accuracy: {accuracy_0:.4f}")


        is_best = accuracy > best_acc
        best_acc = max(accuracy, best_acc)
        if is_best:
            save_checkpoint({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'best_acc': best_acc,
            }, is_best, prefix=result_path + '/')


            f = open(os.path.join(result_path, "result.txt"), 'a')
            f.write("loaded best_checkpoint (epoch %d, best_acc %.4f, acc_bi %.4f)\n" % (epoch, best_acc, accuracy_bi))
            f.close()
            testDNA()





def save_checkpoint(state, is_best, prefix):

    if is_best:
        directory = os.path.dirname(prefix)
        if not os.path.exists(directory):
            os.makedirs(directory)


        torch.save(state, prefix + 'model_best.pth.tar')
        print('save beat check :' + prefix + 'model_best.pth.tar')




def parse_sample_test(file_path):

    file = open(file_path, 'r')
    lines = file.readlines()
    sample = []
    for line in lines:

        line = [int(l) for l in line.split()]
        sample.append(line)
    return sample





def test_model_with_best_checkpoint():

    model = Classifier_CL(num_layers=Num_layers)
    model = model.to(device)


    best_checkpoint = torch.load(os.path.join(result_path, 'model_best.pth.tar'))
    print('load bestcheck from :', os.path.join(result_path, 'model_best.pth.tar'))
    model.load_state_dict(best_checkpoint['model'])


    test_loader = get_alter_loaders_test()


    model.eval()
    correct_preds = 0
    num_labels_1 = 0
    num_labels_0 = 0
    num_correct_1_pred_1 = 0
    num_correct_0_pred_1 = 0
    num_correct_1_pred_2 = 0
    num_correct_0_pred_2 = 0
    total_preds = 0
    with torch.no_grad():
        correct_positive = 0  
        total_positive = 0  
        correct_negative = 0  
        total_negative = 0  
        total_samples = 0  
        all_predictions = []
        all_labels = []

        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            #print(labels.size())
            input_final = torch.zeros(inputs.size(0) * 3, inputs.size(1)).to(device)
            for i in range(inputs.size(0)):
                input_final[3 * i] = inputs[i]
                input_final[3 * i + 1] = inputs[i]
                input_final[3 * i + 2] = inputs[i]
            _, outputs_sup_1, outputs_sup_2 = model(input_final)
            _, predicted_1 = torch.max(outputs_sup_1, 1)
            _, predicted_2 = torch.max(outputs_sup_2, 1)

            #_, labels = torch.max(labels, 1)
            input_final = torch.zeros(inputs.size(0) * 3, inputs.size(1)).to(device)

            total_preds += labels.size(0) * 2
            correct_preds += (predicted_1 == labels).sum().item()
            correct_preds += (predicted_2 == labels).sum().item()
            
            num_labels_1 += torch.sum(labels == 1).item()
            num_labels_0 += torch.sum(labels == 0).item()
            
            num_correct_1_pred_1 += torch.sum((predicted_1 == labels) & (labels == 1)).item()
            num_false_1_pred_1 = torch.sum((predicted_1 != 1) & (labels == 1)).item()

            num_correct_0_pred_1 += torch.sum((predicted_1 == labels) & (labels == 0)).item()
            num_false_0_pred_1 = torch.sum((predicted_1 != 0) & (labels == 0)).item()

            num_correct_1_pred_2 += torch.sum((predicted_2 == labels) & (labels == 1)).item()
            num_false_1_pred_2 = torch.sum((predicted_2 != 1) & (labels == 1)).item()


            num_correct_0_pred_2 += torch.sum((predicted_2 == labels) & (labels == 0)).item()
            num_false_0_pred_2 = torch.sum((predicted_2 != 0) & (labels == 0)).item()
            

            all_predictions.extend(predicted_1.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())


    accuracy_1_num = num_correct_1_pred_1 + num_correct_1_pred_2
    accuracy_0_num = num_correct_0_pred_1 + num_correct_0_pred_2
    accuracy_1 = accuracy_1_num / (num_labels_1 * 2)
    accuracy_0 = accuracy_0_num / (num_labels_0 * 2)
    accuracy_bi = accuracy_1 / 2 + accuracy_0 / 2
    accuracy = correct_preds / total_preds


    conf_matrix = confusion_matrix(all_labels, all_predictions)

    total_samples = len(all_labels)
    total_accuracy = (conf_matrix[0, 0] + conf_matrix[1, 1]) / total_samples if total_samples != 0 else 0


    total_precision = precision_score(all_labels, all_predictions)


    total_recall = recall_score(all_labels, all_predictions)


    total_f1 = f1_score(all_labels, all_predictions)

    print(f"Positive Sample Accuracy: {accuracy_1:.4f}")
    print(f"Negative Sample Accuracy: {accuracy_0:.4f}")
    print(f"Total Accuracy: {total_accuracy:.4f}")

    print(f"Total Precision: {total_precision:.4f}")
    print(f"Total Recall: {total_recall:.4f}")
    print(f"Total F1 Score: {total_f1:.4f}")

    f = open(os.path.join(result_path, "result.txt"), 'a')
    f.write(f"Positive Sample Accuracy: {accuracy_1:.4f}\n")
    f.write(f"Negative Sample Accuracy: {accuracy_0:.4f}\n")
    f.write("Total Accuracy: %.4f\n" % total_accuracy)
    f.write("Total Precision: %.4f\n" % total_precision)
    f.write("Total Recall: %.4f\n" % total_recall)
    f.write("Total F1 Score: %.4f\n" % total_f1)
    f.close()






def get_alter_loaders_test():
    map_file = "./K_3/DNA_map_K_3.txt"

    File_Embed = "./K_3/data_K_3/test/steg.txt"
    File_NoEmbed = "./K_3/data_K_3/test/raw.txt"


    pklfile_test = './K_3/data_K_3/pklfiles/DNA_test.pkl'

    if not os.path.exists(pklfile_test):

        with open(map_file, 'r') as map_file:
            dna_map = {}
            for line in map_file:
                key, value = line.strip().split()
                dna_map[key] = value

        raw_filepath = File_NoEmbed
        steg_filepath = File_Embed

        with open(raw_filepath, 'r') as raw_file:
            raw_data = raw_file.readlines()
            raw_data_mapped = [' '.join([dna_map[base] for base in line.split()]) for line in raw_data]

        with open(steg_filepath, 'r') as steg_file:
            steg_data = steg_file.readlines()
            steg_data_mapped = [' '.join([dna_map[base] for base in line.split()]) for line in steg_data]


        data = raw_data_mapped + steg_data_mapped
        labels = [0] * len(raw_data) + [1] * len(steg_data)


        data_shuffled,labels_shuffled = shuffle(data,labels)
        print('data长度: ',len(data_shuffled))
        
        data_shuffled = torch.tensor([list(map(int, sample.split())) for sample in data_shuffled])
        labels_shuffled = torch.tensor(labels_shuffled)

        dataset = TensorDataset(data_shuffled, labels_shuffled)




        test_loader = DataLoader( dataset, batch_size=BATCH_SIZE, shuffle=False)


        save_variable(pklfile_test, test_loader)

        print("测试集长度:", len(test_loader))

    else: 
        test_loader = pickle.load(open(pklfile_test, 'rb'))  

        print("测试集长度:", len(test_loader))

    return test_loader






def testDNA():
    test_model_with_best_checkpoint()




if __name__ == '__main__':

    #os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    print('\nEPOCH = ', EPOCH)
    print('Num_layers = ', Num_layers)
    #train_loader, val_loader  = get_alter_loaders_large()
    train_loader, val_loader  = get_alter_loaders()
    
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Classifier_CL(num_layers=Num_layers).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR)

    loss_fun_sup = nn.CrossEntropyLoss()
    train_steg_loader = []
    train_cover_loader = []


    for inputs, labels in train_loader:

        for i in range(len(labels)):
            if labels[i] == 1:
                train_steg_loader.append((inputs[i], labels[i]))
            else:
                train_cover_loader.append((inputs[i], labels[i]))

    train_1_loader = torch.utils.data.DataLoader(train_steg_loader, batch_size=train_loader.batch_size, shuffle=True)
    train_0_loader = torch.utils.data.DataLoader(train_cover_loader, batch_size=train_loader.batch_size, shuffle=True)

    train_val_model_CL(model, train_1_loader, train_0_loader, val_loader, optimizer, loss_fun_sup, num_epochs=EPOCH)
