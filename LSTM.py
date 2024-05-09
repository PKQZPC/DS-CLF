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
from sklearn.utils import shuffle
from torch.utils.data import random_split, TensorDataset
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score



BATCH_SIZE = 512  # batch size
EPOCH = 1000  # number of epoch
Num_class = 2
result_path = './K_3/out_put/LSTM'
Num_layers = 1
LAMDA = 0.2 # temperature 超参
LR = 0.001 # learning rate
BN_DIM = 99 # batch normalization dimension

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.backends.cudnn.enabled =False

def save_variable(file_name, variable):
    file_object = open(file_name, "wb")
    pickle.dump(variable, file_object)
    file_object.close()




def get_alter_loaders():

    map_file = "./K_3/DNA_map_K_3.txt"

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




class Classifier_CL(nn.Module):
    def __init__(self, num_layers, num_class=Num_class):
        super(Classifier_CL, self).__init__()
        input_size=1
        hidden_size=32
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.7)
        self.fc = nn.Linear(hidden_size, num_class)

    def forward(self, x):

        x = x.unsqueeze(1)
        x=x.float()

        x = x.permute(0, 2, 1)  
        
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))

        out = out[:, -1, :]

        # Dropout
        out = self.dropout(out)

        out = self.fc(out)

        # Softmax
        out = F.softmax(out, dim=1)
        return out









def train_val_model_CL(model, train_loader, val_loader, optimizer, loss_fun_sup, num_epochs=EPOCH):

    best_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        total_num = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            labels = torch.eye(2).to(device)[labels.unsqueeze(1).long()].squeeze().to(device)
            #print('iutputs',inputs)

            optimizer.zero_grad()
            outputs = model(inputs)

            outputs, labels=outputs.to('cpu'), labels.to('cpu')

            loss = loss_fun_sup(outputs, labels)


            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            total_num += inputs.size(0)

            


        epoch_loss = running_loss / total_num
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

        model.eval()
        correct_preds = 0
        total_preds = 0
        correct_positive = 0 
        total_positive = 0  
        correct_negative = 0  
        total_negative = 0 
        total_samples = 0  
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                #print('predicted',predicted)
                labels = labels.squeeze()
                total_preds += labels.size(0)
                correct_preds += (predicted == labels).sum().item()

                total_samples += labels.size(0)


                positive_mask = (labels == 1)
                total_positive += positive_mask.sum().item()
                correct_positive += (predicted[positive_mask] == labels[positive_mask]).sum().item()


                negative_mask = (labels == 0)
                total_negative += negative_mask.sum().item()
                correct_negative += (predicted[negative_mask] == labels[negative_mask]).sum().item()


        accuracy_positive = correct_positive / total_positive if total_positive != 0 else 0
        accuracy_negative = correct_negative / total_negative if total_negative != 0 else 0
        accuracy_avg = (accuracy_negative+ accuracy_positive) /2
        accuracy_total = (correct_positive + correct_negative) / total_samples if total_samples != 0 else 0


        print(f"eval_Positive Sample Accuracy: {accuracy_positive:.4f}")
        print(f"eval_Negative Sample Accuracy: {accuracy_negative:.4f}")
        print(f"eval_avg Sample Accuracy: {accuracy_avg:.4f}")
        print(f"eval_Total Accuracy: {accuracy_total:.4f}")


        is_best = accuracy_total > best_acc
        best_acc = max(accuracy_total, best_acc)
        if is_best:
            save_checkpoint({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'best_acc': best_acc,
            }, is_best, prefix=result_path + '/')

            f = open(os.path.join(result_path, "result.txt"), 'a')
            f.write("loaded best_checkpoint (epoch %d, best_acc %.4f)\n" % (epoch, best_acc))
            f.close()
            testQIM()





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
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            total_samples += labels.size(0)

            positive_mask = (labels == 1)
            total_positive += positive_mask.sum().item()
            correct_positive += (predicted[positive_mask] == labels[positive_mask]).sum().item()

            negative_mask = (labels == 0)
            total_negative += negative_mask.sum().item()
            correct_negative += (predicted[negative_mask] == labels[negative_mask]).sum().item()

            all_predictions.extend(predicted.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())


    accuracy_positive = correct_positive / total_positive if total_positive != 0 else 0
    accuracy_negative = correct_negative / total_negative if total_negative != 0 else 0
    accuracy_total = (correct_positive + correct_negative) / total_samples if total_samples != 0 else 0


    conf_matrix = confusion_matrix(all_labels, all_predictions)


    total_samples = len(all_labels)
    accuracy_total_1 = (conf_matrix[0, 0] + conf_matrix[1, 1]) / total_samples if total_samples != 0 else 0

    precision_total = precision_score(all_labels, all_predictions)


    recall_total = recall_score(all_labels, all_predictions)


    f1_total = f1_score(all_labels, all_predictions)
    

    print(f"Positive Sample Accuracy: {accuracy_positive:.4f}")
    print(f"Negative Sample Accuracy: {accuracy_negative:.4f}")
    print(f"Total Accuracy: {accuracy_total:.4f}")

    print(f"Total Accuracy_1: {accuracy_total_1:.4f}")
    print(f"Total Precision: {precision_total:.4f}")
    print(f"Total Recall: {recall_total:.4f}")
    print(f"Total F1 Score: {f1_total:.4f}")

    with open(os.path.join(result_path, "result.txt"), 'a') as f:
        f.write(f"Positive Sample Accuracy: {accuracy_positive:.4f}\n")
        f.write(f"Negative Sample Accuracy: {accuracy_negative:.4f}\n")
        f.write(f"Total Accuracy: {accuracy_total:.4f}\n")

        f.write(f"Total Accuracy_1: {accuracy_total_1:.4f}\n")
        f.write(f"Total Precision: {precision_total:.4f}\n")
        f.write(f"Total Recall: {recall_total:.4f}\n")
        f.write(f"Total F1 Score: {f1_total:.4f}\n")





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




def testQIM():
    test_model_with_best_checkpoint()





if __name__ == '__main__':

    #os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    print('\nEPOCH = ', EPOCH)
    print('Num_layers = ', Num_layers)
    #train_loader, val_loader  = get_alter_loaders_large()
    train_loader, val_loader  = get_alter_loaders()
    
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Classifier_CL(num_layers=Num_layers).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR)

    loss_fun_sup = nn.CrossEntropyLoss()

    train_val_model_CL(model, train_loader, val_loader, optimizer, loss_fun_sup, num_epochs=EPOCH)
