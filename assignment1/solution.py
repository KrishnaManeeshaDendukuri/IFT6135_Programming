# This script contains the helper functions you will be using for this assignment

import os
import random

import numpy as np
import torch
import h5py
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class BassetDataset(Dataset):
    """
    BassetDataset class taken with permission from Dr. Ahmad Pesaranghader

    We have already processed the data in HDF5 format: er.h5
    See https://www.h5py.org/ for details of the Python package used

    We used the same data processing pipeline as the paper.
    You can find the code here: https://github.com/davek44/Basset
    """

    # Initializes the BassetDataset
    def __init__(self, path='./data/', f5name='er.h5', split='train', transform=None):
        """
        Args:
            :param path: path to HDF5 file
            :param f5name: HDF5 file name
            :param split: split that we are interested to work with
            :param transform (callable, optional): Optional transform to be applied on a sample
        """

        self.split = split

        split_dict = {'train': ['train_in', 'train_out'],
                      'test': ['test_in', 'test_out'],
                      'valid': ['valid_in', 'valid_out']}

        assert self.split in split_dict, "'split' argument can be only defined as 'train', 'valid' or 'test'"

        # Open hdf5 file where one-hoted data are stored
        self.dataset = h5py.File(os.path.join(path, f5name.format(self.split)), 'r')

        # Keeping track of the names of the target labels
        self.target_labels = self.dataset['target_labels']

        # Get the list of volumes
        self.inputs = self.dataset[split_dict[split][0]]
        self.outputs = self.dataset[split_dict[split][1]]

        self.ids = list(range(len(self.inputs)))
        if self.split == 'test':
            self.id_vars = np.char.decode(self.dataset['test_headers'])

    def __getitem__(self, i):
        """
        Returns the sequence and the target at index i

        Notes:
        * The data is stored as float16, however, your model will expect float32.
          Do the type conversion here!
        * Pay attention to the output shape of the data.
          Change it to match what the model is expecting
          hint: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        * The target must also be converted to float32
        """

        idx = self.ids[i]
        # WRITE CODE HERE

        seq = self.inputs[idx]
        target = self.outputs[idx]

        seq = np.transpose(seq, (1, 2, 0)).astype(np.float32)
        target = np.float32(target)

        seq = torch.from_numpy(seq)
        target = torch.from_numpy(target)
        # Sequence & Target
        output = {'sequence': seq, 
                  'target': target}
        return output

    def __len__(self):
        # WRITE CODE HERE
        return len(self.inputs)

    def get_seq_len(self):
        """
        Answer to Q1 part 2
        """
        # WRITE CODE HERE
        return self.inputs[0].shape[-1]

    def is_equivalent(self):
        """
        Answer to Q1 part 3
        """
        # WRITE CODE HERE
        return True


class Basset(nn.Module):
    """
    Basset model
    Architecture specifications can be found in the supplementary material
    You will also need to use some Convolution Arithmetic
    """

    def __init__(self):
        super(Basset, self).__init__()

        self.num_cell_types = 164

        self.conv1 = nn.Conv2d(1, 300, (19, 4), stride=(1, 1), padding=(9, 0))
        self.conv2 = nn.Conv2d(300, 200, (11, 1), stride=(1, 1), padding=(5, 0))
        self.conv3 = nn.Conv2d(200 , 200, (7, 1), stride=(1, 1), padding=(4, 0))

        self.bn1 = nn.BatchNorm2d(300)
        self.bn2 = nn.BatchNorm2d(200)
        self.bn3 = nn.BatchNorm2d(200)
        self.maxpool1 = nn.MaxPool2d((3, 1))
        self.maxpool2 = nn.MaxPool2d((4, 1))
        self.maxpool3 = nn.MaxPool2d((4, 1))

        self.fc1 = nn.Linear(13*200, 1000)
        self.bn4 = nn.BatchNorm1d(1000)

        self.fc2 = nn.Linear(1000, 1000)
        self.bn5 = nn.BatchNorm1d(1000)

        self.fc3 = nn.Linear(1000, self.num_cell_types)
        self.dropout = 0.3
        self.drop = nn.Dropout(self.dropout)

    def forward(self, x):

        # WRITE CODE HERE
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.maxpool3(x)

        x = x.view(-1, 13*200)

        x = self.fc1(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.drop(x)

        x = self.fc2(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.drop(x)

        x = self.fc3(x)

        return x


def compute_fpr_tpr(y_true, y_pred):
    """
    Computes the False Positive Rate and True Positive Rate
    Args:
        :param y_true: groundtruth labels (np.array of ints)
        :param y_pred: model decisions (np.array of ints)

    :Return: dict with tpr, fpr (values are floats)
    """
    output = {'fpr': 0., 'tpr': 0.}

    # WRITE CODE HERE
    
    fp = np.sum((y_pred == 1) & (y_true == 0))
    tp = np.sum((y_pred == 1) & (y_true == 1))

    fn = np.sum((y_pred == 0) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))

    fpr = fp / (fp + tn)
    tpr = tp / (tp + fn)

    output = {'fpr': fpr, 'tpr': tpr}
    return output


def compute_fpr_tpr_dumb_model():
    """
    Simulates a dumb model and computes the False Positive Rate and True Positive Rate

    :Return: dict with tpr_list, fpr_list.
             These lists contain the tpr and fpr for different thresholds
             fpr and tpr values in the lists should be floats
             Order the lists such that:
                 output['fpr_list'][0] corresponds to k=0.
                 output['fpr_list'][1] corresponds to k=0.05 
                 ...
            Do the same for output['tpr_list']
             
    """
    output = {'fpr_list': [], 'tpr_list': []}

    # WRITE CODE HERE
    target = np.random.randint(2, size= 1000)
    preds = np.random.uniform(low = 0,high = 1,size = 1000)

    thresholds = np.arange(0,1,0.05)
    for k in thresholds:
        y_pred = np.where(preds >= k, 1, 0)

        fp = np.sum((y_pred == 1) & (target == 0))
        tp = np.sum((y_pred == 1) & (target == 1))
        fn = np.sum((y_pred == 0) & (target == 1))
        tn = np.sum((y_pred == 0) & (target == 0))

        output["fpr_list"].append(fp / (fp + tn))
        output["tpr_list"].append(tp / (tp + fn))

    return output


def compute_fpr_tpr_smart_model():
    """
    Simulates a smart model and computes the False Positive Rate and True Positive Rate

    :Return: dict with tpr_list, fpr_list.
             These lists contain the tpr and fpr for different thresholds
             fpr and tpr values in the lists should be floats
             Order the lists such that:
                 output['fpr_list'][0] corresponds to k=0.
                 output['fpr_list'][1] corresponds to k=0.05 
                 ...
            Do the same for output['tpr_list']
    """
    output = {'fpr_list': [], 'tpr_list': []}

    # WRITE CODE HERE

    #simulate target
    target = np.random.randint(2, size= 1000)

    #simulate preds
    inds_true = np.where(target==1)[0] 
    inds_false = np.where(target==0)[0]

    preds_true = np.random.uniform(low = 0.4,high = 1, size=len(inds_true))
    preds_false = np.random.uniform(low = 0,high = 0.6, size=len(inds_false))

    preds = np.empty(shape = (1000,))
    np.put(preds, inds_true, preds_true)
    np.put(preds, inds_false, preds_false)    
    
    #setting thresholds
    thresholds = np.arange(0,1,0.05)
    for k in thresholds:
        y_pred = np.where(preds >= k, 1, 0)

        fp = np.sum((y_pred == 1) & (target == 0))
        tp = np.sum((y_pred == 1) & (target == 1))
        fn = np.sum((y_pred == 0) & (target == 1))
        tn = np.sum((y_pred == 0) & (target == 0))

        output["fpr_list"].append(fp / (fp + tn))
        output["tpr_list"].append(tp / (tp + fn))

    return output


def compute_auc_both_models():
    """
    Simulates a dumb model and a smart model and computes the AUC of both
    :Return: dict with keys 'auc_dumb_model', 'auc_smart_model'.
             These contain the AUC for both models
             auc values in the lists should be floats
    """

    output = {'auc_dumb_model': 0., 'auc_smart_model': 0.}

    # WRITE CODE HERE

    ####### dumb model

    target_dumb = np.random.randint(2, size= 1000)
    preds_dumb = np.random.uniform(low = 0,high = 1,size = 1000)

    ####### smart model

     #simulate target
    target_smart = np.random.randint(2, size= 1000)

    #simulate preds
    inds_true = np.where(target_smart==1)[0] 
    inds_false = np.where(target_smart==0)[0]

    preds_true = np.random.uniform(low = 0.4,high = 1, size=len(inds_true))
    preds_false = np.random.uniform(low = 0,high = 0.6, size=len(inds_false))

    preds_smart = np.empty(shape = (1000,))
    np.put(preds_smart, inds_true, preds_true)
    np.put(preds_smart, inds_false, preds_false)  

    output["auc_dumb_model"] = compute_auc(target_dumb, preds_dumb)["auc"]
    output["auc_smart_model"] = compute_auc(target_smart, preds_smart)["auc"]
    print(output)

    return output


def compute_auc_untrained_model(model, dataloader, device):
    """
    Computes the AUC of your input model
    Args:
        :param model: solution.Basset()
        :param dataloader: torch.utils.data.DataLoader
                           Where the dataset is solution.BassetDataset
        :param device: torch.device
    :Return: dict with key 'auc'.
             This contains the AUC for the model
             auc value should be float
    Notes:
    * Dont forget to re-apply your output activation!
    * Make sure this function works with arbitrarily small dataset sizes!
    * You should collect all the targets and model outputs and then compute AUC at the end
      (compute time should not be as much of a consideration here)
    """
    output = {'auc': 0.}

    # WRITE CODE HERE
    # model = Basset()
    y_pred = torch.tensor([], device=device)
    y_true = torch.tensor([], device=device)
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
                batch_x = batch["sequence"]
                batch_y = batch["target"]

                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                preds = model(batch_x)
                # print(pred.shape)

                y_true = torch.cat((y_true, batch_y), 0)
                y_pred = torch.cat((y_pred, preds), 0)

    y_true = y_true.cpu().numpy()  
    # _, y_pred = torch.max(y_pred, 1)
    # y_pred = y_pred.cpu().numpy()
    y_pred_prob = torch.sigmoid(y_pred).cpu().numpy()

    output["auc"] = compute_auc(y_true, y_pred_prob)["auc"]
    return output


def compute_auc(y_true, y_model):
    """
    Computes area under the ROC curve (using method described in main.ipynb)
    Args:
        :param y_true: groundtruth labels (np.array of ints [0 or 1])
        :param y_model: model outputs (np.array of float32 in [0, 1])
    :Return: dict with key 'auc'.
             This contains the AUC for the model
             auc value should be float
    Note: if you set y_model as the output of solution.Basset, 
    you need to transform it before passing it here!
    """
    output = {'auc': 0.}
    #reference: https://medium.com/building-ibotta/understanding-roc-auc-part-2-2-a1e418a3afdb
    # Total number of observations
    # N = y_true.shape[0]
    
    # # Index vector
    # I = np.arange(1, N + 1)
    
    # # Number of positive observations
    # N_pos = np.sum(y_true)
    
    # # Number of negative observations
    # N_neg = N - N_pos
    
    # # Sort true labels according to scores
    # I = y_model.argsort()[::-1][:N]
    # y_pred = y_true[I]
    
    # # Index vector
    # I = np.arange(1, N + 1)
    
    # output["auc"] = 1. + ((N_pos + 1.) / (2 * N_neg)) - (1. / (N_pos * N_neg)) * I.dot(y_pred)

    fpr_tpr = {'fpr_list': np.array([]), 'tpr_list': np.array([])}
    for thresh in np.arange(0, 1, 0.05):
        y_pred_th = y_model > thresh
        y_pred_th = y_pred_th.astype(int)
        out = compute_fpr_tpr(y_true, y_pred_th)
        fpr_tpr['fpr_list'] = np.append(fpr_tpr['fpr_list'], out['fpr'])
        fpr_tpr['tpr_list'] = np.append(fpr_tpr['tpr_list'], out['tpr'])
    print(fpr_tpr['fpr_list'])
    print(fpr_tpr['tpr_list'])
    dx = np.diff(fpr_tpr['fpr_list'])

    left_riemann_sum = abs(np.sum(fpr_tpr['tpr_list'][:-1] * dx))
    print("Left Riemann Sum:",left_riemann_sum)

    right_riemann_sum = abs(np.sum(fpr_tpr['tpr_list'][1:] * dx))
    print("Right Riemann Sum:",right_riemann_sum)

    output["auc"] = (left_riemann_sum + right_riemann_sum)/2

    return output


def get_critereon():
    """
    Picks the appropriate loss function for our task
    criterion should be subclass of torch.nn
    """

    # WRITE CODE HERE
    criterion = nn.BCEWithLogitsLoss()
    return criterion

def compute_auc_trained_model(y_model, y_true):
    # print("compute auc")
    # print(type(y_model))
    # print(type(y_true))
    y_model = y_model.detach().numpy()
    y_true = y_true.detach().numpy()

    fpr_tpr = {'fpr_list': np.array([]), 'tpr_list': np.array([])}
    for thresh in np.arange(0, 1, 0.05):
        y_pred_th = y_model > thresh
        y_pred_th = y_pred_th.astype(int)
        out = compute_fpr_tpr(y_true, y_pred_th)
        fpr_tpr['fpr_list'] = np.append(fpr_tpr['fpr_list'], out['fpr'])
        fpr_tpr['tpr_list'] = np.append(fpr_tpr['tpr_list'], out['tpr'])
    # print(fpr_tpr['fpr_list'])
    # print(fpr_tpr['tpr_list'])
    dx = np.diff(fpr_tpr['fpr_list'])

    left_riemann_sum = abs(np.sum(fpr_tpr['tpr_list'][:-1] * dx))
    # print("Left Riemann Sum:",left_riemann_sum)

    right_riemann_sum = abs(np.sum(fpr_tpr['tpr_list'][1:] * dx))
    # print("Right Riemann Sum:",right_riemann_sum)
    auc = (left_riemann_sum + right_riemann_sum)/2
    return auc

def evaluate_model(model, dataset_loader, optimizer, criterion, device, type = "loss"):
    LOSSES = 0
    COUNTER = 0
    for batch in dataset_loader:
        optimizer.zero_grad()

        batch_x = batch["sequence"]
        batch_y = batch["target"]

        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        loss = criterion(model(batch_x), batch_y)

        if(type == "loss"):
            n = batch_y.size(0)
            LOSSES += loss.sum().cpu().data.numpy() * n
            COUNTER += n
        else:
            n = batch_y.size(0)
            LOSSES += loss.sum() * n
            COUNTER += n

    return LOSSES / float(COUNTER)
    
def train_loop(model, train_dataloader, device, optimizer, criterion):
    """
    One Iteration across the training set
    Args:
        :param model: solution.Basset()
        :param train_dataloader: torch.utils.data.DataLoader
                                 Where the dataset is solution.BassetDataset
        :param device: torch.device
        :param optimizer: torch.optim
        :param critereon: torch.nn (output of get_critereon)
    :Return: total_score, total_loss.
             float of model score (AUC) and float of model loss for the entire loop (epoch)
             (if you want to display losses and/or scores within the loop, 
             you may print them to screen)
    Make sure your loop works with arbitrarily small dataset sizes!
    Note: you don’t need to compute the score after each training iteration.
    If you do this, your training loop will be really slow!
    You should instead compute it every 50 or so iterations and aggregate ...
    """

    output = {'total_score': 0.,
              'total_loss': 0.}

    # WRITE CODE HERE

    # load the data
    # mnist_train = datasets.MNIST('data', train=True, download=True)
    # mnist_train = list(mnist_train)[:2000]
    # img_to_tensor = transforms.ToTensor()

    # create a new model, initialize random parameters
    model.train()
    
    LOSSES = 0
    COUNTER = 0
    ITERATIONS = 0
    store_every = 50
    learning_curve_loss_train = list()
    learning_curve_auc_train = list()

    # training
    for batch in train_dataloader:
        optimizer.zero_grad()

        batch_x = batch["sequence"]
        batch_y = batch["target"]

        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        model = model.to(device)
            
        loss = criterion(model(batch_x), batch_y)
        loss.backward()
        optimizer.step()
        
        n = batch_y.size(0)
        # print(f"loss sum data type {type(loss.sum()}")
        # print()
        # print(type(loss.sum().data))
        LOSSES += loss.sum().cpu().data.numpy() * n
        COUNTER += n
        ITERATIONS += 1
        if ITERATIONS%(store_every/5) == 0:
            avg_loss = LOSSES / float(COUNTER)
            LOSSES = 0
            COUNTER = 0
            print(" Iteration {}: TRAIN {}".format(
                ITERATIONS, avg_loss))
            output['total_loss'] = avg_loss

    
        if ITERATIONS%(store_every) == 0:     
            train_loss = evaluate_model(model, train_dataloader, optimizer, criterion, device, type = "loss")
            learning_curve_loss_train.append(train_loss)

            train_auc = evaluate_model(model, train_dataloader, optimizer, compute_auc_trained_model, device, type = "auc")
            learning_curve_auc_train.append(train_auc)
                   
            print(" [LOSS] TRAIN {}".format(
                train_loss))
            print(" [AUC] TRAIN {}".format(
                train_auc))

            output['total_score'] = sum(learning_curve_auc_train) / n
    # print(f"TOTAL list auc length: {len(learning_curve_auc_train)}")
    # print(learning_curve_auc_train)
    return output['total_score'], output['total_loss']



def valid_loop(model, valid_dataloader, device, optimizer, criterion):
    """
    One Iteration across the validation set
    Args:
        :param model: solution.Basset()
        :param valid_dataloader: torch.utils.data.DataLoader
                                 Where the dataset is solution.BassetDataset
        :param device: torch.device
        :param optimizer: torch.optim
        :param critereon: torch.nn (output of get_critereon)
    :Return: total_score, total_loss.
             float of model score (AUC) and float of model loss for the entire loop (epoch)
             (if you want to display losses and/or scores within the loop, 
             you may print them to screen)
    Make sure your loop works with arbitrarily small dataset sizes!
    
    Note: if it is taking very long to run, 
    you may do simplifications like with the train_loop.
    """

    output = {'total_score': 0.,
              'total_loss': 0.}

    # WRITE CODE HERE

    val_loss = evaluate_model(model, valid_dataloader, optimizer, criterion, device, type = "loss")
    val_auc = evaluate_model(model, valid_dataloader, optimizer, compute_auc_trained_model, device, type = "auc")
    print(" [LOSS] VALID {}".format(
                val_loss))
    print(" [AUC] VALID {}".format(
                val_auc))

    output['total_score'] = val_auc
    output['total_loss'] = val_loss         
    return output['total_score'], output['total_loss']
