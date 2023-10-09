
import os
import subprocess
import pickle
import logging
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import AutoTokenizer
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import scipy
from datetime import timedelta
from sklearn.manifold import TSNE
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import math


def entropy(probabilities):
    """Computes the entropy per token.

    :param probabilities: Tensor of shape (bsz,seq_len,refer_dims).
    :return: One entropy per token, shape (bsz,seq_len)
    """
    factor = 1 / math.log(probabilities.shape[-1] + 1e-8)
    return -factor * torch.mean(probabilities * torch.log(probabilities + 1e-8), dim=-1)


def get_center(X, Y, num_class=None):
    '''
        Compute the class center of X, 
        Note that this function is suitable for all classes computation,
        A better implementation will be compute_class_feature_center

        Params:
            - X : a tensor has dims (num_samples, hidden_dims)
            - Y : a tensor has dims (num_samples), and the label indexes are from 0 ~ D-1
            - num_class: an integer indicates the label range (0 ~ num_class-1)
        Return:
            - class_center: a tensor has dims (num_seen_class, hidden_dims)
            - class_seen_mask: a list  has dims (num_class) 
            and it represents the seen classes mask
    '''
    # ensure X and Y in the same divice
    X_device = X.device
    Y = Y.to(X_device)

    # set the number of classes
    if num_class==None:
        num_class = int(X.shape[1])

    # get the mask for the class whose center can be calculated
    class_seen_mask = [True if i in Y else False for i in range(num_class)] 
    class_unseen_mask = [True if i not in Y else False for i in range(num_class)]
    num_class_unseen = int(np.sum(class_unseen_mask)) 

    # add dummy samples for the unseen class
    unseen_class_index = torch.where(torch.tensor(class_unseen_mask))[0].to(X_device)
    Y = torch.cat((Y, unseen_class_index))
    unseen_class_X = torch.zeros((num_class_unseen,X.shape[1])).to(X_device)
    X = torch.cat((X, unseen_class_X), dim=0)

    # convert to one-hot label
    Y = torch.eye(num_class)[Y.long()].to(X_device)

    # get center for all classes
    class_center = torch.matmul(torch.matmul(torch.diag(1/torch.sum(Y, dim=0)),Y.T),X)
    class_center = class_center[class_seen_mask,:]

    return class_center, class_seen_mask

def compute_class_feature_center(dataloader, feature_model, select_class_indexes, is_normalize=True, is_return_flatten_feat_and_Y=False):
    '''
        Get features and targets

        Params:
            - dataloader: torch.utils.data.DataLoader
            - feature_model: a model returns features (w/o FC layers)
            - select_class_indexes: a list of selected classes indexes (e.g. [1,2] or [0,1,2])
            - is_normalize: if normalize the features
            - is_return_flatten_feat_and_Y: if return flatten feature_matrix and flatten Y list
        Return:
            - class_center_matrix: a matrix has dims (num_class, hidden_dims),
            each row is the center of a class
            - features_matrix: a matrix has dims (num_samples, hidden_dims),
            - targets_list_flatten: a list has dims (num_samples)
    '''
    (features_matrix,) = compute_feature_by_dataloader(dataloader, 
                                                    feature_model,
                                                    select_label_groups=[
                                                        select_class_indexes
                                                    ],
                                                    is_normalize=is_normalize)

    targets_list_flatten = torch.tensor(get_flatten_for_nested_list(
                                            dataloader.dataset.y,
                                            select_labels=select_class_indexes))

    class_center_list= [] 
    select_class_mask = torch.zeros_like(targets_list_flatten)
    for class_idx in select_class_indexes:
        class_mask = torch.eq(targets_list_flatten, class_idx)
        select_class_mask = torch.logical_or(select_class_mask, class_mask)
        class_center_list.append(torch.mean(features_matrix[class_mask],dim=0,keepdim=True))
    class_center_matrix = torch.cat(class_center_list, dim=0)

    if is_return_flatten_feat_and_Y:
        return class_center_matrix, features_matrix[select_class_mask], targets_list_flatten[select_class_mask]
    return class_center_matrix

def compute_feature_by_dataloader(dataloader, feature_model, select_label_groups=[], is_normalize=False):
    '''
        Compute the feature of dataloader{(X, Y)}, X has dims (num_sentences, num_words)

        Params:
            - dataloader: torch.utils.data.DataLoader
            - feature_model: a model returns features (w/o FC layers)
            - select_label_groups: select the features of the chosen labels
            and return groups of features if "select_label_groups" is a nested list 
            (e.g.[1,2,3], or [[0],[1,2,3]])
            - is_normalize: if normalize the features
        Return:
            - features_matrix: a groups of such as [(num_samples, hidden_dims),...]
            according to the select_label_groups
    '''
    feature_model.eval() # old model encoder

    num_groups = len(select_label_groups)
    return_feature_groups = [[] for i in range(num_groups)]

    features_list = []
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.cuda() # (bsz, seq_len)
            inputs_feature = feature_model(inputs)[1][-1].cpu() # 最后一层的特征 (bsz, seq_len, hidden_dim)
            if num_groups>0:
                for i in range(num_groups):
                    select_mask = torch.zeros_like(targets) # (bsz, seq_len)
                    for select_label in select_label_groups[i]:
                        select_mask = torch.logical_or(select_mask, targets==select_label)
                    return_feature_groups[i].append(\
                        inputs_feature[select_mask].reshape(-1,inputs_feature.shape[-1]))
            else:
                features_list.append(inputs_feature.reshape(-1,inputs_feature.shape[-1]))
    feature_model.train()

    if num_groups>0:
        for i in range(num_groups):
            return_feature_groups[i] = torch.cat(return_feature_groups[i], dim=0) # 0: (标注为当前任务label的token数量,hidden_dim) 1: (标注为0的token数量,hidden_dim)
        if is_normalize:
            for i in range(num_groups):
                return_feature_groups[i] = F.normalize(return_feature_groups[i], p=2, dim=-1)
        features_matrix = tuple(return_feature_groups)
    else:
        features_matrix = torch.cat(features_list, dim=0)
        if is_normalize:
            features_matrix = F.normalize(features_matrix, p=2, dim=-1)

    return features_matrix

def compute_feature_by_input(X, feature_model, batch_size=64):
    '''
        Compute the feature of X

        Params:
            - X: input sentences have dims (num_sentences, num_words)
            - feature_model: a model returns features (w/o FC layers)
            - batch_size: the batch size for loading
        Return:
            - features_matrix: a matrix has dims (num_samples, hidden_dims)
    '''
    feature_model.eval()
    features_list = []
    start_idx = 0
    with torch.no_grad():
        for inputs in X.spilt(batch_size):
            inputs = inputs.cuda()
            inputs_feature = feature_model(inputs)[1][-1].cpu()
            features_list.append(inputs_feature.reshape(-1,inputs_feature.shape[-1]))
            start_idx = start_idx+inputs.shape[0]
    feature_model.train()
    features_matrix = torch.cat(features_list, dim=0)
    print('features_matrix has shape %s'%str(features_matrix.shape))
    return features_matrix

def pdist(e, squared=False, eps=1e-12):
    '''
        Compute the L2 distance of all features

        Params:
            - e: a feature matrix has dims (num_samples, hidden_dims)
            - squared: if return the squared results of the distance
            - eps: the threshold to avoid negative distance
        Return:
            - res: a distance matrix has dims (num_samples, num_samples)
    '''
    e_square = e.pow(2).sum(dim=1)
    prod = e @ e.t()
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

    if not squared:
        res = res.sqrt()

    res = res.clone()
    res[range(len(e)), range(len(e))] = 0
    return res

def get_match_id(flatten_feat_train, top_k, max_samples=5000):
    '''
        Compute the nearest samples id for each sample,

        Params:
            - flatten_feat_train: a matrix has dims (num_samples, hidden_dims)
            - top_k: for each sample, return the id of the top_k nearest samples
            - max_samples: number of maximum samples for computation.
            if it is set too large, "out of memory" may happen.
        Return:
            - match_id: a list has dims (num_samples*top_k) 
            and it represents the ids of the nearest samples of each sample.
    '''
    num_samples_all = flatten_feat_train.shape[0] # 当前任务训练数据中标注为当前任务label的token数量
    if num_samples_all>max_samples: # 提升效率
        # 2.1. calculate the L2 distance inside z0
        dist_z =  scipy.spatial.distance.cdist(flatten_feat_train,
                                flatten_feat_train[:max_samples],
                                'euclidean')
        dist_z = torch.tensor(dist_z)
        # 2.2. calculate distance mask: do not use itself
        mask_input = torch.clamp(torch.ones_like(dist_z)-torch.eye(num_samples_all, max_samples), 
                                min=0)
    else:
        # 2.1. calculate the L2 distance inside z0        
        # (当前任务训练数据中标注为当前任务label的token数量, 当前任务训练数据中标注为当前任务label的token数量) 
        # 基于这些token的特征两两计算距离   
        dist_z = pdist(flatten_feat_train, squared=False) 
        # 2.2. calculate distance mask: do not use itself
        mask_input = torch.clamp(torch.ones_like(dist_z)-torch.eye(num_samples_all), min=0)
    # 2.3 find the image meets label requirements with nearest old feature
    mask_input = mask_input.float() * dist_z
    mask_input[mask_input == 0] = float("inf") # 自身距离设置为无穷 排除自身
    match_id = torch.flatten(torch.topk(mask_input, top_k, largest=False, dim=1)[1]) # 每个token 找自己特征最相似的其他 三个token 返回id

    # Show the average distance
    # topk_value = torch.topk(dist_z, k=top_k, largest=False, dim=1)[0][:,1:]
    # distance_mean = torch.mean(dist_z,dim=1).reshape(-1,1)
    # topk_ratio = topk_value/distance_mean
    # print(topk_ratio)
    # print(torch.mean(topk_ratio).item())
    return match_id

def get_flatten_for_nested_list(all_label_train, select_labels, is_return_pos_matrix=False, max_seq_length=512):
    '''
        Return a flatten version of the nested_list contains only select_labels,
        and a position matrix. 
        
        Params:
            - all_label_train: a nested list and each element is the token label  2维列表
            - select_labels: a list indicates the select labels  当前任务的label集合
            - is_return_pos_matrix: if return the pos matirx of each select element,
            e.g., [[1,4],[1,5],[2,1],[2,2],...]
            - max_seq_length: the longest length for each sentence
        Return:
            - flatten_label: a flatten version of the nested_list
            - pos_matrix: a "Numpy" matrix has dims (num_samples, 2)
            and it indicates the position (i-th sentence, j-th token) of each entity
    '''
    flatten_list = []
    pos_matrix = []
    for i,s in enumerate(all_label_train):
        if len(s)>max_seq_length:
            s=s[:max_seq_length]
        mask_4_sent = np.isin(s, select_labels) # s中的label 是否在select_labels中，在为true
        pos_4_sent = np.where(mask_4_sent)[0] # 返回true的位置
        flatten_list.extend(np.array(s)[mask_4_sent])
        if is_return_pos_matrix and len(pos_4_sent)>0:
            pos_matrix.append(np.array([[i,j] for j in pos_4_sent]))
    if is_return_pos_matrix:
        if len(pos_matrix)>0:
            pos_matrix = np.concatenate(pos_matrix, axis=0)
        else:
            pos_matrix = []
        return flatten_list, pos_matrix
    return flatten_list

def plot_embedding(X, Y):
    '''
        Plot the feature X

        Params:
            - X: a feature matrix has dims (num_samples, hidden_dims)
            - Y: a label list has dims (num_samples)
    '''
    plt.scatter(X[:,0], 
                X[:,1], 
                c=Y, 
                marker='.',
                cmap=plt.cm.Spectral)

def plot_centers(X, label_list):
    '''
        Plot the feature centers X

        Params:
            - X: a feature matrix has dims (num_classes, hidden_dims)
            - label_list: a list has dims (num_samples) 
            and it represents the name of each class
    '''
    plt.scatter(X[:,0], X[:,1], 
                c=[i+1 for i in range(X.shape[0])], 
                marker='*')
    for i, l_name in enumerate(label_list):
        plt.text(X[i,0], X[i,1],
                s=str(l_name),
                size=15)

def plot_distribution(X, Y, label_list, class_center_matrix=None, sample_ratio=1.0, select_labels=None):
    '''
        Visualize the feature X in the 2-D space

        Params:
            - X: a feature matrix has dims (num_samples, hidden_dims)
            - Y: a label list has dims (num_samples)
            - label_list: a list has dims (num_classes) 
            and it represents the name of each class
            - class_center_matrix: if not None, plot the class center of each class;
            it has dims (num_classes, hidden_dims)
            - sample_ratio: the ratio of the samples used for visualization
            - select_labels: a list represents the selected labels for visualization
    '''
    # clone and convert to tensor
    if isinstance(X, list):
        _X = torch.tensor(X)
    else:
        _X = X.clone().detach().cpu()
    if isinstance(Y, list):
        _Y = torch.tensor(Y)
    else:
        _Y = Y.clone().detach().cpu()
    num_samples = _Y.shape[0]
    print('Total %d samples for visualization'%num_samples)

    # random sampling
    if sample_ratio<1.0:
        assert sample_ratio>0.0, "Invalid sample ratio!!!"
        
        sample_lst = list(range(num_samples))
        random.shuffle(sample_lst)
        sample_lst = sample_lst[:int(num_samples*sample_ratio)]
        _X = _X[sample_lst]
        _Y = _Y[sample_lst]
        print('Select %d samples for visualization'%_Y.shape[0])

    if select_labels!=None and len(select_labels)>0:
        for i,l in enumerate(select_labels):
            if i==0:
                class_mask = np.equal(_Y,l)
            else:
                class_mask = np.logical_or(class_mask,np.equal(_Y,l))
        _Y = _Y[class_mask]
        _X = _X[class_mask]
    
    # t-SNE for visualization
    tsne = TSNE(n_components=2)
    if not class_center_matrix is None:
        assert len(label_list)==class_center_matrix.shape[0], "Number of classes is not consistent!!!"
        num_class = class_center_matrix.shape[0]
        concat_X = torch.cat((_X, class_center_matrix),dim=0)
        concat_low_repre = torch.tensor(tsne.fit_transform(concat_X))

        # scale to 0-1
        x_min, x_max = torch.min(concat_low_repre, 0)[0], torch.max(concat_low_repre, 0)[0]
        concat_low_repre = (concat_low_repre - x_min) / (x_max - x_min)

        low_repre = concat_low_repre[:-num_class,:]
        plot_embedding(low_repre, _Y)
        class_low_repre = concat_low_repre[-num_class:,:]
        plot_centers(class_low_repre, label_list)
        plt.show()
    else:
        low_repre = torch.tensor(tsne.fit_transform(_X))

        # scale to 0-1
        x_min, x_max = torch.min(low_repre, 0)[0], torch.max(low_repre, 0)[0]
        low_repre = (low_repre - x_min) / (x_max - x_min)

        plot_embedding(low_repre, _Y)
        plt.show()
    

def plot_confusion_matrix(pred_list, y_list, label_list, pad_token_label_id=-100):
    '''
        Plot confusion matrix for model predictions

        Params:
            - pred_list: a tensor has dims (num_samples,)
            - y_list: a tensor has dims (num_samples,)
            - label_list: a list indicates the label list
            - pad_token_label_id: a index for padding label
    '''
    # filter out padding label
    pred_list, y_list = torch.tensor(pred_list), torch.tensor(y_list)
    pad_mask = torch.not_equal(y_list, pad_token_label_id)
    pred_list, y_list = pred_list[pad_mask], y_list[pad_mask]

    pred_list = list(pred_list.numpy())
    y_list = list(y_list.numpy())

    O_index = label_list.index('O')
    cm = confusion_matrix(y_list, pred_list)
    cm_without_o = np.concatenate((cm[:O_index,:],cm[O_index+1:,:]),axis=0)
    cm_without_o = np.concatenate((cm_without_o[:,:O_index],cm_without_o[:,O_index+1:]),axis=1)
    df = pd.DataFrame(cm_without_o,
                    columns=label_list[:O_index]+label_list[O_index+1:],
                    index=label_list[:O_index]+label_list[O_index+1:])
    cmap = sns.color_palette("mako", as_cmap=True)
    sns.heatmap(df, cmap=cmap, xticklabels=True, yticklabels=True,annot=True)
    plt.xticks(rotation=-45)
    plt.xlabel('Predict label')
    plt.ylabel('Actual label')
    plt.show()

def plot_prob_hist_each_class(y_list, logits_list, ignore_label_lst=[-100,0]):
    '''
        Plot probability histogram for each class

        Params:
            - y_list: a tensor has dims (num_samples,)
            - logits_list: a tensor has dims (num_samples, num_classes)
    '''
    
    pad_mask = torch.not_equal(y_list, -100)
    y_list, logits_list = y_list[pad_mask], logits_list[pad_mask]
    
    pred_list = torch.argmax(logits_list, dim=-1)
    prob_list = torch.softmax(logits_list, dim=-1)
    
    for label_id in list(set(np.array(y_list))):
        if label_id in ignore_label_lst:
            continue
        # print("label_id=%d:"%label_id)
        y_mask = torch.eq(pred_list, label_id)
        y_mask_correct = torch.logical_and(\
                            torch.eq(y_list, label_id),
                            y_mask)
        y_mask_wrong = torch.logical_and(\
                            torch.not_equal(y_list, label_id),
                            y_mask)
        y_logits_correct = np.array(prob_list[y_mask_correct][:,label_id])
        y_logits_wrong = np.array(prob_list[y_mask_wrong][:,label_id])
        print(len(y_logits_correct))
        print(len(y_logits_wrong))
        plt.hist([y_logits_correct,y_logits_wrong], 
                    bins=list(np.arange(0,0.9,0.1))\
                            +[0.9, 0.99, 0.999, 0.9999, 0.99999, 1],
                    color=['green','red'], 
                    alpha=0.75)
        plt.legend(['Correct','Wrong'])
        plt.title('Prob distribution for class idx %d'%label_id)
        plt.show()

def decode_sentence(sentence, auto_tokenizer):
    '''
        Decode the sentences batch from ids to words (string)

        Params:
            - sentence: a list of ids (encoded by the tokenizer)
            - auto_tokenizer: a tokenizer for the transformers
        Return:
            - sent_str: sentence string
    '''
    sent_str = ''
    for word_id in sentence:
        word = str(auto_tokenizer.decode(word_id))
        # skip the special tokens ['[PAD]','[CLS]','[SEP]','[UNK]','MASK']
        if word in ['[PAD]','[CLS]','[SEP]','[UNK]','MASK']: 
            continue
        # concat the subwords
        if word.find('##')==0: 
            sent_str = sent_str+word[2:]
        else:
            sent_str = sent_str+' '+word
    return sent_str

def decode_word_from_sentence(sentence, pos_idx, auto_tokenizer):
    '''
        Decode the i-th word from sentence

        Params:
            - sentence: a list of ids (encoded by the tokenizer)
            - pos_idx: the position index of the word
            - auto_tokenizer: a tokenizer for the transformers
        Returns:
            - word_str: a string of the selected word

    '''
    word_str = auto_tokenizer.decode(sentence[pos_idx])
    tmp_cnt = 1
    while len(sentence)>pos_idx+tmp_cnt:
        next_word = auto_tokenizer.decode(sentence[pos_idx+tmp_cnt])
        # skip the special tokens ['[PAD]','[CLS]','[SEP]','[UNK]','MASK']
        if next_word in ['[PAD]','[CLS]','[SEP]','[UNK]','MASK'] or next_word.find('##')!=0:
            break
        word_str = word_str + next_word[2:]
        tmp_cnt += 1
    return word_str

def assert_no_old_samples(labels, ref_dims, all_dims, pad_token_label_id):
    '''
        Check the labels contains no samples from old classes

        Params:
            - labels: a 2-dimentional tensor
            - ref_dims: the dimension of old classes
            - all_dims: the dimension of all classes
            - pad_token_label_id: a index for padding label
    '''
    no_pad_nonzero_mask = torch.logical_and(labels!=pad_token_label_id, labels!=0)
    if no_pad_nonzero_mask.any():
        assert labels[no_pad_nonzero_mask].max()<all_dims and \
            labels[no_pad_nonzero_mask].min()>=refer_dims, \
            "the training data contains old classes!!!"

def init_experiment(params, logger_filename):
    '''
        Initialize the experiment, save parameters and create a logger

        Params:
            - params: a dict contains all hyper-parameters and experimental settings
            - logger_filename: the logger file name
    '''
    # create save path
    get_saved_path(params)

    # create a logger
    logger = create_logger(os.path.join(params.dump_path, logger_filename))
    logger.info('============ Initialized logger ============')
    logger.info('\n'.join('%s: %s' % (k, str(v))
                          for k, v in sorted(dict(vars(params)).items())))
    logger.info('The experiment will be stored in %s\n' % params.dump_path)

    return logger


class LogFormatter():
    '''
        A formatter adding date and time informations
    '''
    def __init__(self):
        self.start_time = time.time()

    def format(self, record):
        elapsed_seconds = round(record.created - self.start_time)

        prefix = "%s - %s - %s" % (
            record.levelname,
            time.strftime('%x %X'),
            timedelta(seconds=elapsed_seconds)
        )
        message = record.getMessage()
        message = message.replace('\n', '\n' + ' ' * (len(prefix) + 3))
        return "%s - %s" % (prefix, message) if message else ''


def create_logger(filepath):
    '''
        Create logger for the experiment

        Params:
            - filepath: the path which the log file is saved
    '''
    # create log formatter
    log_formatter = LogFormatter()
    
    # create file handler and set level to debug
    if filepath is not None:
        file_handler = logging.FileHandler(filepath, "a")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(log_formatter)

    # create console handler and set level to info
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_formatter)

    # create logger and set level to debug
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if filepath is not None:
        logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # reset logger elapsed time
    def reset_time():
        log_formatter.start_time = time.time()
    logger.reset_time = reset_time

    return logger


def get_saved_path(params):
    '''
        Create a directory to store the experiment

        Params:
            - params: a dict contains all hyper-parameters and experimental settings
    '''
    dump_path = "./" if params.dump_path == "" else params.dump_path
    if not os.path.isdir(dump_path):
        subprocess.Popen("mkdir -p %s" % dump_path, shell=True).wait()
    assert os.path.isdir(dump_path)

    # create experiment path if it does not exist
    exp_path = os.path.join(dump_path, params.exp_name)
    if not os.path.exists(exp_path):
        subprocess.Popen("mkdir -p %s" % exp_path, shell=True).wait()
    
    # generate id for this experiment
    if params.exp_id == "":
        chars = "0123456789"
        while True:
            exp_id = "".join(random.choice(chars) for _ in range(0, 3))
            if not os.path.isdir(os.path.join(exp_path, exp_id)):
                break
    else:
        exp_id = params.exp_id
    # update dump_path
    params.dump_path = os.path.join(exp_path, exp_id)
    if not os.path.isdir(params.dump_path):
        subprocess.Popen("mkdir -p %s" % params.dump_path, shell=True).wait()
    assert os.path.isdir(params.dump_path)
