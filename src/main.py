import random
import gc
import numpy as np
import torch
import torch.optim as optim
import pandas as pd
from utils import set_seed,get_metrics, data_chuli,DataLoader
from sklearn.metrics import auc, roc_curve, precision_recall_curve
from Model import main_model
import csv
import json
import time
from sklearn.model_selection import KFold
# set_seed(666)
from sklearn import metrics
from para import parser_add_main_args
from graphT import DIFFormer
import argparse

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser(description='General Training Pipeline')
    parser_add_main_args(parser)
    args = parser.parse_args()
    set_seed(args)
    def _init_fn(worker_id):
        torch.cuda.manual_seed((args.seed))
        torch.manual_seed((args.seed))

    num_workers = 8
    train_loader = DataLoader(datad_train, batch_size=100, shuffle=False, num_workers=num_workers, pin_memory=True,
                             worker_init_fn=_init_fn)
    valid_loader = DataLoader(datad_valid, batch_size=100, shuffle=False, num_workers=num_workers, pin_memory=True,
                             worker_init_fn=_init_fn)
    test_loader = DataLoader(datad_test, batch_size=100, shuffle=False, num_workers=num_workers, pin_memory=True,
                             worker_init_fn=_init_fn)

    model = main_model(args).cuda()
    # model = main_model(args).cuda()
    best_val_loss = float('inf')
    optimizer = optim.AdamW(model.parameters(), args.lr)
    for epoch in range(args.epochs):
        model.train()
        loss_all = 0
        for i, data in enumerate(train_loader):
            # model.train()
            optimizer.zero_grad()
            data.cuda()
            data_copy1_feature = data.x
            out1, embedding1, fea_train = model(data)
            loss_fun = torch.nn.BCELoss(reduction='mean')
            loss_gae = torch.nn.MSELoss(reduction='mean')
            loss_1 = loss_fun(out1, data.y.float())
            loss_2 = loss_gae(embedding1, data_copy1_feature.float())
            loss = loss_1 + loss_2
            loss.backward()
            optimizer.step()
            loss_all += loss
        loss_train = loss_all/len(train_loader)

        y_pred_all = []
        real_all = []
        model.eval()
        loss_all1 = 0
        with torch.no_grad():
            for i, data in enumerate(valid_loader):
                data.cuda()
                data_copy2_feature = data.x
                out2, embedding2, fea_valid = model(data)
                y_pred_all.append(out2)
                real_all.append(data.y.float())
                loss_fun1 = torch.nn.BCELoss(reduction='mean')
                loss_gae = torch.nn.MSELoss(reduction='mean')
                loss = loss_fun(out2, data.y.float()) + loss_gae(embedding2, data_copy2_feature.float())
                loss_all1 += loss
        loss_valid = loss_all1/len(valid_loader)
        y_pred_all = torch.cat(y_pred_all, dim=0).cpu().reshape(-1).numpy()
        real_all = torch.cat(real_all, dim=0).cpu().reshape(-1).numpy()
        metric_tmp = get_metrics(real_all, y_pred_all)
        #if (epoch + 1) % 10 == 0 or epoch == 0:
        print('epoch  {}: train_Loss: {},  valid_loss: {}'.format(epoch + 1, loss_train, loss_valid))
        print('valid_performance:{}'.format(metric_tmp))
        if loss_valid < best_val_loss:
            best_val_loss = loss_valid
            torch.save(model.state_dict(), '../model_para/modelpara_10.pt')
if __name__=="__main__" :
    start_time = time.time()
    train()
    end_time = time.time()
    final_time = (end_time - start_time) / 60
    print(f"train  took {final_time:.2f} minutes")