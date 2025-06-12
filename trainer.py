import os
import sys

sys.path.append("..")
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.loss import NTXentLoss
from sklearn.metrics import f1_score



def Trainer(model, temporal_contr_model, model_optimizer, temp_cont_optimizer, train_dl, valid_dl, test_dl, device, logger, config, experiment_log_dir, training_mode):
    # Start training
    logger.debug("Training started ....")

    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, 'min')

    for epoch in range(1, config.num_epoch + 1):
        # Train and validate
        train_loss, train_acc = model_train(model, temporal_contr_model, model_optimizer, temp_cont_optimizer, criterion, train_dl, config, device, training_mode)
        valid_loss, valid_acc, _, _ = model_evaluate(model, temporal_contr_model, valid_dl, device, training_mode)
        if training_mode != 'self_supervised':  # use scheduler in all other modes.
            scheduler.step(valid_loss)

        logger.debug(f'\nEpoch : {epoch}\n'
                     f'Train Loss     : {train_loss:.4f}\t | \tTrain Accuracy     : {train_acc:2.4f}\n'
                     f'Valid Loss     : {valid_loss:.4f}\t | \tValid Accuracy     : {valid_acc:2.4f}')

    os.makedirs(os.path.join(experiment_log_dir, "saved_models"), exist_ok=True)
    chkpoint = {'model_state_dict': model.state_dict(), 'temporal_contr_model_state_dict': temporal_contr_model.state_dict()}
    torch.save(chkpoint, os.path.join(experiment_log_dir, "saved_models", f'ckp_last.pt'))

    if training_mode != "self_supervised":  # no need to run the evaluation for self-supervised mode.
        # evaluate on the test set
        logger.debug('\nEvaluate on the Test set:')
        test_loss, test_acc, outs, trgs = model_evaluate(model, temporal_contr_model, test_dl, device, training_mode)
        mf1 = f1_score(trgs, outs, average='macro', zero_division=0)
        logger.debug(f'Test loss      :{test_loss:0.4f}\t | Test Accuracy      : {test_acc:0.4f}')
        logger.debug(f'Test Macro F1   :{mf1:0.4f}')

    logger.debug("\n################## Training is Done! #########################")


def model_train(model, temporal_contr_model, model_optimizer, temp_cont_optimizer, criterion, train_loader, config, device, training_mode):
    total_loss = []
    total_acc = []
    model.train()
    temporal_contr_model.train()

    if training_mode == "self_supervised":
        for batch_idx, (data, labels, aug1, aug2) in enumerate(train_loader):
            data, labels = data.float().to(device), labels.long().to(device)
            aug1, aug2 = aug1.float().to(device), aug2.float().to(device)

            model_optimizer.zero_grad()
            temp_cont_optimizer.zero_grad()

            predictions1, features1 = model(aug1)
            predictions2, features2 = model(aug2)

            features1 = F.normalize(features1, dim=1)
            features2 = F.normalize(features2, dim=1)

            temp_cont_loss1, temp_cont_lstm_feat1 = temporal_contr_model(features1, features2)
            temp_cont_loss2, temp_cont_lstm_feat2 = temporal_contr_model(features2, features1)

            zis = temp_cont_lstm_feat1
            zjs = temp_cont_lstm_feat2

            lambda1 = 1
            lambda2 = 0.7
            nt_xent_criterion = NTXentLoss(
            device=device,
            temperature=config.Context_Cont.temperature,
            use_cosine_similarity=config.Context_Cont.use_cosine_similarity
            )



            loss = (temp_cont_loss1 + temp_cont_loss2) * lambda1 + nt_xent_criterion(zis, zjs) * lambda2

            total_loss.append(loss.item())
            loss.backward()
            model_optimizer.step()
            temp_cont_optimizer.step()

        total_acc = 0  # accuracy not used in self-supervised

    else:
        for batch_idx, batch in enumerate(train_loader):
            # Handle both 2-tuple (data, labels) and 4-tuple (data, labels, aug1, aug2)
            if len(batch) == 4:
                data, labels = batch[0], batch[1]
            else:
                data, labels = batch

            data, labels = data.float().to(device), labels.long().to(device)

            model_optimizer.zero_grad()

            predictions, features = model(data)
            loss = criterion(predictions, labels)

            total_loss.append(loss.item())
            loss.backward()
            model_optimizer.step()

            total_acc.append(labels.eq(predictions.detach().argmax(dim=1)).float().mean())

        total_acc = torch.tensor(total_acc).mean()

    total_loss = torch.tensor(total_loss).mean()
    return total_loss, total_acc




def model_evaluate(model, temporal_contr_model, test_dl, device, training_mode):
    model.eval()
    temporal_contr_model.eval()

    total_loss = []
    total_acc = []

    criterion = nn.CrossEntropyLoss()
    outs = np.array([])
    trgs = np.array([])

    with torch.no_grad():
        for batch in test_dl:
            # ✅ Robust unpacking: support for both (data, labels) and (data, labels, aug1, aug2)
            if len(batch) == 4:
                data, labels = batch[0], batch[1]
            else:
                data, labels = batch

            data, labels = data.float().to(device), labels.long().to(device)

            if training_mode != "self_supervised":
                output = model(data)
                predictions, features = output

                loss = criterion(predictions, labels)
                total_acc.append(labels.eq(predictions.detach().argmax(dim=1)).float().mean())
                total_loss.append(loss.item())

                pred = predictions.max(1, keepdim=True)[1]
                outs = np.append(outs, pred.cpu().numpy())
                trgs = np.append(trgs, labels.data.cpu().numpy())

    if training_mode == "self_supervised":
        return 0, 0, [], []

    total_loss = torch.tensor(total_loss).mean()
    total_acc = torch.tensor(total_acc).mean()
    return total_loss, total_acc, outs, trgs
