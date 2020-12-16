import copy
import torch
from time import time
from tqdm import tqdm
from model.bert import BERT
from model.fgm import FGM
from model.pgd import PGD
from torch.optim import AdamW, swa_utils
from reader.reader import tensorize, divide_dataset
from transformers import get_linear_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup

def train(args, tokenizer, array, device):
    # do k-fold
    if len(args.fold) == 1: folds = [args.fold[0]]
    else: folds = range(args.fold[0], args.fold[1]+1)

    for fold in folds:
        # new model for each fold
        model = BERT(args).to(device)
        swa_model = swa_utils.AveragedModel(model, device)

        # use at
        if args.use_at == 'fgm': fgm = FGM(model)
        elif args.use_at == 'pgd': 
            pgd = PGD(model)
            K = 3

        # new tensorized data and maps
        train_data, valid_data = divide_dataset(array, args.num_fold, fold)
        train_loader = tensorize(train_data, tokenizer, args, mode='random')
        valid_loader = tensorize(valid_data, tokenizer, args, mode='seq')

        # new optimizer and scheduler
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                'weight_decay_rate': 0.0},
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

        scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=args.max_epoches*len(train_loader),
            num_cycles=int(args.max_epoches/args.avg_steps)
        )

        # training
        stop_ct, best_F1, best_model, best_epoch = 0, 0, None, -1
        print('training start.. on fold', fold)
        train_losses = 0
        for i in range(args.max_epoches):
            model.train()
            start_time = time()

            # use tqdm
            if args.use_tqdm:
                train_iter = tqdm(train_loader, ncols=50)
            else:
                train_iter = train_loader

            # training process
            for _, batch_data in enumerate(train_iter):
                batch_data = tuple(i.to(device) for i in batch_data)
                ids, masks, labels = batch_data

                model.zero_grad()
                loss, logits = model(ids, masks, labels).to_tuple()

                # process loss
                loss.backward()
                train_losses += loss.item()

                # fgm adversial training
                if args.use_at == 'fgm':
                    fgm.attack()
                    loss_adv, _ = model(ids, masks, labels).to_tuple()
                    loss_adv.backward()
                    fgm.restore()

                # pgd adversarial training
                if args.use_at == 'pgd':
                    pgd.backup_grad()
                    for t in range(K):
                        pgd.attack(is_first_attack=(t==0)) # 在embedding上添加对抗扰动, first attack时备份param.data
                        if t != K-1:
                            model.zero_grad()
                        else:
                            pgd.restore_grad()
                        loss_adv, _ = model(batch_input, batch_label).to_tuple()
                        loss_adv.backward() # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                    pgd.restore() # restore embedding parameters

                # tackle exploding gradients
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=args.max_grad_norm)
                optimizer.step()
                scheduler.step()

            # swa_model.update_parameters(model)
            train_losses /= len(train_loader)
            
            # evaluate
            if (i+1) % args.avg_steps == 0:
                swa_model.eval()
                with torch.no_grad():
                    valid_losses = 0
                    pred_logits, pred_labels = [], []
                    for idx, batch_data in enumerate(valid_loader):
                        print(idx, len(valid_loader))
                        batch_data = tuple(i.to(device) for i in batch_data)
                        ids, masks, labels = batch_data
                        
                        loss, logits = swa_model(ids, masks, labels).to_tuple()
                        pred_logits.append(logits)
                        pred_labels.append(labels)

                        # process loss
                        valid_losses += loss.item()

                    valid_losses /= len(valid_loader)
                    precision, recall, F1 = swa_model.module.calculate_F1(pred_logits, pred_labels)
            
                torch.save(swa_model.module, args.model_dir + '/MOD' + str(fold) + '_' + str(i+1))
                print('Epoch %d train:%.2e valid:%.2e precision:%.4f recall:%.4f F1:%.4f time:%.0f' % \
                    (i+1, train_losses, valid_losses, precision, recall, F1, time()-start_time))

                # if F1 > best_F1:
                #     best_F1 = F1
                #     best_model = copy.deepcopy(swa_model.module)
                #     best_epoch = i+1
                # else:
                #     stop_ct += 1
                #     if stop_ct == args.stop_epoches:
                #         if args.save_models:
                #             torch.save(best_model, args.model_dir + '/MOD' + str(fold) + '_' + str(best_epoch))
                #         break