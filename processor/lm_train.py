import copy
import torch
from time import time
from tqdm import tqdm
from model.bert import BERTLM
from model.fgm import FGM
from model.pgd import PGD
from torch.optim import swa_utils
from utils.optim import get_optimizer_scheduler
from processor.preprocessor import lm_tensorize
from utils.utils import divide_dataset

def lm_train(args, tokenizer, array, device):
    # do k-fold
    if len(args.fold) == 1: folds = [args.fold[0]]
    else: folds = range(args.fold[0], args.fold[1]+1)

    for fold in folds:
        # new model for each fold
        model = BERTLM(args).to(device)
        print('training start.. on fold', fold)

        if args.avg_steps:
            swa_model = swa_utils.AveragedModel(model, device)
            valid_model = swa_model
            valid_module = swa_model.module
        else:
            valid_model, valid_module = model, model

        # use at
        if args.use_at == 'fgm': fgm = FGM(model)
        elif args.use_at == 'pgd': 
            pgd = PGD(model)
            K = args.pgd_K

        # new tensorized data and maps
        train_data, valid_data = divide_dataset(args.seed, array, args.num_fold, fold)
        train_loader = lm_tensorize(train_data, tokenizer, args, mode='random')
        valid_loader = lm_tensorize(valid_data, tokenizer, args, mode='seq')
        len_train = len(train_loader)

        # optim
        training_steps = args.max_epoches*len_train
        optimizer, scheduler = get_optimizer_scheduler(args, model, training_steps)

        # training
        stop_ct, best_F1, best_model, best_epoch = 0, 0, None, -1
        train_losses = 0
        start_time = time()
        for i in range(args.max_epoches):
            model.train()

            # use tqdm
            if args.use_tqdm:
                train_iter = tqdm(train_loader, ncols=50)
                valid_iter = tqdm(valid_loader, ncols=50)
                train_iter.set_description('Train')
                valid_iter.set_description('Test')
            else:
                train_iter = train_loader
                valid_iter = valid_loader

            # training process
            for kdx, batch_data in enumerate(train_iter):          
                batch_data = tuple(i.to(device) for i in batch_data)
                ids, masks, genes, labels = batch_data

                model.zero_grad()
                loss, logits = model(ids, masks, labels)
                if args.use_at == 'pgd':
                    ori_F1 = model.calculate_F1([logits], [labels], [genes])

                # process loss
                loss.backward()
                train_losses += loss.item()

                # fgm adversial training
                if args.use_at == 'fgm':
                    fgm.attack()
                    loss_adv, _ = model(ids, masks, labels)
                    loss_adv.backward()
                    fgm.restore()

                # pgd adversarial training
                elif args.use_at == 'pgd':
                    pgd.backup_grad()
                    for t in range(K):
                        pgd.attack(is_first_attack=(t==0)) # 在embedding上添加对抗扰动, first attack时备份param.data
                        if t != K-1:
                            model.zero_grad()
                        else:
                            pgd.restore_grad()
                        loss_adv, at_logits = model(ids, masks, labels)
                        new_F1 = model.calculate_F1([at_logits], [labels], [genes])
                        if new_F1 < ori_F1:
                            pgd.restore_grad()
                            loss_adv.backward()
                            break
                        loss_adv.backward() # 反向传播，并在正常的grad基础上，累加对抗训练的梯度

                    pgd.restore() # restore embedding parameters

                # tackle exploding gradients
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=args.max_grad_norm)
                optimizer.step()
                scheduler.step()

            train_losses /= len_train
            
            # evaluate
            steps = max(1, args.avg_steps)
            if (i+1) % steps == 0:
                if args.avg_steps:
                    swa_model.update_parameters(model)
                with torch.no_grad():
                    valid_model.eval()
                    valid_losses = 0
                    pred_logits, pred_labels, gs = [], [], []
                    for idx, batch_data in enumerate(valid_iter):
                        batch_data = tuple(i.to(device) for i in batch_data)
                        ids, masks, genes, labels = batch_data
                        
                        loss, logits = valid_model(ids, masks, labels)

                        pred_logits.append(logits)
                        pred_labels.append(labels)
                        gs.append(genes)

                        # process loss
                        valid_losses += loss.item()

                    valid_losses /= len(valid_loader)

                    F1 = valid_module.calculate_F1(pred_logits, pred_labels, gs)
            
                if args.save_models and args.avg_steps > 0:
                    torch.save(valid_module, args.model_dir + '/MOD' + str(fold) + '_' + str(i+1))

                print('Epoch %d train:%.2e valid:%.2e precision:%.4f time:%.0f' % \
                    (i+1, train_losses, valid_losses, F1, time()-start_time))
                start_time = time()

                if args.avg_steps == 0:
                    if F1 > best_F1:
                        stop_ct = 0
                        best_F1 = F1
                        best_model = copy.deepcopy(valid_module)
                        best_epoch = i+1
                    else:
                        stop_ct += 1
                        if stop_ct == args.stop_epoches:
                            if args.save_models:
                                torch.save(best_model, args.model_dir + '/MOD' + str(fold) + '_' + str(best_epoch))
                            break

                if i == args.max_epoches-1 and args.save_models:
                    torch.save(best_model, args.model_dir + '/MOD' + str(fold) + '_' + str(best_epoch))