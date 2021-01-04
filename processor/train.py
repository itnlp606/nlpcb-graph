import copy
import torch
from time import time
from tqdm import tqdm
from model.bert import BERTCLAS, BERTNER
from model.fgm import FGM
from model.pgd import PGD
from torch.optim import swa_utils
from utils.optim import get_optimizer_scheduler
from utils.utils import divide_dataset
from reader.reader import clas_tensorize, ner_tensorize

def train(args, tokenizer, array, device):
    # decide tensorize, model based on task
    if args.task == 'clas':
        tensorize = clas_tensorize
        BERT = BERTCLAS
    elif args.task == 'ner':
        tensorize = ner_tensorize
        BERT = BERTNER

    # do k-fold
    if len(args.fold) == 1: folds = [args.fold[0]]
    else: folds = range(args.fold[0], args.fold[1]+1)

    for fold in folds:
        # new model for each fold
        model = BERT(args, device).to(device)

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
            K = 3

        # new tensorized data and maps
        train_data, valid_data = divide_dataset(array, args.num_fold, fold)
        pos_loader, neg_loader = tensorize(train_data, tokenizer, args, mode='random')
        valid_loader = tensorize(valid_data, tokenizer, args, mode='seq')

        # optim
        training_steps = args.max_epoches*len(pos_loader)
        optimizer, scheduler = get_optimizer_scheduler(args, model, training_steps)

        # training
        stop_ct, best_F1, best_model, best_epoch = 0, 0, None, -1
        train_losses = 0
        start_time = time()

        print('training start.. on fold', fold)
        for i in range(args.max_epoches):
            model.train()

            # use tqdm
            if args.use_tqdm:
                train_iter = tqdm(zip(pos_loader, neg_loader), ncols=50, total=len(pos_loader))
                valid_iter = tqdm(valid_loader, ncols=50)
                train_iter.set_description('Train')
                valid_iter.set_description('Test')
            else:
                train_iter = zip(pos_loader, neg_loader)
                valid_iter = valid_loader

            # training process
            for kdx, (pos_data, neg_data) in enumerate(train_iter):
                if kdx == 1: break
                model.zero_grad()
                loss, logits = model(pos_data, neg_data).to_tuple()
                if args.use_at == 'pgd':
                    _, _, ori_F1 = model.calculate_F1([logits], [labels])

                # process loss
                loss.backward()
                train_losses += loss.item()

                # fgm adversial training
                if args.use_at == 'fgm':
                    fgm.attack()
                    loss_adv, _ = model(pos_data, neg_data).to_tuple()
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
                        loss_adv, at_logits = model(pos_data, neg_data).to_tuple()
                        _, _, new_F1 = model.calculate_F1([at_logits], [labels])
                        loss_adv.backward() # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                        if new_F1 < ori_F1:
                            break

                    pgd.restore() # restore embedding parameters

                # tackle exploding gradients
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=args.max_grad_norm)
                optimizer.step()
                scheduler.step()

            train_losses /= len(pos_loader)
            
            # evaluate
            steps = max(1, args.avg_steps)
            if (i+1) % steps == 0:
                if args.avg_steps:
                    swa_model.update_parameters(model)
                with torch.no_grad():
                    valid_model.eval()
                    precision, recall, F1, valid_losses = valid_module.calculate_F1(valid_iter, valid_model)
            
                if args.save_models and args.avg_steps > 0:
                    torch.save(valid_module, args.model_dir + '/MOD' + str(fold) + '_' + str(i+1))

                print('Epoch %d train:%.2e valid:%.2e precision:%.4f recall:%.4f F1:%.4f time:%.0f' % \
                    (i+1, train_losses, valid_losses, precision, recall, F1, time()-start_time))
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