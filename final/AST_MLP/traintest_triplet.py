import sys
import os
from utilities import *
import time
import torch
from torch import nn
import numpy as np
import pickle

def train_tri(audio_model, train_loader, test_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('running on ' + str(device))
    torch.set_grad_enabled(True)

    loss_meter = AverageMeter()
    best_epoch, best_cum_epoch, best_mAP, best_acc, best_cum_mAP = 0, 0, -np.inf, -np.inf, -np.inf
    global_step, epoch = 0, 0
    exp_dir = args.exp_dir
    classes_num = args.n_class
    loss_weight = 0.05

    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)

    audio_model = audio_model.to(device)
    trainables = [p for p in audio_model.parameters() if p.requires_grad]
    mlp_list = ['mlp_head.0.weight', 'mlp_head.0.bias', 'mlp_head.1.weight', 'mlp_head.1.bias']
    mlp_params = list(filter(lambda kv: kv[0] in mlp_list, audio_model.module.named_parameters()))
    base_params = list(filter(lambda kv: kv[0] not in mlp_list, audio_model.module.named_parameters()))
    mlp_params = [i[1] for i in mlp_params]
    base_params = [i[1] for i in base_params]
    optimizer = torch.optim.Adam([{'params': base_params, 'lr': args.lr}, {'params': mlp_params, 'lr': args.lr * args.head_lr}], weight_decay=5e-7, betas=(0.95, 0.999))
    mlp_lr = optimizer.param_groups[1]['lr']
    lr_list = [args.lr, mlp_lr]

    if args.adaptschedule == True:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=args.lr_patience, verbose=True)
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, list(range(args.lrscheduler_start, 1000, args.lrscheduler_step)),gamma=args.lrscheduler_decay)
    main_metrics = args.metrics
    if args.loss == 'BCE':
        loss_fn = nn.BCEWithLogitsLoss()
        args.loss_fn = loss_fn
    elif args.loss == 'CE':
        loss_fn = nn.CrossEntropyLoss()
        args.loss_fn = loss_fn
    elif args.loss == 'BCE_TRI':
        loss_fn = nn.BCEWithLogitsLoss()
        args.loss_fn = loss_fn
    elif args.loss == 'CE_TRI':
        loss_fn = nn.CrossEntropyLoss()
        args.loss_fn = loss_fn
    loss_func2 = nn.TripletMarginLoss(margin=1.0, p=2)
    args.loss_fun2 = loss_func2

    epoch += 1
    print("current #steps=%s, #epochs=%s" % (global_step, epoch))
    result = np.zeros([args.n_epochs, 10])
    audio_model.train()
    while epoch < args.n_epochs + 1:
        begin_time = time.time()
        audio_model.train()
        print('---------------')
        print("current #epochs=%s, #steps=%s" % (epoch, global_step))

        for i, (audio_input, labels) in enumerate(train_loader):

            B = audio_input.size(0)
            audio_input = audio_input.to(device, non_blocking=True)

            if global_step <= 1000 and global_step % 50 == 0 and args.warmup == True:
                for group_id, param_group in enumerate(optimizer.param_groups):
                    warm_lr = (global_step / 1000) * lr_list[group_id]
                    param_group['lr'] = warm_lr
                    
            audio_output = audio_model(audio_input, args.task)
            labels_tri = labels.numpy().argmax(1)
            label_to_indices = {label: np.where(labels_tri == label)[0] for label in range(classes_num)}

            pos_index_list = []
            neg_index_list = []
            for j in range(args.batch_size):
                now_label = labels_tri[j]
                label_set = [*range(classes_num)]

                pos_label = now_label
                pos_index = random.choice(label_to_indices[pos_label])
                label_set.remove(pos_label) 
                if classes_num == 3:
                    other_label_concat = np.concatenate((label_to_indices[label_set[0]], label_to_indices[label_set[1]]))
                elif classes_num == 2:
                    other_label_concat = label_to_indices[label_set[0]]
                elif classes_num == 4:
                    other_label_concat = np.concatenate((label_to_indices[label_set[0]], label_to_indices[label_set[1]], label_to_indices[label_set[2]]))

                if len(other_label_concat) == 0:
                    continue
                neg_index = random.choice(other_label_concat)

                pos_index_list.append(pos_index)
                neg_index_list.append(neg_index)

            if len(pos_index_list) != args.batch_size:
                continue
            pos_embedding = audio_output['embedding'][pos_index_list]
            neg_embedding = audio_output['embedding'][neg_index_list]

            
            labels = labels.to(device, non_blocking=True)
            if isinstance(loss_fn, torch.nn.CrossEntropyLoss):
                if "TRI" in args.loss:
                    loss = (1 - loss_weight) * loss_fn(audio_output['x'], torch.argmax(labels.long(), axis=1)) + loss_weight * loss_func2(audio_output['embedding'], pos_embedding, neg_embedding)
                else:
                    loss = loss_fn(audio_output['x'], torch.argmax(labels.long(), axis=1))
            else:
                if "TRI" in args.loss:
                    loss = (1 - loss_weight) * loss_fn(audio_output['x'], labels) + loss_weight * loss_func2(audio_output['embedding'], pos_embedding, neg_embedding)
                else:
                    loss = loss_fn(audio_output['x'], labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_meter.update(loss.item(), B)            
            print_step = global_step % args.n_print_steps == 0
            early_print_step = epoch == 0 and global_step % (args.n_print_steps/10) == 0
            print_step = print_step or early_print_step

            if print_step and global_step != 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                  'Train Loss {loss_meter.avg:.4f}\t'.format(
                   epoch, i, len(train_loader), loss_meter=loss_meter), flush=True)

            global_step += 1
        stats, valid_loss = validate_tri(audio_model, test_loader, args, epoch)

        cum_stats = validate_ensemble(args, epoch)
        cum_mAP = np.mean([stat['AP'] for stat in cum_stats])
        cum_mAUC = np.mean([stat['auc'] for stat in cum_stats])
        cum_acc = cum_stats[0]['acc']

        mAP = np.mean([stat['AP'] for stat in stats])
        mAUC = np.mean([stat['auc'] for stat in stats])
        acc = stats[0]['acc']

        middle_ps = [stat['precisions'][int(len(stat['precisions'])/2)] for stat in stats]
        middle_rs = [stat['recalls'][int(len(stat['recalls'])/2)] for stat in stats]
        average_precision = np.mean(middle_ps)
        average_recall = np.mean(middle_rs)

        if main_metrics == 'mAP':
            print("mAP: {:.6f}".format(mAP))
        else:
            print("acc: {:.6f}".format(acc))
        print("AUC: {:.6f}".format(mAUC))
        print("Avg Precision: {:.6f}".format(average_precision))
        print("Avg Recall: {:.6f}".format(average_recall))
        print("d_prime: {:.6f}".format(d_prime(mAUC)))
        print("train_loss: {:.6f}".format(loss_meter.avg))
        print("valid_loss: {:.6f}".format(valid_loss))

        if main_metrics == 'mAP':
            result[epoch-1, :] = [mAP, mAUC, average_precision, average_recall, d_prime(mAUC), loss_meter.avg, valid_loss, cum_mAP, cum_mAUC, optimizer.param_groups[0]['lr']]
        else:
            result[epoch-1, :] = [acc, mAUC, average_precision, average_recall, d_prime(mAUC), loss_meter.avg, valid_loss, cum_acc, cum_mAUC, optimizer.param_groups[0]['lr']]
        np.savetxt(exp_dir + '/result.csv', result, delimiter=',')
        print('validation finished')

        if mAP > best_mAP:
            best_mAP = mAP
            if main_metrics == 'mAP':
                best_epoch = epoch

        if acc > best_acc:
            best_acc = acc
            if main_metrics == 'acc':
                best_epoch = epoch

        if cum_mAP > best_cum_mAP:
            best_cum_epoch = epoch
            best_cum_mAP = cum_mAP

        if best_epoch == epoch:
            torch.save(audio_model.state_dict(), "%s/models/best_audio_model.pth" % (exp_dir))
            torch.save(optimizer.state_dict(), "%s/models/best_optim_state.pth" % (exp_dir))

        torch.save(audio_model.state_dict(), "%s/models/audio_model.%d.pth" % (exp_dir, epoch))
        if len(train_loader.dataset) > 2e5:
            torch.save(optimizer.state_dict(), "%s/models/optim_state.%d.pth" % (exp_dir, epoch))

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            print('adaptive learning rate scheduler step')
            scheduler.step(mAP)
        else:
            print('normal learning rate scheduler step')
            scheduler.step()

        print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))
        print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[1]['lr']))

        with open(exp_dir + '/stats_' + str(epoch) +'.pickle', 'wb') as handle:
            pickle.dump(stats, handle, protocol=pickle.HIGHEST_PROTOCOL)

        finish_time = time.time()
        print('epoch {:d} training time: {:.3f}'.format(epoch, finish_time-begin_time))

        epoch += 1
        loss_meter.reset()

def validate_tri(audio_model, val_loader, args, epoch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_time = AverageMeter()
    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)
    audio_model = audio_model.to(device)
    audio_model.eval()
    classes_num = args.n_class
    loss_weight = 0.05
    
    end = time.time()
    A_predictions = []
    A_targets = []
    A_loss = []
    with torch.no_grad():
        for i, (audio_input, labels) in enumerate(val_loader):
            audio_input = audio_input.to(device)

            audio_output = audio_model(audio_input, args.task)
            audio_output_ = torch.sigmoid(audio_output['x'])
            predictions = audio_output_.to('cpu').detach()

            A_predictions.append(predictions)
            A_targets.append(labels)
            
            labels_tri = labels.numpy().argmax(1)
            label_to_indices = {label: np.where(labels_tri == label)[0] for label in range(classes_num)}
            
            pos_index_list = []
            neg_index_list = []
            for j in range(len(labels_tri)):
                now_label = labels_tri[j]
                label_set = [*range(classes_num)]

                pos_label = now_label
                pos_index = random.choice(label_to_indices[pos_label])
                label_set.remove(pos_label) 
                if classes_num == 3:
                    other_label_concat = np.concatenate((label_to_indices[label_set[0]], label_to_indices[label_set[1]]))
                elif classes_num == 2:
                    other_label_concat = label_to_indices[label_set[0]]
                elif classes_num == 4:
                    other_label_concat = np.concatenate((label_to_indices[label_set[0]], label_to_indices[label_set[1]], label_to_indices[label_set[2]]))

                if len(other_label_concat) == 0:
                    continue
                neg_index = random.choice(other_label_concat)

                pos_index_list.append(pos_index)
                neg_index_list.append(neg_index)

            pos_embedding = audio_output['embedding'][pos_index_list]
            neg_embedding = audio_output['embedding'][neg_index_list]
            if audio_output['embedding'].shape != pos_embedding.shape:
                continue
            
            labels = labels.to(device, non_blocking=True)
            if isinstance(args.loss_fn, torch.nn.CrossEntropyLoss):
                ce_loss = (1 - loss_weight) * args.loss_fn(audio_output_, torch.argmax(labels.long(), axis=1))
            else:
                ce_loss = (1 - loss_weight) * args.loss_fn(audio_output_, labels)
            tri_loss = loss_weight * args.loss_fun2(audio_output['embedding'], pos_embedding, neg_embedding)
            loss = ce_loss + tri_loss
            print(f'ce_loss: {ce_loss}, tri_loss: {tri_loss}')
            A_loss.append(loss.to('cpu').detach())

            batch_time.update(time.time() - end)
            end = time.time()

        audio_output = torch.cat(A_predictions)
        target = torch.cat(A_targets)
        loss = np.mean(A_loss)
        stats = calculate_stats(audio_output, target)

        exp_dir = args.exp_dir
        if os.path.exists(exp_dir+'/predictions') == False:
            os.mkdir(exp_dir+'/predictions')
            np.savetxt(exp_dir+'/predictions/target.csv', target, delimiter=',')
        np.savetxt(exp_dir+'/predictions/predictions_' + str(epoch) + '.csv', audio_output, delimiter=',')

    return stats, loss

def validate_ensemble(args, epoch):
    exp_dir = args.exp_dir
    target = np.loadtxt(exp_dir+'/predictions/target.csv', delimiter=',')
    if epoch == 1:
        cum_predictions = np.loadtxt(exp_dir + '/predictions/predictions_1.csv', delimiter=',')
    else:
        cum_predictions = np.loadtxt(exp_dir + '/predictions/cum_predictions.csv', delimiter=',') * (epoch - 1)
        predictions = np.loadtxt(exp_dir+'/predictions/predictions_' + str(epoch) + '.csv', delimiter=',')
        cum_predictions = cum_predictions + predictions
        os.remove(exp_dir+'/predictions/predictions_' + str(epoch-1) + '.csv')

    cum_predictions = cum_predictions / epoch
    np.savetxt(exp_dir+'/predictions/cum_predictions.csv', cum_predictions, delimiter=',')

    stats = calculate_stats(cum_predictions, target)
    return stats
