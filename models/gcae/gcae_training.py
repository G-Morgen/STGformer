import os
import time
import shutil
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader

from utils.train_utils import calc_reg_loss
from models.dc_gcae.dc_gcae_training import adjust_lr
from utils.scoring_utils import dpmm_calc_scores, make_show_high, score_dataset, avg_scores_by_trans, make_show
import cv2
import imageio
import matplotlib.pyplot as plt

def get_img(current_data, meta, img_num, dataset='shanghai'):
    if dataset in ['shanghai', 'HR_shanghai', 'avenue', 'HR_avenue', 'corridor']:
        img_postfix = '.jpg'
    elif dataset in ['ucsdped1', 'ucsdped2']:
        img_postfix = '.tif'
    # print(current_data)
    # print(current_data.tolist()[0])
    scene_id, clip_id, person_id, start_frame = current_data.tolist()[0]
    scene_len, clip_len, img_len = meta[1:]
    img_dir = os.path.join(meta[0], str(scene_id).zfill(scene_len) + '_' + str(clip_id).zfill(clip_len))
    img_ls = []
    for i in range(img_num):
        img_path = os.path.join(img_dir, str(start_frame).zfill(img_len) + img_postfix)
        img_np = cv2.imread(img_path)
        if img_np is None:
            print(img_path)
        img_ls.append(img_np)
        start_frame += 1
    return img_ls

class Trainer:
    def __init__(self, args, model, loss, train_loader, test_loader, test_dataset,
                 optimizer_f=None, scheduler_f=None, fn_suffix=''):
        self.model = model
        self.args = args
        self.arch = getattr(args, 'arch', 'gcae')
        self.seg_len = getattr(args, 'seg_len', 6)
        self.vid_path = getattr(args, 'vid_path')
        self.dataset = getattr(args, 'dataset')
        self.vid_res = getattr(args, 'vid_res')
        # self.batchsize = getattr(args, 'ae_batch_size')
        # self.batchsize = args.batch_size
        self.args.start_epoch = 0
        self.train_loader = train_loader
        self.test_loader = test_loader
        if args.level == 'high':
            batch_size = 1
        else:
            batch_size = args.batch_size
        self.batchsize = batch_size
        self.test_loader_eval = DataLoader(test_dataset, batch_size=batch_size, num_workers=args.num_workers,
                                            shuffle=False, drop_last=False, pin_memory=True)
        self.test_dataset = test_dataset
        self.test_dataset_meta = test_dataset.metadata
        np.save(os.path.join(self.args.ckpt_dir, 'test_dataset_meta'), self.test_dataset_meta)
        self.fn_suffix = fn_suffix  # For checkpoint filename
        # Loss, Optimizer and Scheduler
        self.loss = loss
        if optimizer_f is None:
            self.optimizer = self.get_optimizer()
        else:
            self.optimizer = optimizer_f(self.model.parameters())
        if scheduler_f is None:
            self.scheduler = None
        else:
            self.scheduler = scheduler_f(self.optimizer)

    def get_optimizer(self):
        if self.args.optimizer == 'adam':
            if self.args.lr:
                return optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
            else:
                return optim.Adam(self.model.parameters())
        else:
            return optim.SGD(self.model.parameters(), lr=self.args.lr,)

    def adjust_lr(self, epoch):
        return adjust_lr(self.optimizer, epoch, self.args.lr, self.args.lr_decay, self.scheduler)

    def save_checkpoint(self, epoch, args, is_best=False, filename=None):
        """
        state: {'epoch': cur_epoch + 1, 'state_dict': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict()}
        """
        state = self.gen_checkpoint_state(epoch)
        if filename is None:
            filename = 'checkpoint.pth.tar'

        state['args'] = args

        path_join = os.path.join(self.args.ckpt_dir, filename)
        torch.save(state, path_join)
        if is_best:
            shutil.copy(path_join, os.path.join(self.args.ckpt_dir, 'checkpoint_best.pth.tar'))

    def load_checkpoint(self, filename):
        filename = self.args.ckpt_dir + filename
        try:
            checkpoint = torch.load(filename)
            self.args.start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("Checkpoint loaded successfully from '{}' at (epoch {})\n"
                  .format(self.args.ckpt_dir, checkpoint['epoch']))
        except FileNotFoundError:
            print("No checkpoint exists from '{}'. Skipping...\n".format(self.args.ckpt_dir))
            
    def rescale_low_0(self, input, bbox):
        # print(input.shape)
        img_size = [856, 480]
        anchor_bbox = [85, 145]
        bbox = [torch.min(input[:, 0, 0, :] * img_size[0], 1)[0], torch.min(input[:, 1, 0, :] * img_size[1], 1)[0], 
                torch.max(input[:, 0, 0, :] * img_size[0], 1)[0], torch.max(input[:, 1, 0, :] * img_size[1], 1)[0]]
        # bbox = torch.concat(bbox, axis=1)
        # print(bbox[0].shape, bbox[1].shape)
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        # print(area.shape)
        ratio = area / (anchor_bbox[0] * anchor_bbox[1])
        # print(ratio.shape)
        # output = input / ratio
        return ratio
    
    def rescale_high(self, bbox):
        # input nctpv
        # bbox ntp4
        # img_size = [856, 480]
        anchor_bbox = [85, 145]
        bbox = bbox[:, -1, :, :]    # np4
        area = (bbox[:, :, 2] - bbox[:, :, 0]) * (bbox[:, :, 3] - bbox[:, :, 1])    # np
        # print(area.shape)
        ratio = area / (anchor_bbox[0] * anchor_bbox[1])
        # print(ratio.shape)
        # output = input / ratio
        # print(bbox, area, ratio)
        return ratio
    
    def rescale_low(self, bbox):
        # input nctpv
        # bbox ntp4
        # img_size = [856, 480]
        print(bbox.shape)
        anchor_bbox = [85, 145]
        bbox = bbox[:, -1, :, :]    # np4
        area = (bbox[:, :, 2] - bbox[:, :, 0]) * (bbox[:, :, 3] - bbox[:, :, 1])    # np
        # print(area.shape)
        ratio = area / (anchor_bbox[0] * anchor_bbox[1])
        # print(ratio.shape)
        # output = input / ratio
        # print(bbox, area, ratio)
        return ratio
        
    def train(self, num_epochs=None, log=True, checkpoint_filename=None, args=None):
        time_str = time.strftime("%b%d_%H%M_")
        if checkpoint_filename is None:
            checkpoint_filename = time_str + self.fn_suffix + '_checkpoint.pth.tar'
        if num_epochs is None:  # For manually setting number of epochs, i.e. for fine tuning
            start_epoch = self.args.start_epoch
            num_epochs = self.args.epochs
        else:
            start_epoch = 0

        level = args.level
        weights = (1, 1)
        rescale = True
       
        self.model = self.model.to(args.device)
        for epoch in range(start_epoch, num_epochs):
            self.model.train()
            loss_list = []
            ep_start_time = time.time()
            print("Started epoch {}".format(epoch))
            save_train_img_num = 20
            save_evel_img_num = 20
            train_print_iter = (len(self.train_loader.dataset) // self.batchsize) // save_train_img_num
            test_print_iter = (len(self.test_loader_eval.dataset) // self.batchsize) // save_evel_img_num
            if train_print_iter == 0:
                train_print_iter = 1
            if test_print_iter == 0:
                test_print_iter = 1
            # test_print_iter = 1
            print(len(self.train_loader.dataset), self.batchsize, train_print_iter)
            
            for itern, data_arr in enumerate(self.train_loader):
                data = data_arr[0].to(args.device, non_blocking=True)   # N, C, T, V
                time0 = time.time()
                reco_data, output = self.model(data)
                time1 = time.time()
                # print(time1 - time0)
                # bbox = data_arr[-1]   # N, T, P, 4   
                # print('bbox: ', bbox.shape)
                # print(data_arr[-1].shape)
                # print(type(data_arr[-1]))
                bbox = data_arr[-1].cuda()
                

                if args.arch == 'gtae_pred':
                    output = output[:, :2, ...]
                    # print('output: ', output.shape)
                    if level == 'low':
                        # ratio = self.rescale_low(bbox).cuda().float()
                        # if rescale:
                        #     rescale_ratio = self.rescale_low(output)  
                        #     reco_loss = torch.sum((torch.abs(output - reco_data) / rescale_ratio[:, None, None, None]), axis=(1, 2, 3))
                        # else:
                        reco_loss = torch.sum((torch.abs(output - reco_data)), axis=(1, 2, 3))
                        reco_loss = torch.mean(reco_loss)
                    elif level == 'high':
                        # output nctpv
                        # bbox ntp4
                        reco_loss_temp = torch.sum(torch.abs(output - reco_data), axis=(1, 2, 3, 4))
                        reco_loss = torch.mean(reco_loss_temp)
                    # reco_loss = None

                    if itern % train_print_iter == 0:
                        if level == 'low':
                            img_np_ls = get_img(data_arr[2], self.vid_path['train'], self.seg_len, self.dataset)
                            show_img = make_show(data[:, :2, :, :], reco_data, reco_loss, img_size=self.vid_res, graph_img_size=self.vid_res, img_ls=img_np_ls)
                            save_path = os.path.join(self.args.ckpt_dir, 'epoch{}_train_iter{}.jpg'.format(epoch, itern))
                            cv2.imwrite(save_path, show_img)
                            # imageio.mimsave(save_path, show_img, 'GIF', duration=0.3)
                        elif level == 'high':
                            img_np_ls = get_img(data_arr[2], self.vid_path['train'], self.seg_len, self.dataset)
                            show_img = make_show_high(data[:, :2, ...], reco_data, 0, img_size=self.vid_res, graph_img_size=self.vid_res, img_ls=img_np_ls)
                            save_path = os.path.join(self.args.ckpt_dir, 'epoch{}_train_iter{}.jpg'.format(epoch, itern))
                            cv2.imwrite(save_path, show_img)
                        with open(args.log_path, 'a') as f:
                            f.write('epoch {}, iter {}, loss {}\n'.format(epoch, itern, reco_loss))
                        print('epoch {}, iter {}, loss {}'.format(epoch, itern, reco_loss))    
                            
                else:
                    # print(output.shape, reco_data.shape)
                    reco_loss = self.loss(output, reco_data)
                    if itern % train_print_iter == 0:
                        with open(args.log_path, 'a') as f:
                            f.write('epoch {}, iter {}, loss {}\n'.format(epoch, itern, reco_loss))
                        print('epoch {}, iter {}, loss {}'.format(epoch, itern, reco_loss))

                reg_loss = calc_reg_loss(self.model)
                loss = reco_loss + args.alpha * reg_loss
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                loss_list.append(loss.item())


            print("Epoch {0} done, loss: {1:.7f}, took: {2:.3f}sec".format(epoch, np.mean(loss_list),
                                                                           time.time()-ep_start_time))
            with open(args.log_path, 'a') as f:
                f.write("Epoch {0} done, loss: {1:.7f}, took: {2:.3f}sec\n".format(epoch, np.mean(loss_list),
                                                                           time.time()-ep_start_time))
                
            new_lr = self.adjust_lr(epoch)
            print('lr: {0:.3e}'.format(new_lr))
            with open(args.log_path, 'a') as f:
                f.write('lr: {0:.3e}\n'.format(new_lr))

            self.model.save_checkpoint(epoch, args=args, filename=str(epoch) + '_' + checkpoint_filename)

            if args.new_pipeline:
                # eval step
                self.model.eval()
                eval_scores = []
                eval_scores_rescale = []
                for itern, data_arr in enumerate(self.test_loader_eval):
                    data = data_arr[0].to(args.device, non_blocking=True)
                    reco_data, output = self.model(data)
                    # print(data_arr[-1].shape)
                    # print(type(data_arr[-1]))
                    bbox = data_arr[-1].cuda()
                    
                    if level == 'low':
                        # if rescale:
                        err = torch.sum((torch.abs(output - reco_data)), axis=(1, 2, 3))
                        err_rescale = err
                        
                        # rescale_ratio = self.rescale_low(output)  
                        # err_rescale = torch.sum((torch.abs(output - reco_data) / rescale_ratio[:, None, None, None]), axis=(1, 2, 3))
                        
                                              
                        # err = torch.sum((torch.abs(output - reco_data) / rescale_ratio[:, None, None, None]), axis=(1, 2, 3))
                        # reco_loss = torch.mean(reco_loss)
                    elif level == 'high':
                        ratio = self.rescale_high(bbox).cuda().float()
                        # output nctpv
                        # bbox ntp4
                        err_personwise = torch.sum(torch.abs(output - reco_data), axis=(1, 2, 4))   # np
                        # err_personwise_rescale = err_personwise
                        err_personwise_rescale = err_personwise / ratio
                        err, _ = torch.max(err_personwise, axis=1)
                        err_rescale, _ = torch.max(err_personwise_rescale, axis=1)
                    eval_scores.append(err.detach().cpu().float().numpy())
                    eval_scores_rescale.append(err_rescale.detach().cpu().float().numpy())

                    if itern % test_print_iter == 0:
                        if level == 'low':
                            img_np_ls = get_img(data_arr[2], self.vid_path['test'], self.seg_len, self.dataset)
                            
                            show_img = make_show(data[:, :2, :, :], reco_data, err, img_size=self.vid_res, graph_img_size=self.vid_res, img_ls=img_np_ls)
                            save_path = os.path.join(self.args.ckpt_dir, 'epoch{}_eval_iter{}.jpg'.format(epoch, itern))
                            cv2.imwrite(save_path, show_img)
                            # imageio.mimsave(save_path, show_img, 'GIF', duration=0.3)
                        elif level == 'high':
                            img_np_ls = get_img(data_arr[2], self.vid_path['test'], self.seg_len, self.dataset)
                            show_img = make_show_high(data[:, :2, ...], reco_data, reco_loss, img_size=self.vid_res, graph_img_size=self.vid_res, img_ls=img_np_ls)
                            save_path = os.path.join(self.args.ckpt_dir, 'epoch{}_eval_iter{}.jpg'.format(epoch, itern))
                            cv2.imwrite(save_path, show_img)


                eval_scores = np.concatenate(eval_scores)
                np.save(os.path.join(self.args.ckpt_dir, '{}_reco_loss'.format(epoch)), eval_scores)
                print(eval_scores.shape)
                max_clip = 5 if args.debug else None
                
                dp_auc, dp_shift, dp_sigma, fpr, tpr = score_dataset(eval_scores, self.test_dataset_meta, max_clip=max_clip, level=level, dataset=self.dataset)
                print("Done with {} AuC for {} samples".format(dp_auc, eval_scores.shape[0]))
                with open(args.log_path, 'a') as f:
                    f.write("Done with {} AuC for {} samples\n".format(dp_auc, eval_scores.shape[0]))
                auc_save_path = os.path.join(args.ckpt_dir, 'auc_epoch{}.jpg'.format(epoch))
                # fpr, tpr = plot_material
                plt.figure(figsize=(6,6))
                plt.title('Validation ROC')
                plt.plot(fpr, tpr, 'b', label = 'Val AUC = %0.3f' % dp_auc)
                plt.legend(loc = 'lower right')
                plt.plot([0, 1], [0, 1],'r--')
                plt.xlim([0, 1])
                plt.ylim([0, 1])
                plt.ylabel('True Positive Rate')
                plt.xlabel('False Positive Rate')
                plt.savefig(auc_save_path)
                plt.close()
                
                
                eval_scores_rescale = np.concatenate(eval_scores_rescale)
                np.save(os.path.join(self.args.ckpt_dir, '{}_reco_loss_rescale'.format(epoch)), eval_scores_rescale)
                print(eval_scores_rescale.shape)
                max_clip = 5 if args.debug else None
                dp_auc, dp_shift, dp_sigma, fpr, tpr = score_dataset(eval_scores_rescale, self.test_dataset_meta, max_clip=max_clip, level=level, dataset=self.dataset)
                print("Done with {} AuC for {} samples rescale".format(dp_auc, eval_scores_rescale.shape[0]))
                with open(args.log_path, 'a') as f:
                    f.write("Done with {} AuC for {} samples\n".format(dp_auc, eval_scores_rescale.shape[0]))
                auc_save_path = os.path.join(args.ckpt_dir, 'auc_epoch{}_rescale.jpg'.format(epoch))
                # fpr, tpr = plot_material
                plt.figure(figsize=(6,6))
                plt.title('Validation ROC')
                plt.plot(fpr, tpr, 'b', label = 'Val AUC = %0.3f' % dp_auc)
                plt.legend(loc = 'lower right')
                plt.plot([0, 1], [0, 1],'r--')
                plt.xlim([0, 1])
                plt.ylim([0, 1])
                plt.ylabel('True Positive Rate')
                plt.xlabel('False Positive Rate')
                plt.savefig(auc_save_path)
                plt.close()
            
        exit(-1)
        return checkpoint_filename, np.mean(loss_list)

    def gen_checkpoint_state(self, epoch):
        checkpoint_state = {'epoch': epoch + 1,
                            'state_dict': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict(), }
        if hasattr(self.model, 'num_class'):
            checkpoint_state['n_classes'] = self.model.num_class
        if hasattr(self.model, 'h_dim'):
            checkpoint_state['h_dim'] = self.model.h_dim
        return checkpoint_state

    def test(self, cur_epoch, ret_sfmax=False, log=True, args=None):
        self._test(cur_epoch, self.test_loader, ret_sfmax=ret_sfmax, log=log, args=args)

    def _test(self, cur_epoch, test_loader, ret_sfmax=True, log=True, args=None):
        print("Testing")
        self.model.eval()
        test_loss = 0
        output_arr = []
        for itern, data_arr in enumerate(test_loader):
            # Get Data
            with torch.no_grad():
                data = data_arr[0].to(args.device)
                output = self.model(data)

            if ret_sfmax:
                output_sfmax = output
                output_arr.append(output_sfmax.detach().cpu().numpy())
                del output_sfmax

            loss = self.loss(output, data)
            test_loss += loss.item()

        test_loss /= len(test_loader.dataset)
        print("--> Test set loss {:.7f}".format(test_loss))
        self.model.train()
        if ret_sfmax:
            return output_arr

