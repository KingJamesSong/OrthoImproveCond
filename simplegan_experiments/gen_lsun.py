"""General-purpose training script for image generation.

This script works for various models (with option '--model': e.g., gan, gan128) and
different datasets (with option '--dataset_mode': e.g., dsprites, celeba).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a GAN model:
        python train.py --dataroot ./datasets/ --name dsprites_orojar --model gan
    Train a GAN128 model:
        python train.py --dataroot ./datasets/CelebA/ --name celeba_dspirtes --model gan128

See options/base_options.py and options/train_options.py for more training options.
"""
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
import torch
from util.util import loop_print
from util.vis_tools import make_mp4_video
import os
import cv2


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    opt = TrainOptions().parse()   # get training options
    opt.log_file = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    loop_print(opt.log_file, 'The number of training images = %d' % dataset_size)
    visualizer = Visualizer(opt)  # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations

    opt.epoch_count=1

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            data = data['img']
            for j in range(data.shape[0]):
                img_1 = data[j, torch.LongTensor([2, 1, 0]), :, :]
                img_1 = img_1.cpu().detach().numpy().transpose((1, 2, 0))
                img_1 = (img_1 + 1) * 127.5
                cv2.imwrite(
                    os.path.join('/nfs/data_chaos/ysong/lsun/images/',
                                 'img_%06d.jpg' % (i * opt.batch_size + j)), img_1)

        loop_print(opt.log_file, 'End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))