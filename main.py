# -*- coding: utf-8 -*-

import argparse
from model import pix2pix
#import tensorflow as tf
import tensorflow.compat.v1 as tf

parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset_name', dest='dataset_name', default='models_img', help='name of the dataset')
parser.add_argument('--epoch', dest='epoch', type=int, default=20000, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='# images in batch')
parser.add_argument('--train_size', dest='train_size', type=int, default=2999, help='# images used to train')
parser.add_argument('--load_size', dest='load_size', type=int, default=512, help='scale images to this size')
parser.add_argument('--fine_size', dest='fine_size', type=int, default=512, help='then crop to this size')
parser.add_argument('--ngf', dest='ngf', type=int, default=64, help='# of gen filters in first conv layer')
parser.add_argument('--ndf', dest='ndf', type=int, default=32, help='# of discri filters in first conv layer')
parser.add_argument('--input_nc', dest='input_nc', type=int, default=3, help='# of input image channels')
parser.add_argument('--output_nc', dest='output_nc', type=int, default=3, help='# of output image channels')
parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='initial learning rate for adam') #0.0002
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--phase', dest='phase', default='test', help='train, test')
parser.add_argument('--save_latest_freq', dest='save_latest_freq', type=int, default=500, help='save the latest model every latest_freq sgd iterations (overwrites the previous latest model)')
parser.add_argument('--print_freq', dest='print_freq', type=int, default=150, help='print the debug information every print_freq iterations')
parser.add_argument('--continue_train', dest='continue_train', type=bool, default=True, help='if continue training, load the latest model: 1: true, 0: false')
parser.add_argument('--serial_batches', dest='serial_batches', type=bool, default=False, help='f 1, takes images in order to make batches, otherwise takes them randomly')
parser.add_argument('--serial_batch_iter', dest='serial_batch_iter', type=bool, default=True, help='iter into serial image list')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('--data_load', dest='data_load', default='./datasets/combined_2/Images/', help='images are stored here')
parser.add_argument('--sample_dir', dest='sample_dir', default='./samples', help='sample are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='./test', help='test sample are saved here')
parser.add_argument('--L1_lambda', dest='L1_lambda', type=float, default=400.0, help='weight on L1 term in objective')
parser.add_argument('--deviation', dest='deviation', type=int, default=3, help='deviation of images')

args = parser.parse_args()
tf.reset_default_graph ()

def main(_):
    with tf.Session() as sess:
        model = pix2pix(sess, image_size=args.fine_size, batch_size=args.batch_size,
                        output_size=args.fine_size,gf_dim=args.ngf, df_dim=args.ndf,
                        L1_lambda=args.L1_lambda, input_c_dim=args.input_nc,
                        output_c_dim=args.output_nc,dataset_name=args.dataset_name,
                        checkpoint_dir=args.checkpoint_dir, sample_dir=args.sample_dir,
                        train_size=args.train_size, data_load=args.data_load, phase=args.phase,
                        save_latest_freq=args.save_latest_freq, print_freq=args.print_freq, deviation=args.deviation)

        if args.phase == 'train':
            model.train(args)
        else:
            model.test(args)

if __name__ == '__main__':
    tf.app.run()
