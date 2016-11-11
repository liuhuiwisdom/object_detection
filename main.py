#!/usr/bin/env python
import os
import sys
import argparse
import tensorflow as tf

from model import *
from dataset import *
from utils.coco.coco import *

def main(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--phase', default='train', help='Phase: Can be train, val or test')
    parser.add_argument('--component', default='all', help='Component to train: Can be rpn, rcn or all')
    parser.add_argument('--load', action='store_true', default=False, help='Turn on to load the pretrained model')

    parser.add_argument('--mean_file', default='./utils/ilsvrc_2012_mean.npy', help='Dataset image mean: a Numpy array with (Channel, Height, Width) dimensions')
    parser.add_argument('--basic_model', default='vgg16', help='CNN model to use: Can be vgg16, resnet50, resnet101 or resnet152')
    parser.add_argument('--basic_model_file', default='./tfmodels/vgg16.tfmodel', help='Tensorflow model file for the chosen CNN model')
    parser.add_argument('--load_basic_model', action='store_true', default=False, help='Turn on to load the pretrained CNN model')

    parser.add_argument('--dataset', default='pascal', help='Dataset: Can be coco or pascal')

    parser.add_argument('--train_coco_image_dir', default='./train/coco/images/', help='Directory containing the COCO train2014 images')
    parser.add_argument('--train_coco_annotation_file', default='./train/coco/instances_train2014.json', help='JSON file storing the objects for COCO train2014 images') 
    parser.add_argument('--train_coco_data_dir', default='./train/coco/data/', help='Directory to store temporary training data for COCO')

    parser.add_argument('--train_pascal_image_dir', default='./train/pascal/images/', help='Directory containing the PASCAL training images')
    parser.add_argument('--train_pascal_annotation_dir', default='./train/pascal/annotations/', help='Directory containing the PASCAL training annotations') 
    parser.add_argument('--train_pascal_data_dir', default='./train/pascal/data/', help='Directory to store temporary training data for PASCAL')

    parser.add_argument('--val_coco_image_dir', default='./val/coco/images/', help='Directory containing the COCO val2014 images')
    parser.add_argument('--val_coco_annotation_file', default='./val/coco/instances_val2014.json', help='JSON file storing the objects for COCO val2014 images')

    parser.add_argument('--val_pascal_image_dir', default='./val/pascal/images/', help='Directory containing the PASCAL validation images')
    parser.add_argument('--val_pascal_annotation_dir', default='./val/pascal/annotations/', help='Directory containing the PASCAL validation annotations')

    parser.add_argument('--test_image_dir', default='./test/images/', help='Directory containing the testing images')
    parser.add_argument('--test_result_file', default='./test/result.pickle', help='File to store the testing results')
    parser.add_argument('--test_result_dir', default='./test/results/', help='Directory to store the testing results as images')

    parser.add_argument('--save_dir', default='./models/', help='Directory to contain the trained model')
    parser.add_argument('--save_period', type=int, default=1000, help='Period to save the trained model')
    
    parser.add_argument('--solver', default='adam', help='Optimizer to use: Can be adam, momentum, rmsprop or sgd') 
    parser.add_argument('--num_epoch', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum (for some optimizers)') 
    parser.add_argument('--decay', type=float, default=0.9, help='Decay (for some optimizers)') 
    parser.add_argument('--batch_norm', action='store_true', default=False, help='Turn on to use batch normalization')  
   
    parser.add_argument('--num_roi', type=int, default=100, help='Maximum number of RoIs')    
    parser.add_argument('--bbox_per_class', action='store_true', default=False, help='Turn on to do one bounding box regression for each class')    
    parser.add_argument('--rpn_weight', type=float, default=1.0, help='Weight for the loss of RPN')    
    parser.add_argument('--rcn_weight', type=float, default=1.0, help='Weight for the loss of RCN')    
    parser.add_argument('--rpn_reg_weight', type=float, default=10.0, help='Relative weight for bounding box regression loss vs classification loss of RPN')  
    parser.add_argument('--rcn_reg_weight', type=float, default=10.0, help='Relative weight for bounding box regression loss vs classification loss of RCN')   
    parser.add_argument('--class_balancing_factor', type=float, default=0.8, help='Class balancing factor. The larger it is, the more attention the rare classes receive.') 
    parser.add_argument('--prepare_anchor_data', action='store_true', default=False, help='Turn on to prepare useful anchor data for training. Must do this for the first time of training.')

    args = parser.parse_args()

    with tf.Session() as sess:
        # Training phase
        if args.phase == 'train':
            if args.dataset == 'coco':
                train_coco, train_data = prepare_train_coco_data(args)
            else:
                train_data = prepare_train_pascal_data(args)

            model = ObjectDetector(args, 'train')
            sess.run(tf.initialize_all_variables())

            if args.load:
                model.load(sess)
            elif args.load_basic_model:
                model.load2(args.basic_model_file, sess)

            if args.prepare_anchor_data:
                model.prepare_anchor_data(train_data)

            # Train both RPN and RCN
            if args.component == 'all':               
                model.train(sess, train_data)
 
            # Train RPN only
            elif args.component == 'rpn':          
                model.train_rpn(sess, train_data)

            # Train RCN only
            else:
                model.train_rcn(sess, train_data)      
 
        # Validation phase
        elif args.phase == 'val':
            model = ObjectDetector(args, 'val')
            model.load(sess)
            if args.dataset == 'coco':
                val_coco, val_data = prepare_val_coco_data(args)
                model.val_coco(sess, val_coco, val_data)
            else:
                val_pascal, val_data = prepare_val_pascal_data(args)
                model.val_pascal(sess, val_pascal, val_data)

        # Testing phase
        else:
            test_data = prepare_test_data(args)
            model = ObjectDetector(args, 'test')  
            model.load(sess)
            model.test(sess, test_data)

if __name__=="__main__":
     main(sys.argv)

