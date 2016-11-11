import math
import numpy as np
import tensorflow as tf
import time
from tqdm import tqdm

from base_model import *
from bbox import *
from utils.nn import *

class ObjectDetector(BaseModel):         
    def build(self):
        """ Build the model. """
        if self.basic_model=='vgg16':
            self.build_basic_vgg16()

        elif self.basic_model=='resnet50':
            self.build_basic_resnet50()

        elif self.basic_model=='resnet101':
            self.build_basic_resnet101()

        else:
            self.build_basic_resnet152()

        self.build_anchors()
        self.build_rpn()
        self.build_rcn()
        self.build_final()

    def build_basic_vgg16(self):
        """ Build the basic VGG16 net. """
        print("Building the basic VGG16 net...")
        bn = self.batch_norm

        imgs = tf.placeholder(tf.float32, [self.batch_size]+self.img_shape)
        is_train = tf.placeholder(tf.bool)    

        conv1_1_feats = convolution(imgs, 3, 3, 64, 1, 1, 'conv1_1')
        conv1_1_feats = batch_norm(conv1_1_feats, 'bn1_1', is_train, bn, 'relu')
        conv1_2_feats = convolution(conv1_1_feats, 3, 3, 64, 1, 1, 'conv1_2')
        conv1_2_feats = batch_norm(conv1_2_feats, 'bn1_2', is_train, bn, 'relu')
        pool1_feats = max_pool(conv1_2_feats, 2, 2, 2, 2, 'pool1')

        conv2_1_feats = convolution(pool1_feats, 3, 3, 128, 1, 1, 'conv2_1')
        conv2_1_feats = batch_norm(conv2_1_feats, 'bn2_1', is_train, bn, 'relu')
        conv2_2_feats = convolution(conv2_1_feats, 3, 3, 128, 1, 1, 'conv2_2')
        conv2_2_feats = batch_norm(conv2_2_feats, 'bn2_2', is_train, bn, 'relu')
        pool2_feats = max_pool(conv2_2_feats, 2, 2, 2, 2, 'pool2')

        conv3_1_feats = convolution(pool2_feats, 3, 3, 256, 1, 1, 'conv3_1')
        conv3_1_feats = batch_norm(conv3_1_feats, 'bn3_1', is_train, bn, 'relu')
        conv3_2_feats = convolution(conv3_1_feats, 3, 3, 256, 1, 1, 'conv3_2')
        conv3_2_feats = batch_norm(conv3_2_feats, 'bn3_2', is_train, bn, 'relu')
        conv3_3_feats = convolution(conv3_2_feats, 3, 3, 256, 1, 1, 'conv3_3')
        conv3_3_feats = batch_norm(conv3_3_feats, 'bn3_3', is_train, bn, 'relu')
        pool3_feats = max_pool(conv3_3_feats, 2, 2, 2, 2, 'pool3')

        conv4_1_feats = convolution(pool3_feats, 3, 3, 512, 1, 1, 'conv4_1')
        conv4_1_feats = batch_norm(conv4_1_feats, 'bn4_1', is_train, bn, 'relu')
        conv4_2_feats = convolution(conv4_1_feats, 3, 3, 512, 1, 1, 'conv4_2')
        conv4_2_feats = batch_norm(conv4_2_feats, 'bn4_2', is_train, bn, 'relu')
        conv4_3_feats = convolution(conv4_2_feats, 3, 3, 512, 1, 1, 'conv4_3')
        conv4_3_feats = batch_norm(conv4_3_feats, 'bn4_3', is_train, bn, 'relu')
        pool4_feats = max_pool(conv4_3_feats, 2, 2, 2, 2, 'pool4')

        conv5_1_feats = convolution(pool4_feats, 3, 3, 512, 1, 1, 'conv5_1')
        conv5_1_feats = batch_norm(conv5_1_feats, 'bn5_1', is_train, bn, 'relu')
        conv5_2_feats = convolution(conv5_1_feats, 3, 3, 512, 1, 1, 'conv5_2')
        conv5_2_feats = batch_norm(conv5_2_feats, 'bn5_2', is_train, bn, 'relu')
        conv5_3_feats = convolution(conv5_2_feats, 3, 3, 512, 1, 1, 'conv5_3')
        conv5_3_feats = batch_norm(conv5_3_feats, 'bn5_3', is_train, bn, 'relu')

        self.conv_feats = conv5_3_feats
        self.conv_feat_shape = [40, 40, 512]

        self.roi_warped_feat_shape = [16, 16, 512]
        self.roi_pooled_feat_shape = [8, 8, 512]

        self.imgs = imgs
        self.is_train = is_train
        print("Basic VGG16 net built.")

    def basic_block(self, input_feats, name1, name2, is_train, bn, c, s=2):
        """ A basic block of ResNets. """
        branch1_feats = convolution_no_bias(input_feats, 1, 1, 4*c, s, s, name1+'_branch1')
        branch1_feats = batch_norm(branch1_feats, name2+'_branch1', is_train, bn, None)

        branch2a_feats = convolution_no_bias(input_feats, 1, 1, c, s, s, name1+'_branch2a')
        branch2a_feats = batch_norm(branch2a_feats, name2+'_branch2a', is_train, bn, 'relu')

        branch2b_feats = convolution_no_bias(branch2a_feats, 3, 3, c, 1, 1, name1+'_branch2b')
        branch2b_feats = batch_norm(branch2b_feats, name2+'_branch2b', is_train, bn, 'relu')

        branch2c_feats = convolution_no_bias(branch2b_feats, 1, 1, 4*c, 1, 1, name1+'_branch2c')
        branch2c_feats = batch_norm(branch2c_feats, name2+'_branch2c', is_train, bn, None)

        output_feats = branch1_feats + branch2c_feats
        output_feats = nonlinear(output_feats, 'relu')
        return output_feats

    def basic_block2(self, input_feats, name1, name2, is_train, bn, c):
        """ Another basic block of ResNets. """
        branch2a_feats = convolution_no_bias(input_feats, 1, 1, c, 1, 1, name1+'_branch2a')
        branch2a_feats = batch_norm(branch2a_feats, name2+'_branch2a', is_train, bn, 'relu')

        branch2b_feats = convolution_no_bias(branch2a_feats, 3, 3, c, 1, 1, name1+'_branch2b')
        branch2b_feats = batch_norm(branch2b_feats, name2+'_branch2b', is_train, bn, 'relu')

        branch2c_feats = convolution_no_bias(branch2b_feats, 1, 1, 4*c, 1, 1, name1+'_branch2c')
        branch2c_feats = batch_norm(branch2c_feats, name2+'_branch2c', is_train, bn, None)

        output_feats = input_feats + branch2c_feats
        output_feats = nonlinear(output_feats, 'relu')
        return output_feats

    def build_basic_resnet50(self):
        """ Build the basic ResNet50 net. """
        print("Building the basic ResNet50 net...")
        bn = self.batch_norm

        imgs = tf.placeholder(tf.float32, [self.batch_size]+self.img_shape)
        is_train = tf.placeholder(tf.bool)     

        conv1_feats = convolution(imgs, 7, 7, 64, 2, 2, 'conv1')
        conv1_feats = batch_norm(conv1_feats, 'bn_conv1', is_train, bn, 'relu')
        pool1_feats = max_pool(conv1_feats, 3, 3, 2, 2, 'pool1')

        res2a_feats = self.basic_block(pool1_feats, 'res2a', 'bn2a', is_train, bn, 64, 1)
        res2b_feats = self.basic_block2(res2a_feats, 'res2b', 'bn2b', is_train, bn, 64)
        res2c_feats = self.basic_block2(res2b_feats, 'res2c', 'bn2c', is_train, bn, 64)
  
        res3a_feats = self.basic_block(res2c_feats, 'res3a', 'bn3a', is_train, bn, 128)
        res3b_feats = self.basic_block2(res3a_feats, 'res3b', 'bn3b', is_train, bn, 128)
        res3c_feats = self.basic_block2(res3b_feats, 'res3c', 'bn3c', is_train, bn, 128)
        res3d_feats = self.basic_block2(res3c_feats, 'res3d', 'bn3d', is_train, bn, 128)

        res4a_feats = self.basic_block(res3d_feats, 'res4a', 'bn4a', is_train, bn, 256)
        res4b_feats = self.basic_block2(res4a_feats, 'res4b', 'bn4b', is_train, bn, 256)
        res4c_feats = self.basic_block2(res4b_feats, 'res4c', 'bn4c', is_train, bn, 256)
        res4d_feats = self.basic_block2(res4c_feats, 'res4d', 'bn4d', is_train, bn, 256)
        res4e_feats = self.basic_block2(res4d_feats, 'res4e', 'bn4e', is_train, bn, 256)
        res4f_feats = self.basic_block2(res4e_feats, 'res4f', 'bn4f', is_train, bn, 256)

        res5a_feats = self.basic_block(res4f_feats, 'res5a', 'bn5a', is_train, bn, 512)
        res5b_feats = self.basic_block2(res5a_feats, 'res5b', 'bn5b', is_train, bn, 512)
        res5c_feats = self.basic_block2(res5b_feats, 'res5c', 'bn5c', is_train, bn, 512)

        self.conv_feats = res5c_feats
        self.conv_feat_shape = [20, 20, 2048]

        self.roi_warped_feat_shape = [10, 10, 2048]
        self.roi_pooled_feat_shape = [5, 5, 2048]

        self.imgs = imgs
        self.is_train = is_train
        print("Basic ResNet50 net built.")

    def build_basic_resnet101(self):
        """ Build the basic ResNet101 net. """
        print("Building the basic ResNet101 net...")
        bn = self.batch_norm

        imgs = tf.placeholder(tf.float32, [self.batch_size]+self.img_shape)
        is_train = tf.placeholder(tf.bool)  

        conv1_feats = convolution(imgs, 7, 7, 64, 2, 2, 'conv1')
        conv1_feats = batch_norm(conv1_feats, 'bn_conv1', is_train, bn, 'relu')
        pool1_feats = max_pool(conv1_feats, 3, 3, 2, 2, 'pool1')

        res2a_feats = self.basic_block(pool1_feats, 'res2a', 'bn2a', is_train, bn, 64, 1)
        res2b_feats = self.basic_block2(res2a_feats, 'res2b', 'bn2b', is_train, bn, 64)
        res2c_feats = self.basic_block2(res2b_feats, 'res2c', 'bn2c', is_train, bn, 64)
  
        res3a_feats = self.basic_block(res2c_feats, 'res3a', 'bn3a', is_train, bn, 128)       
        temp = res3a_feats
        for i in range(1, 4):
            temp = self.basic_block2(temp, 'res3b'+str(i), 'bn3b'+str(i), is_train, bn, 128)
        res3b3_feats = temp
 
        res4a_feats = self.basic_block(res3b3_feats, 'res4a', 'bn4a', is_train, bn, 256)
        temp = res4a_feats
        for i in range(1, 23):
            temp = self.basic_block2(temp, 'res4b'+str(i), 'bn4b'+str(i), is_train, bn, 256)
        res4b22_feats = temp

        res5a_feats = self.basic_block(res4b22_feats, 'res5a', 'bn5a', is_train, bn, 512)
        res5b_feats = self.basic_block2(res5a_feats, 'res5b', 'bn5b', is_train, bn, 512)
        res5c_feats = self.basic_block2(res5b_feats, 'res5c', 'bn5c', is_train, bn, 512)

        self.conv_feats = res5c_feats
        self.conv_feat_shape = [20, 20, 2048]

        self.roi_warped_feat_shape = [10, 10, 2048]
        self.roi_pooled_feat_shape = [5, 5, 2048]

        self.imgs = imgs
        self.is_train = is_train
        print("Basic ResNet101 net built.")

    def build_basic_resnet152(self):
        """ Build the basic ResNet152 net. """
        print("Building the basic ResNet152 net...")
        bn = self.batch_norm

        imgs = tf.placeholder(tf.float32, [self.batch_size]+self.img_shape)
        is_train = tf.placeholder(tf.bool)

        conv1_feats = convolution(imgs, 7, 7, 64, 2, 2, 'conv1')
        conv1_feats = batch_norm(conv1_feats, 'bn_conv1', is_train, bn, 'relu')
        pool1_feats = max_pool(conv1_feats, 3, 3, 2, 2, 'pool1')

        res2a_feats = self.basic_block(pool1_feats, 'res2a', 'bn2a', is_train, bn, 64, 1)
        res2b_feats = self.basic_block2(res2a_feats, 'res2b', 'bn2b', is_train, bn, 64)
        res2c_feats = self.basic_block2(res2b_feats, 'res2c', 'bn2c', is_train, bn, 64)
  
        res3a_feats = self.basic_block(res2c_feats, 'res3a', 'bn3a', is_train, bn, 128)       
        temp = res3a_feats
        for i in range(1, 8):
            temp = self.basic_block2(temp, 'res3b'+str(i), 'bn3b'+str(i), is_train, bn, 128)
        res3b7_feats = temp
 
        res4a_feats = self.basic_block(res3b7_feats, 'res4a', 'bn4a', is_train, bn, 256)
        temp = res4a_feats
        for i in range(1, 36):
            temp = self.basic_block2(temp, 'res4b'+str(i), 'bn4b'+str(i), is_train, bn, 256)
        res4b35_feats = temp

        res5a_feats = self.basic_block(res4b35_feats, 'res5a', 'bn5a', is_train, bn, 512)
        res5b_feats = self.basic_block2(res5a_feats, 'res5b', 'bn5b', is_train, bn, 512)
        res5c_feats = self.basic_block2(res5b_feats, 'res5c', 'bn5c', is_train, bn, 512)

        self.conv_feats = res5c_feats
        self.conv_feat_shape = [20, 20, 2048]

        self.roi_warped_feat_shape = [10, 10, 2048]
        self.roi_pooled_feat_shape = [5, 5, 2048]

        self.imgs = imgs
        self.is_train = is_train
        print("Basic ResNet152 net built.")

    def build_anchors(self):
        """ Build the anchors and their parents which include the surrounding contexts. """
        print("Building the anchors...")
        img_shape = np.array(self.img_shape[:2], np.int32)

        # Build small anchors
        current_feat_shape = np.array(self.conv_feat_shape[:2], np.int32) 
        for i in range(3):
            for j in range(3):
                num_anchor, anchors, anchor_is_untruncated, num_untruncated_anchor, parent_anchors, parent_anchor_is_untruncated, num_untruncated_parent_anchor = generate_anchors(img_shape, current_feat_shape, self.anchor_scales[i], self.anchor_ratios[j])
                if i==0 and j==0:
                    self.num_anchor = num_anchor
                    self.anchors = anchors
                    self.anchor_is_untruncated = anchor_is_untruncated
                    self.num_untruncated_anchor = num_untruncated_anchor
                    self.parent_anchors = parent_anchors
                    self.parent_anchor_is_untruncated = parent_anchor_is_untruncated
                    self.num_untruncated_parent_anchor = num_untruncated_parent_anchor
                else:
                    self.num_anchor = np.concatenate((self.num_anchor, num_anchor))
                    self.anchors = np.concatenate((self.anchors, anchors))
                    self.anchor_is_untruncated = np.concatenate((self.anchor_is_untruncated, anchor_is_untruncated))
                    self.num_untruncated_anchor = np.concatenate((self.num_untruncated_anchor, num_untruncated_anchor))
                    self.parent_anchors = np.concatenate((self.parent_anchors, parent_anchors))
                    self.parent_anchor_is_untruncated = np.concatenate((self.parent_anchor_is_untruncated, parent_anchor_is_untruncated))
                    self.num_untruncated_parent_anchor = np.concatenate((self.num_untruncated_parent_anchor, num_untruncated_parent_anchor))

        # Build large anchors
        current_feat_shape = (current_feat_shape/2).astype(np.int32) 
        for i in range(3, 6):
            for j in range(3):
                num_anchor, anchors, anchor_is_untruncated, num_untruncated_anchor, parent_anchors, parent_anchor_is_untruncated, num_untruncated_parent_anchor = generate_anchors(img_shape, current_feat_shape, self.anchor_scales[i], self.anchor_ratios[j])
                self.num_anchor = np.concatenate((self.num_anchor, num_anchor))
                self.anchors = np.concatenate((self.anchors, anchors))
                self.anchor_is_untruncated = np.concatenate((self.anchor_is_untruncated, anchor_is_untruncated))
                self.num_untruncated_anchor = np.concatenate((self.num_untruncated_anchor, num_untruncated_anchor))
                self.parent_anchors = np.concatenate((self.parent_anchors, parent_anchors))
                self.parent_anchor_is_untruncated = np.concatenate((self.parent_anchor_is_untruncated, parent_anchor_is_untruncated))
                self.num_untruncated_parent_anchor = np.concatenate((self.num_untruncated_parent_anchor, num_untruncated_parent_anchor))
 
        self.total_num_anchor = np.sum(self.num_anchor)
        self.total_num_untruncated_anchor = np.sum(self.num_untruncated_anchor)
        self.total_num_truncated_anchor = self.total_num_anchor - self.total_num_untruncated_anchor

        # Show the statistics of anchors
        for i in range(self.num_anchor_type):
            print("Anchor type [%d, %d]: %d untruncated, %d truncated" %(self.anchor_shapes[i][0], self.anchor_shapes[i][1], self.num_untruncated_anchor[i], self.num_anchor[i]-self.num_untruncated_anchor[i]))

        print("Anchors built.")

    def build_rpn(self):
        """ Build the RPN. """
        print("Building the RPN...")
        params = self.params
        bn = self.batch_norm
        is_train = self.is_train

        feats = tf.placeholder(tf.float32, [self.batch_size]+self.conv_feat_shape) 
        gt_anchor_labels = tf.placeholder(tf.int32, [self.batch_size, self.total_num_anchor])
        gt_anchor_regs = tf.placeholder(tf.float32, [self.batch_size, self.total_num_anchor, 4])
        anchor_masks = tf.placeholder(tf.float32, [self.batch_size, self.total_num_anchor])
        anchor_weights = tf.placeholder(tf.float32, [self.batch_size, self.total_num_anchor])
        anchor_reg_masks = tf.placeholder(tf.float32, [self.batch_size, self.total_num_anchor])

        self.feats = feats
        self.gt_anchor_labels = gt_anchor_labels
        self.gt_anchor_regs = gt_anchor_regs
        self.anchor_masks = anchor_masks
        self.anchor_weights = anchor_weights
        self.anchor_reg_masks = anchor_reg_masks

        # Compute the RoI proposals
        all_rpn_logits = []
        all_rpn_regs = []

        current_feats = feats

        if self.basic_model == 'vgg16':
            kernel_sizes = [10, 10]
        else:
            kernel_sizes = [5, 5]

        for i in range(2):           
            label_i = '_'+str(i)
            rpn1 = convolution(current_feats, kernel_sizes[0], kernel_sizes[1], 512, 1, 1, 'rpn1'+label_i, group_id=1)
            rpn1 = nonlinear(rpn1, 'relu')
            rpn1 = dropout(rpn1, 0.5, is_train)

            for j in range(9):
                label_ij = str(i)+'_'+str(j)

                rpn_logits = convolution(rpn1, 1, 1, 2, 1, 1, 'rpn_logits'+label_ij, group_id=1)
                rpn_logits = tf.reshape(rpn_logits, [self.batch_size, -1, 2])
                all_rpn_logits.append(rpn_logits)

                rpn_regs = convolution(rpn1, 1, 1, 4, 1, 1, 'rpn_regs'+label_ij, group_id=1)
                rpn_regs = tf.clip_by_value(rpn_regs, -0.2, 0.2)
                rpn_regs = tf.reshape(rpn_regs, [self.batch_size, -1, 4])
                all_rpn_regs.append(rpn_regs)

            if i<1:
                current_feats = max_pool(current_feats, 2, 2, 2, 2, 'rpn_pool'+label_i)

        all_rpn_logits = tf.concat(1, all_rpn_logits)
        all_rpn_regs = tf.concat(1, all_rpn_regs)

        all_rpn_logits = tf.reshape(all_rpn_logits, [-1, 2])
        all_rpn_regs = tf.reshape(all_rpn_regs, [-1, 4])

        # Compute the loss function
        gt_anchor_labels = tf.reshape(gt_anchor_labels, [-1])       
        gt_anchor_regs = tf.reshape(gt_anchor_regs, [-1, 4])
        anchor_masks = tf.reshape(anchor_masks, [-1])
        anchor_weights = tf.reshape(anchor_weights, [-1])
        anchor_reg_masks = tf.reshape(anchor_reg_masks, [-1])

        loss0 = tf.nn.sparse_softmax_cross_entropy_with_logits(all_rpn_logits, gt_anchor_labels) * anchor_masks
        loss0 = tf.reduce_sum(loss0 * anchor_weights) / tf.reduce_sum(anchor_weights)

        w = self.l2_loss(all_rpn_regs, gt_anchor_regs) * anchor_reg_masks
        z = tf.reduce_sum(anchor_reg_masks)
        loss0 = tf.cond(tf.less(0.0, z), lambda: loss0 + params.rpn_reg_weight * tf.reduce_sum(w) / z, lambda: loss0)

        loss1 = params.weight_decay * tf.add_n(tf.get_collection('l2_1'))
        loss = loss0 + loss1

        # Build the optimizer
        if params.solver == 'adam':
            solver = tf.train.AdamOptimizer(params.learning_rate)
        elif params.solver == 'momentum':
            solver = tf.train.MomentumOptimizer(params.learning_rate, params.momentum)
        elif params.solver == 'rmsprop':
            solver = tf.train.RMSPropOptimizer(params.learning_rate, params.decay, params.momentum)
        else:
            solver = tf.train.GradientDescentOptimizer(params.learning_rate)

        opt_op = solver.minimize(loss, global_step=self.global_step)

        rpn_probs = tf.nn.softmax(all_rpn_logits)
        rpn_scores = tf.squeeze(tf.slice(rpn_probs, [0, 1], [-1, 1]))
        rpn_scores = tf.reshape(rpn_scores, [self.batch_size, self.total_num_anchor])
        rpn_regs = tf.reshape(all_rpn_regs, [self.batch_size, self.total_num_anchor, 4])                     

        self.rpn_loss = loss
        self.rpn_loss0 = loss0
        self.rpn_loss1 = loss1
        self.rpn_opt_op = opt_op

        self.rpn_scores = rpn_scores
        self.rpn_regs = rpn_regs
        print("RPN built.")

    def build_rcn(self):
        """ Build the RCN. """
        print("Building the RCN...")
        params = self.params
        num_roi = self.num_roi
        is_train = self.is_train
        bn = self.batch_norm

        roi_warped_feats = tf.placeholder(tf.float32, [self.batch_size, num_roi]+self.roi_warped_feat_shape)  
        gt_roi_classes = tf.placeholder(tf.int32, [self.batch_size, num_roi]) 
        gt_roi_regs = tf.placeholder(tf.float32, [self.batch_size, num_roi, 4]) 
        roi_masks = tf.placeholder(tf.float32, [self.batch_size, num_roi]) 
        roi_weights = tf.placeholder(tf.float32, [self.batch_size, num_roi]) 
        roi_reg_masks = tf.placeholder(tf.float32, [self.batch_size, num_roi]) 

        self.roi_warped_feats = roi_warped_feats
        self.gt_roi_classes = gt_roi_classes
        self.gt_roi_regs = gt_roi_regs
        self.roi_masks = roi_masks
        self.roi_weights = roi_weights
        self.roi_reg_masks = roi_reg_masks
        
        # Get the RoI pooled feats
        roi_warped_feats = tf.reshape(roi_warped_feats, [self.batch_size*num_roi]+self.roi_warped_feat_shape)
        roi_pooled_feats = max_pool(roi_warped_feats, 2, 2, 2, 2, 'roi_pool')
        roi_pooled_feats = tf.reshape(roi_pooled_feats, [self.batch_size*num_roi, -1])

        # Compute the RoI classification results
        fc6_feats = fully_connected(roi_pooled_feats, 4096, 'rcn_fc6', group_id=2)
        fc6_feats = nonlinear(fc6_feats, 'relu')
        fc6_feats = dropout(fc6_feats, 0.5, is_train)

        fc7_feats = fully_connected(fc6_feats, 4096, 'rcn_fc7', group_id=2)
        fc7_feats = nonlinear(fc7_feats, 'relu')
        fc7_feats = dropout(fc7_feats, 0.5, is_train)

        logits = fully_connected(fc7_feats, self.num_class, 'rcn_logits', group_id=2)

        gt_roi_classes = tf.reshape(gt_roi_classes, [-1])
        gt_roi_regs = tf.reshape(gt_roi_regs, [-1, 4])
        roi_masks = tf.reshape(roi_masks, [-1])
        roi_weights = tf.reshape(roi_weights, [-1])
        roi_reg_masks = tf.reshape(roi_reg_masks, [-1])

        if self.bbox_per_class:
            regs = fully_connected(fc7_feats, 4*self.num_class, 'rcn_reg', group_id=2)
            regs = tf.clip_by_value(regs, -0.2, 0.2)
            useful_regs = []
            for i in range(self.batch_size*num_roi):
                useful_regs.append(tf.squeeze(tf.slice(regs, [i, 4*gt_roi_classes[i]], [1, 4])))
            useful_regs = tf.pack(useful_regs) 
        else:
            regs = fully_connected(fc7_feats, 4, 'rcn_reg', group_id=2)
            regs = tf.clip_by_value(regs, -0.2, 0.2)
            useful_regs = regs

        # Compute the loss function
        loss0 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, gt_roi_classes) * roi_masks
        loss0 = tf.reduce_sum(loss0 * roi_weights) / tf.reduce_sum(roi_weights)

        w = self.l2_loss(useful_regs, gt_roi_regs) * roi_reg_masks
        z = tf.reduce_sum(roi_reg_masks)
        loss0 = tf.cond(tf.less(0.0, z), lambda: loss0 + params.rcn_reg_weight * tf.reduce_sum(w) / z, lambda: loss0)

        loss1 = params.weight_decay * tf.add_n(tf.get_collection('l2_2'))
        loss = loss0 + loss1

        # Build the optimizer
        if params.solver == 'adam':
            solver = tf.train.AdamOptimizer(params.learning_rate)
        elif params.solver == 'momentum':
            solver = tf.train.MomentumOptimizer(params.learning_rate, params.momentum)
        elif params.solver == 'rmsprop':
            solver = tf.train.RMSPropOptimizer(params.learning_rate, params.decay, params.momentum)
        else:
            solver = tf.train.GradientDescentOptimizer(params.learning_rate)

        opt_op = solver.minimize(loss, global_step=self.global_step)

        probs = tf.nn.softmax(logits)
        classes = tf.argmax(probs, 1)
        scores = tf.reduce_max(probs, 1) 
        scores = scores * roi_masks

        res_classes = tf.reshape(classes, [self.batch_size, num_roi])
        res_scores = tf.reshape(scores, [self.batch_size, num_roi])

        if self.bbox_per_class:
            res_regs = []
            for i in range(self.batch_size*num_roi):
                res_regs.append(tf.squeeze(tf.slice(regs, [i, 4*classes[i]], [1, 4])))
            res_regs = tf.pack(res_regs) 
        else:
            res_regs = regs
        res_regs = tf.reshape(res_regs, [self.batch_size, num_roi, 4])

        self.rcn_loss = loss
        self.rcn_loss0 = loss0
        self.rcn_loss1 = loss1
        self.rcn_opt_op = opt_op

        self.res_classes = res_classes
        self.res_scores = res_scores
        self.res_regs = res_regs
  
        print("RCN built.")

    def build_final(self):
        """ Build the global loss function and its optimizer. """
        params = self.params

        # Compute the global loss function
        loss0 = params.rpn_weight * self.rpn_loss0 + params.rcn_weight * self.rcn_loss0
        loss1 = params.weight_decay * (tf.add_n(tf.get_collection('l2_1')) + tf.add_n(tf.get_collection('l2_2')))
        loss = loss0 + loss1

        # Build the optimizer
        if params.solver == 'adam':
            solver = tf.train.AdamOptimizer(params.learning_rate)
        elif params.solver == 'momentum':
            solver = tf.train.MomentumOptimizer(params.learning_rate, params.momentum)
        elif params.solver == 'rmsprop':
            solver = tf.train.RMSPropOptimizer(params.learning_rate, params.decay, params.momentum)
        else:
            solver = tf.train.GradientDescentOptimizer(params.learning_rate)

        opt_op = solver.minimize(loss, global_step=self.global_step)

        self.loss = loss
        self.loss0 = loss0
        self.loss1 = loss1
        self.opt_op = opt_op

    def l2_loss(self, s, t):
        """ L2 loss function. """
        d = s - t
        x = d * d
        loss = tf.reduce_sum(x, 1)
        return loss

    def smooth_l1_loss(self, s, t): 
        """ Smooth L1 loss function. """
        d = s - t
        x = 0.5 * d * d
        y = tf.nn.relu(d-1) + tf.nn.relu(-d-1)
        y = 0.5 * y * y
        z = x - y
        loss = tf.reduce_sum(z, 1)
        return loss

    def get_roi_feats(self, feats, rois):
        """ Get the RoI warped feats for a batch. """
        roi_warped_feats = []
        for i in range(self.batch_size):
            current_feats = feats[i]
            current_rois = rois[i]
            roi_warped_feats.append(self.roi_warp(current_feats, current_rois))
        roi_warped_feats = np.array(roi_warped_feats)
        return roi_warped_feats

    def roi_warp(self, feats, rois):  
        """ Apply the RoI warping layer. """
        ch, cw, c = self.conv_feat_shape
        th, tw, c = self.roi_warped_feat_shape
        num_roi = self.num_roi
        warped_feats = []

        for k in range(num_roi):
            y, x, h, w = rois[k, 0], rois[k, 1], rois[k, 2], rois[k, 3] 

            j = np.array(list(range(h)), np.float32)
            i = np.array(list(range(w)), np.float32)
            tj = np.array(list(range(th)), np.float32)
            ti = np.array(list(range(tw)), np.float32)

            j = np.expand_dims(np.expand_dims(np.expand_dims(j, 1), 2), 3)
            i = np.expand_dims(np.expand_dims(np.expand_dims(i, 0), 2), 3)
            tj = np.expand_dims(np.expand_dims(np.expand_dims(tj, 1), 0), 1)
            ti = np.expand_dims(np.expand_dims(np.expand_dims(ti, 0), 0), 1)

            j = np.tile(j, (1, w, th, tw)) 
            i = np.tile(i, (h, 1, th, tw)) 
            tj = np.tile(tj, (h, w, 1, tw)) 
            ti = np.tile(ti, (h, w, th, 1)) 

            b = tj * h * 1.0 / th - j
            a = ti * w * 1.0 / tw - i

            b = np.maximum(np.zeros_like(b), 1 - np.absolute(b))
            a = np.maximum(np.zeros_like(a), 1 - np.absolute(a))

            G = b * a
            G = G.reshape((h*w, th*tw))

            sliced_feat = feats[y:y+h, x:x+w, :]
            sliced_feat = sliced_feat.swapaxes(0, 1)
            sliced_feat = sliced_feat.swapaxes(0, 2)
            sliced_feat = sliced_feat.reshape((-1, h*w))

            warped_feat = np.matmul(sliced_feat, G)
            warped_feat = warped_feat.reshape((-1, th, tw))
            warped_feat = warped_feat.swapaxes(0, 1)
            warped_feat = warped_feat.swapaxes(1, 2)

            warped_feats.append(warped_feat)

        warped_feats = np.array(warped_feats)
        return warped_feats

    def get_feed_dict_for_rpn(self, batch, is_train, feats):
        """ Get the feed dictionary for RPN. """
        # Training phase
        if is_train:
            _, anchor_files = batch
            gt_anchor_labels, gt_anchor_regs, anchor_masks, anchor_weights, anchor_reg_masks = self.process_anchor_data(anchor_files)                 
            return {self.feats: feats, self.gt_anchor_labels: gt_anchor_labels, self.gt_anchor_regs: gt_anchor_regs, self.anchor_masks: anchor_masks, self.anchor_weights: anchor_weights, self.anchor_reg_masks: anchor_reg_masks, self.is_train: is_train}

        # Validation or testing phase
        else:
            return {self.feats: feats, self.is_train: is_train}

    def get_feed_dict_for_rcn(self, batch, is_train, feats, rois=None, masks=None):
        """ Get the feed dictionary for RCN. """
        # Training phase
        if is_train:
            _, anchor_files = batch
            rois, gt_roi_classes, gt_roi_regs, roi_masks, roi_weights, roi_reg_masks = self.process_roi_data(anchor_files)
            rois = rois.reshape((-1, 4))
            rois = convert_bbox(rois, self.img_shape[:2], self.conv_feat_shape[:2])
            rois = rois.reshape((self.batch_size, self.num_roi, 4))
            roi_warped_feats = self.get_roi_feats(feats, rois)
            return {self.roi_warped_feats: roi_warped_feats, self.gt_roi_classes: gt_roi_classes, self.gt_roi_regs: gt_roi_regs, self.roi_masks: roi_masks, self.roi_weights: roi_weights, self.roi_reg_masks: roi_reg_masks, self.is_train: is_train}

        # Validation or testing phase
        else:
            rois = rois.reshape((-1, 4))
            rois = convert_bbox(rois, self.img_shape[:2], self.conv_feat_shape[:2])
            rois = rois.reshape((self.batch_size, self.num_roi, 4))
            roi_warped_feats = self.get_roi_feats(feats, rois)
            return {self.roi_warped_feats: roi_warped_feats, self.roi_masks: masks, self.is_train: is_train}

    def get_feed_dict_for_all(self, batch, is_train, feats=None):
        """ Get the feed dictionary for both RPN and RCN. """
        # Training phase
        if is_train:
            _, anchor_files = batch
            gt_anchor_labels, gt_anchor_regs, anchor_masks, anchor_weights, anchor_reg_masks = self.process_anchor_data(anchor_files)
            rois, gt_roi_classes, gt_roi_regs, roi_masks, roi_weights, roi_reg_masks = self.process_roi_data(anchor_files)
            rois = rois.reshape((-1, 4))
            rois = convert_bbox(rois, self.img_shape[:2], self.conv_feat_shape[:2])
            rois = rois.reshape((self.batch_size, self.num_roi, 4))
            roi_warped_feats = self.get_roi_feats(feats, rois)
            return {self.feats: feats, self.gt_anchor_labels: gt_anchor_labels, self.gt_anchor_regs: gt_anchor_regs, self.anchor_masks: anchor_masks, self.anchor_weights: anchor_weights, self.anchor_reg_masks: anchor_reg_masks, self.roi_warped_feats: roi_warped_feats, self.gt_roi_classes: gt_roi_classes, self.gt_roi_regs: gt_roi_regs, self.roi_masks: roi_masks, self.roi_weights: roi_weights, self.roi_reg_masks: roi_reg_masks, self.is_train: is_train}

        # Validation or testing phase (not used at this moment)
        else:  
            img_files = batch  
            imgs = self.img_loader.load_imgs(img_files)
            return {self.imgs: imgs, self.is_train: is_train}

    def process_anchor_data(self, anchor_files): 
        """ Prepare the anchor data for training RPN. """
        gt_anchor_labels = []
        gt_anchor_regs = []
        anchor_masks = []
        anchor_weights = []
        anchor_reg_masks = []
        t = self.num_anchor_type

        for i in range(self.batch_size):
            anchor_data = np.load(anchor_files[i])
            labels = anchor_data['labels']
            regs = anchor_data['regs']
            ious = anchor_data['ious']
            ioas = anchor_data['ioas']
            iogs = anchor_data['iogs']

            start = 0
            masks = np.array([])
            weights = np.array([])
            reg_masks = np.array([])
            for j in range(t):
                end = start + self.num_anchor[j]
                current_labels = labels[start:end]
                current_ious = ious[start:end]
                current_ioas = ioas[start:end]
                current_iogs = iogs[start:end]
                flags = self.anchor_is_untruncated[start:end]

                idx1 = np.array(np.floor((current_ious-0.01)/0.2)+1, np.int32) 
                max_ioa_iogs = np.maximum(current_ioas, current_iogs)
                idx2 = np.array(np.floor((max_ioa_iogs-0.01)/0.2)+1, np.int32) 

                current_masks = np.zeros((self.num_anchor[j]), np.float32)
                current_weights = np.zeros((self.num_anchor[j]), np.float32)
                current_reg_masks = np.zeros((self.num_anchor[j]), np.float32)

                for k in range(self.num_anchor[j]):
                    current_masks[k] = flags[k]
                    current_weights[k] = flags[k] * self.anchor_iou_weight[j, idx1[k], idx2[k]]
                    current_reg_masks[k] = flags[k] * self.anchor_iou_weight[j, idx1[k], idx2[k]] * (current_labels[k]==1)

                masks = np.concatenate((masks, current_masks))
                weights = np.concatenate((weights, current_weights))
                reg_masks = np.concatenate((reg_masks, current_reg_masks))
                start = end

            labels[np.where(labels==-1)[0]] = 0

            gt_anchor_labels.append(labels)
            gt_anchor_regs.append(regs)
            anchor_masks.append(masks)
            anchor_weights.append(weights)
            anchor_reg_masks.append(reg_masks)

        gt_anchor_labels = np.array(gt_anchor_labels)
        gt_anchor_regs = np.array(gt_anchor_regs)
        anchor_masks = np.array(anchor_masks)
        anchor_weights = np.array(anchor_weights)
        anchor_reg_masks = np.array(anchor_reg_masks)

        return gt_anchor_labels, gt_anchor_regs, anchor_masks, anchor_weights, anchor_reg_masks

    def process_roi_data(self, anchor_files):
        """ Prepare the RoI data for training RCN. """
        num_roi = self.num_roi
        rois = []
        gt_roi_classes = []
        gt_roi_regs = []
        roi_masks = []
        roi_weights = []
        roi_reg_masks = []

        X = self.num_object
        Y = self.num_background

        for i in range(self.batch_size):
            anchor_data = np.load(anchor_files[i])

            labels = anchor_data['labels']
            regs = anchor_data['regs'] 
            classes = anchor_data['classes']
            ious = anchor_data['ious']
            ioas = anchor_data['ioas']
            iogs = anchor_data['iogs']
            sorted_idx = anchor_data['sorted_idx']
            
            A = len(np.where(labels==1)[0])
            B = len(np.where(labels==0)[0])
            C = self.total_num_truncated_anchor

            U = min(X, A)
            V = min(Y, B)           

            if U>0:
                p = int(A*1.0/U)
                f = int(np.random.uniform(0, 1) * p)
                obj_idx = np.array(list(range(f, A, p)), np.int32)
            else:
                obj_idx = np.array([], np.int32)      

            if V>0:
                q = int(B*1.0/V)          
                g = int(np.random.uniform(0, 1) * q)  
                bg_idx = -np.array(list(range(g+C+1, B+C+1, q)), np.int32)
            else:
                bg_idx = np.array([], np.int32)      

            chosen_idx = np.concatenate((obj_idx, bg_idx))
            chosen_idx = sorted_idx[chosen_idx]

            num_real_roi = len(chosen_idx) 
            real_rois = self.parent_anchors[chosen_idx]
            real_roi_regs = regs[chosen_idx]
            real_roi_classes = classes[chosen_idx]
            real_roi_ious = ious[chosen_idx]
            real_roi_ioas = ioas[chosen_idx]
            real_roi_iogs = iogs[chosen_idx]

            idx1 = np.array(np.floor((real_roi_ious-0.01)/0.2)+1, np.int32) 
            max_ioa_iogs = np.maximum(real_roi_ioas, real_roi_iogs)
            idx2 = np.array(np.floor((max_ioa_iogs-0.01)/0.2)+1, np.int32) 

            real_roi_masks = np.ones((num_real_roi), np.float32)
            real_roi_weights = np.zeros((num_real_roi), np.float32)
            real_roi_reg_masks = np.zeros((num_real_roi), np.float32)

            for k in range(num_real_roi):
                real_roi_weights[k] = self.class_iou_weight[real_roi_classes[k], idx1[k], idx2[k]]
                real_roi_reg_masks[k] = self.class_iou_weight[real_roi_classes[k], idx1[k], idx2[k]] * (real_roi_classes[k]!=self.background_id) 

            current_rois = np.ones((num_roi, 4), np.int32) * 3 
            current_rois[:num_real_roi] = real_rois 

            current_roi_classes = np.ones((num_roi), np.int32) 
            current_roi_classes[:num_real_roi] = real_roi_classes 

            current_roi_regs = np.ones((num_roi, 4), np.float32) 
            current_roi_regs[:num_real_roi] = real_roi_regs 

            current_roi_masks = np.zeros((num_roi), np.float32) 
            current_roi_masks[:num_real_roi] = real_roi_masks 

            current_roi_weights = np.zeros((num_roi), np.float32)
            current_roi_weights[:num_real_roi] = real_roi_weights  

            current_roi_reg_masks = np.zeros((num_roi), np.float32)
            current_roi_reg_masks[:num_real_roi] = real_roi_reg_masks 

            rois.append(current_rois) 
            gt_roi_classes.append(current_roi_classes) 
            gt_roi_regs.append(current_roi_regs) 
            roi_masks.append(current_roi_masks) 
            roi_weights.append(current_roi_weights) 
            roi_reg_masks.append(current_roi_reg_masks) 

        rois = np.array(rois)
        gt_roi_classes = np.array(gt_roi_classes)
        gt_roi_regs = np.array(gt_roi_regs)
        roi_masks = np.array(roi_masks)
        roi_weights = np.array(roi_weights)
        roi_reg_masks = np.array(roi_reg_masks)

        return rois, gt_roi_classes, gt_roi_regs, roi_masks, roi_weights, roi_reg_masks

    def prepare_anchor_data(self, dataset, show_data=False):
        """ Prepare useful anchor data for training. """
        print("Labeling the anchors...")
        t = self.num_anchor_type
        r = self.num_class

        anchor_iou_freq = np.zeros((t, 6, 6), np.float32)
        class_iou_freq = np.zeros((r, 6, 6), np.float32)
        
        for i in tqdm(list(range(dataset.count))):
            img_file = dataset.img_files[i]
            H, W = dataset.img_heights[i], dataset.img_widths[i]
            gt_classes = np.array(dataset.gt_classes[i]) 
            gt_bboxes = np.array(dataset.gt_bboxes[i])
            gt_bboxes = convert_bbox(gt_bboxes, [H, W], self.img_shape[:2])

            # Label the anchors and find their closest ground truth bounding boxes
            labels, bboxes, classes, ious, ioas, iogs = label_anchors(self.anchors, self.anchor_is_untruncated, gt_classes, gt_bboxes, self.background_id) 

            start = 0
            for j in range(t):
                end = start + self.num_anchor[j]
                current_labels = labels[start:end]
                current_classes = classes[start:end]
                current_ious = ious[start:end]
                current_ioas = ioas[start:end]
                current_iogs = iogs[start:end]
                flags = self.anchor_is_untruncated[start:end]

                idx1 = np.array(np.floor((current_ious-0.01)/0.2)+1, np.int32) 
                max_ioa_iogs = np.maximum(current_ioas, current_iogs)
                idx2 = np.array(np.floor((max_ioa_iogs-0.01)/0.2)+1, np.int32) 

                for k in range(self.num_anchor[j]):                    
                    anchor_iou_freq[j, idx1[k], idx2[k]] += flags[k]
                    class_iou_freq[current_classes[k], idx1[k], idx2[k]] += flags[k]

                start = end

            sorted_idx = np.argsort(ious)[::-1]
            num_hit = len(np.where(labels==1)[0])  

            regs = param_bbox(bboxes, self.anchors)

            np.savez(dataset.anchor_files[i], labels=labels, bboxes=bboxes, regs=regs, classes=classes, ious=ious, ioas=ioas, iogs=iogs, sorted_idx=sorted_idx)

            # Show the positive anchors if required
            if show_data:
                img = cv2.imread(img_file)
                targets = convert_bbox(bboxes, self.img_shape[:2], [H, W])
                scaled_anchors = convert_bbox(self.anchors, self.img_shape[:2], [H, W])
                for k in range(self.total_num_anchor):
                    y, x, h, w = targets[k]
                    cv2.rectangle(img, (x, y), (x+w-1, y+h-1), (255, 0, 0), 2)
                    if labels[k]==1:
                        cv2.rectangle(img, (scaled_anchors[k][1], scaled_anchors[k][0]), (scaled_anchors[k][1]+scaled_anchors[k][3]-1, scaled_anchors[k][0]+scaled_anchors[k][2]-1), (255, 255, 255), 2)
                winname = '%d Hits' %(num_hit)
                cv2.imshow(winname, img)
                cv2.moveWindow(winname, 100, 100)
                cv2.waitKey(1000)
                cv2.destroyAllWindows()

        # Save the statistics
        self.anchor_iou_freq = (anchor_iou_freq + 0.001) / dataset.count
        self.class_iou_freq = (class_iou_freq + 0.001) / dataset.count
        np.savez(self.anchor_stat_file, anchor_iou_freq=self.anchor_iou_freq, class_iou_freq=self.class_iou_freq)

    def process_rpn_result(self, probs, rois):
        """ Process the RPN result. """
        probs = probs[np.where(self.anchor_is_untruncated==1)[0]]
        rois = rois[np.where(self.anchor_is_untruncated==1)[0]]
        num_roi, _, top_k_rois = nms(probs, rois, self.num_roi)
        return num_roi, np.array(top_k_rois)

    def process_rcn_result(self, probs, classes, bboxes, H, W):
        """ Process the RCN result. """
        valid_idx = np.where(classes!=self.background_id)[0]
        det_probs = probs[valid_idx]
        det_classes = classes[valid_idx]
        det_bboxes = bboxes[valid_idx]

        if len(valid_idx)==0:
            return 0, np.array([]), np.array([]), np.array([])

        num_det, top_k_scores, top_k_classes, top_k_bboxes = postprocess(det_probs, det_classes, det_bboxes)  

        top_k_bboxes = convert_bbox(top_k_bboxes, self.img_shape[:2], [H, W])

        return num_det, np.array(top_k_scores), np.array(top_k_classes), np.array(top_k_bboxes)
