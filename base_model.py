import os
import math
import sys
import numpy as np
import tensorflow as tf
import cv2
import cPickle as pickle
from tqdm import tqdm

from dataset import *
from bbox import *
from utils.coco.coco import *
from utils.coco.cocoeval import *

class ImageLoader(object):
    def __init__(self, mean_file):
        self.bgr = True 
        self.scale_shape = np.array([640, 640], np.int32)
        self.crop_shape = np.array([640, 640], np.int32)
        self.mean = np.load(mean_file).mean(1).mean(1)

    def load_img(self, img_file):      
        """ Load and preprocess an image. """
        img = cv2.imread(img_file)

        if self.bgr:
            temp = img.swapaxes(0, 2)
            temp = temp[::-1]
            img = temp.swapaxes(0, 2)

        img = cv2.resize(img, (self.scale_shape[0], self.scale_shape[1]))
        offset = (self.scale_shape - self.crop_shape) / 2
        offset = offset.astype(np.int32)
        img = img[offset[0]:offset[0]+self.crop_shape[0], offset[1]:offset[1]+self.crop_shape[1], :]
        img = img - self.mean
        return img

    def load_imgs(self, img_files):
        """ Load and preprocess a list of images. """
        imgs = []
        for img_file in img_files:
            imgs.append(self.load_img(img_file))
        imgs = np.array(imgs, np.float32)
        return imgs


class BaseModel(object):
    def __init__(self, params, mode):
        self.params = params

        self.mode = mode
        self.batch_size = params.batch_size if mode=='train' else 1
        self.batch_norm = params.batch_norm

        if params.dataset == 'coco':
            self.type = 'coco'
            self.num_class = coco_num_class
            self.class_names = coco_class_names
            self.class_colors = coco_class_colors
            self.class_to_category = coco_class_to_category
            self.category_to_class = coco_category_to_class
            self.background_id = self.num_class - 1
        else:
            self.type = 'pascal'
            self.num_class = pascal_num_class
            self.class_names = pascal_class_names
            self.class_colors = pascal_class_colors
            self.class_ids = pascal_class_ids
            self.background_id = self.num_class - 1

        self.basic_model = params.basic_model
        self.num_roi = params.num_roi
        self.bbox_per_class = params.bbox_per_class

        self.class_balancing_factor = params.class_balancing_factor

        self.label = self.type + '/' + self.basic_model + '/'
        self.save_dir = os.path.join(params.save_dir, self.label)

        self.img_loader = ImageLoader(params.mean_file)
        self.img_shape = [640, 640, 3]

        self.anchor_scales = [50, 100, 200, 300, 400, 500] 
        self.anchor_ratios = [[1.0/math.sqrt(2), math.sqrt(2)], [1.0, 1.0], [math.sqrt(2), 1.0/math.sqrt(2)]]
        self.num_anchor_type = len(self.anchor_scales) * len(self.anchor_ratios)

        self.anchor_shapes = []
        for s in self.anchor_scales:
            for r in self.anchor_ratios:
                self.anchor_shapes.append([int(s*r[0]), int(s*r[1])])

        self.anchor_stat_file = self.type + '_anchor_stats.npz'
        
        self.global_step = tf.Variable(0, name = 'global_step', trainable = False) 
        self.build() 
        self.saver = tf.train.Saver(max_to_keep = 100) 
   
    def build(self):
        raise NotImplementedError()

    def prepare_anchor_data(self, dataset, show_data=False):
        raise NotImplementedError() 

    def process_rpn_result(self, probs, regs):
        raise NotImplementedError()

    def process_rcn_result(self, probs, classes, regs, rois, h, w):
        raise NotImplementedError()

    def get_feed_dict_for_rpn(self, batch, is_train, feats):
        raise NotImplementedError()

    def get_feed_dict_for_rcn(self, batch, is_train, feats, rois=None, masks=None):
        raise NotImplementedError()

    def get_feed_dict_for_all(self, batch, is_train, feats=None):
        raise NotImplementedError()

    def train_rpn(self, sess, train_dataset):
        """ Train the RPN. """
        print("Training the RPN...")
        params = self.params
        self.setup()

        for epoch_no in tqdm(list(range(params.num_epoch)), desc='epoch'): 
            for idx in tqdm(list(range(train_dataset.num_batches)), desc='batch'):
                batch = train_dataset.next_batch()
                img_files, _ = batch
                imgs = self.img_loader.load_imgs(img_files)
                feats = sess.run(self.conv_feats, feed_dict={self.imgs:imgs, self.is_train:False})               
                feed_dict = self.get_feed_dict_for_rpn(batch, is_train=True, feats=feats)
                _, loss0, loss1, global_step = sess.run([self.rpn_opt_op, self.rpn_loss0, self.rpn_loss1, self.global_step], feed_dict=feed_dict)
                print(" loss0=%f loss1=%f" %(loss0, loss1))

                if (global_step+1) % params.save_period == 0:
                    self.save(sess)

            train_dataset.reset()

        print("RPN trained.")

    def train_rcn(self, sess, train_dataset):
        """ Train the RCN. """
        print("Training the RCN...")
        params = self.params
        self.setup()

        for epoch_no in tqdm(list(range(params.num_epoch)), desc='epoch'): 
            for idx in tqdm(list(range(train_dataset.num_batches)), desc='batch'): 
                batch = train_dataset.next_batch() 
                img_files, _ = batch
                imgs = self.img_loader.load_imgs(img_files)
                feats = sess.run(self.conv_feats, feed_dict={self.imgs:imgs, self.is_train:False})
                feed_dict = self.get_feed_dict_for_rcn(batch, is_train=True, feats=feats) 
                _, loss0, loss1, global_step = sess.run([self.rcn_opt_op, self.rcn_loss0, self.rcn_loss1, self.global_step], feed_dict=feed_dict) 
                print(" loss0=%f loss1=%f" %(loss0, loss1)) 

                if (global_step+1) % params.save_period == 0:
                    self.save(sess)

            train_dataset.reset()

        print("RCN trained.")

    def train(self, sess, train_dataset):
        """ Train both the RPN and RCN. """
        print("Training the model...")
        params = self.params
        self.setup()
        
        for epoch_no in tqdm(list(range(params.num_epoch)), desc='epoch'): 
            for idx in tqdm(list(range(train_dataset.num_batches)), desc='batch'): 
                batch = train_dataset.next_batch() 
                img_files, _ = batch
                imgs = self.img_loader.load_imgs(img_files)
                feats = sess.run(self.conv_feats, feed_dict={self.imgs:imgs, self.is_train:False})
                feed_dict = self.get_feed_dict_for_all(batch, is_train=True, feats=feats) 
                _, loss0, loss1, global_step = sess.run([self.opt_op, self.loss0, self.loss1, self.global_step], feed_dict=feed_dict) 
                print(" loss0=%f loss1=%f" %(loss0, loss1)) 

                if (global_step+1) % params.save_period == 0:
                    self.save(sess)

            train_dataset.reset()

        print("Model trained.")

    def val_coco(self, sess, val_coco, val_dataset):
        """ Validate the model on COCO dataset. """
        print("Validating the model...")
        num_roi = self.num_roi
        det_scores = []
        det_classes = []
        det_bboxes = []

        for k in tqdm(list(range(val_dataset.count))):
            batch = val_dataset.next_batch()
            img_files = batch
            img_file = img_files[0]
            H, W = val_dataset.img_heights[k], val_dataset.img_widths[k]
            
            # Propose the RoIs
            imgs = self.img_loader.load_imgs(img_files)
            feats = sess.run(self.conv_feats, feed_dict={self.imgs:imgs, self.is_train:False})           

            feed_dict = self.get_feed_dict_for_rpn(batch, is_train=False, feats=feats)
            scores, regs = sess.run([self.rpn_scores, self.rpn_regs], feed_dict=feed_dict)

            rois = unparam_bbox(regs.squeeze(), self.anchors, self.img_shape[:2])
            num_real_roi, real_rois = self.process_rpn_result(scores.squeeze(), rois)

            # Add dummy RoIs if necessary
            rois = np.ones((num_roi, 4), np.int32) * 3
            rois[:num_real_roi] = real_rois
            expanded_rois = expand_bbox(rois, self.img_shape[:2])
            expanded_rois = np.expand_dims(expanded_rois, 0)
            
            masks = np.zeros((num_roi), np.float32)
            masks[:num_real_roi] = 1.0
            masks = np.expand_dims(masks, 0)
            
            # Classify the RoIs
            feed_dict = self.get_feed_dict_for_rcn(batch, is_train=False, feats=feats, rois=expanded_rois, masks=masks)
            scores, classes, regs = sess.run([self.res_scores, self.res_classes, self.res_regs], feed_dict=feed_dict)
            bboxes = unparam_bbox(regs.squeeze(), rois)

            # Postprocess
            num_det, scores, classes, bboxes = self.process_rcn_result(scores.squeeze(), classes.squeeze(), bboxes, H, W)
            
            det_scores.append(scores)
            det_classes.append(classes)
            det_bboxes.append(bboxes)

        val_dataset.reset() 

        # Evaluate the results 
        results = [] 
        for i in range(val_dataset.count): 
            for s, c, b in zip(det_scores[i], det_classes[i], det_bboxes[i]): 
                results.append({'image_id': val_dataset.img_ids[i], 'category_id': self.class_to_category[c], 'bbox':[b[1], b[0], b[3]-1, b[2]-1], 'score': s}) 

        res_coco = val_coco.loadRes2(results) 
        E = COCOeval(val_coco, res_coco) 
        E.evaluate() 
        E.accumulate() 
        E.summarize() 
        print("Validation complete.")

    def val_pascal(self, sess, val_pascal, val_dataset):
        """ Validate the model on PASCAL dataset. """
        print("Validating the model...")
        num_roi = self.num_roi
        det_scores = []
        det_classes = []
        det_bboxes = []

        for k in tqdm(list(range(val_dataset.count))):
            batch = val_dataset.next_batch()
            img_files = batch
            img_file = img_files[0]
            H, W = val_dataset.img_heights[k], val_dataset.img_widths[k]
            
            # Propose the RoIs
            imgs = self.img_loader.load_imgs(img_files)
            feats = sess.run(self.conv_feats, feed_dict={self.imgs:imgs, self.is_train:False})           

            feed_dict = self.get_feed_dict_for_rpn(batch, is_train=False, feats=feats)
            scores, regs = sess.run([self.rpn_scores, self.rpn_regs], feed_dict=feed_dict)

            rois = unparam_bbox(regs.squeeze(), self.anchors, self.img_shape[:2])
            num_real_roi, real_rois = self.process_rpn_result(scores.squeeze(), rois)

            # Add dummy RoIs if necessary
            rois = np.ones((num_roi, 4), np.int32) * 3
            rois[:num_real_roi] = real_rois
            expanded_rois = expand_bbox(rois, self.img_shape[:2])
            expanded_rois = np.expand_dims(expanded_rois, 0)
            
            masks = np.zeros((num_roi), np.float32)
            masks[:num_real_roi] = 1.0
            masks = np.expand_dims(masks, 0)
            
            # Classify the RoIs
            feed_dict = self.get_feed_dict_for_rcn(batch, is_train=False, feats=feats, rois=expanded_rois, masks=masks)
            scores, classes, regs = sess.run([self.res_scores, self.res_classes, self.res_regs], feed_dict=feed_dict)
            bboxes = unparam_bbox(regs.squeeze(), rois)

            # Postprocess
            num_det, scores, classes, bboxes = self.process_rcn_result(scores.squeeze(), classes.squeeze(), bboxes, H, W)
            
            det_scores.append(scores)
            det_classes.append(classes)
            det_bboxes.append(bboxes)
 
        val_dataset.reset() 

        # Evaluate the results 
        results = {} 
        for i in range(val_dataset.count): 
            file_name = val_dataset.img_files[i].split(os.sep)[-1]
            results[file_name] = []
            for s, c, b in zip(det_scores[i], det_classes[i], det_bboxes[i]): 
                results[file_name].append({'class_id': c, 'bbox':[b[1], b[0], b[1]+b[3]-1, b[0]+b[2]-1], 'score': s}) 

        eval_pascal(val_pascal, results)
        print("Validation complete.")

    def test(self, sess, test_dataset, show_rois=True, show_dets=True):
        """ Test the model. """
        print("Testing the model...")
        num_roi = self.num_roi
        font = cv2.FONT_HERSHEY_COMPLEX
        result_dir = self.params.test_result_dir
        det_scores = []
        det_classes = []
        det_bboxes = []

        for k in tqdm(list(range(test_dataset.count))):
            batch = test_dataset.next_batch()
            img_files = batch
            img_file = img_files[0]
            img_name = os.path.splitext(img_file.split(os.sep)[-1])[0]
            H, W = test_dataset.img_heights[k], test_dataset.img_widths[k]

            # Propose the RoIs
            imgs = self.img_loader.load_imgs(img_files)
            feats = sess.run(self.conv_feats, feed_dict={self.imgs:imgs, self.is_train:False})           

            feed_dict = self.get_feed_dict_for_rpn(batch, is_train=False, feats=feats)
            scores, regs = sess.run([self.rpn_scores, self.rpn_regs], feed_dict=feed_dict)

            rois = unparam_bbox(regs.squeeze(), self.anchors, self.img_shape[:2])
            num_real_roi, real_rois = self.process_rpn_result(scores.squeeze(), rois)

            # Show the RoIs if required
            scaled_rois = convert_bbox(real_rois, self.img_shape[:2], [H, W])
            img = cv2.imread(img_file)
            for roi in scaled_rois:                               
                y, x, h, w = roi
                cv2.rectangle(img, (x, y), (x+w-1, y+h-1), (255,255,255), 2)

            if show_rois:
                winname = '%d RoIs' %(num_real_roi)
                cv2.imshow(winname, img)
                cv2.moveWindow(winname, 100, 100)
                cv2.waitKey(1000)

            cv2.imwrite(os.path.join(result_dir, img_name+'_rois.jpg'), img)

            # Add dummy RoIs if necessary
            rois = np.ones((num_roi, 4), np.int32) * 3
            rois[:num_real_roi] = real_rois
            expanded_rois = expand_bbox(rois, self.img_shape[:2])
            expanded_rois = np.expand_dims(expanded_rois, 0)
            
            masks = np.zeros((num_roi), np.float32)
            masks[:num_real_roi] = 1.0
            masks = np.expand_dims(masks, 0)
            
            # Classify the RoIs
            feed_dict = self.get_feed_dict_for_rcn(batch, is_train=False, feats=feats, rois=expanded_rois, masks=masks)
            scores, classes, regs = sess.run([self.res_scores, self.res_classes, self.res_regs], feed_dict=feed_dict)
            bboxes = unparam_bbox(regs.squeeze(), rois)

            # Postprocess
            num_det, scores, classes, bboxes = self.process_rcn_result(scores.squeeze(), classes.squeeze(), bboxes, H, W)
            
            det_scores.append(scores)
            det_classes.append(classes)
            det_bboxes.append(bboxes)
 
            # Show the detection result if required
            img = cv2.imread(img_file)
            for i in range(num_det):                               
                y, x, h, w = bboxes[i]
                c = self.class_colors[classes[i]]
                cv2.rectangle(img, (x, y), (x+w-1, y+h-1), c, 2)
                cv2.rectangle(img, (x, y-8), (x+w-1, y), c, -1)

            for i in range(num_det):                               
                y, x, h, w = bboxes[i]
                n = self.class_names[classes[i]]
                cv2.putText(img, '%s' %(n), (x+5, y), font, 0.4, (255, 255, 255), 1)

            if show_dets:
                winname = '%d Detections' %(num_det)
                cv2.imshow(winname, img)
                cv2.moveWindow(winname, 700, 100)
                cv2.waitKey(1000)
                cv2.destroyAllWindows()

            cv2.imwrite(os.path.join(result_dir, img_name+'_result.jpg'), img)

        # Save the results
        results = {} 
        for i in range(test_dataset.count): 
            img_file = test_dataset.img_files[i]
            results[img_file] = []
            for s, c, b in zip(det_scores[i], det_classes[i], det_bboxes[i]): 
                results[img_file].append({'class_name': self.class_names[c], 'bbox':[b[1], b[0], b[3]-1, b[2]-1], 'score': s}) 

        pickle.dump(results, open(self.params.test_result_file, 'wb')) 
        print("Testing complete.") 

    def setup(self, show_data=True):
        """ Setup useful parameters for class balancing. """
        p = self.class_balancing_factor      

        stats = np.load(self.anchor_stat_file)
        self.anchor_iou_freq = stats['anchor_iou_freq']
        self.class_iou_freq = stats['class_iou_freq']

        if show_data:
            print("Class frequencies:")
            for j in range(self.num_anchor_type):
                print("Type [%d, %d]:" %(self.anchor_shapes[j][0], self.anchor_shapes[j][1]))
                print(self.anchor_iou_freq[j])

            for j in range(self.num_class):
                print("Class %s:" %(self.class_names[j]))
                print(self.class_iou_freq[j])

        self.anchor_iou_weight = np.exp(-np.log(self.anchor_iou_freq)*p) 
        self.anchor_iou_weight[np.where(self.anchor_iou_weight>1e5)] = 0
        self.anchor_iou_weight[:, :3, :] *= 0.2

        M = np.sum(self.class_iou_freq[:-1, 4:, :]) * 1.0
        K = np.sum(self.class_iou_freq[-1, :3, :]) * 1.0

        self.num_object = min(M, self.num_roi*0.6)
        self.num_background = min(K, self.num_roi*0.4)

        self.obj_filter_rate = self.num_object / M
        self.bg_filter_rate = self.num_background / K

        self.class_iou_weight = np.exp(-np.log(self.class_iou_freq*self.obj_filter_rate)*p) 
        self.class_iou_weight[-1] = np.exp(-np.log(self.class_iou_freq[-1]*self.bg_filter_rate)*p) * 0.2
        self.class_iou_weight[np.where(self.class_iou_weight>1e5)] = 0

        if show_data:
            print("Class weights:")
            for j in range(self.num_anchor_type):
                print("Type [%d, %d]:" %(self.anchor_shapes[j][0], self.anchor_shapes[j][1]))
                print(self.anchor_iou_weight[j])

            for j in range(self.num_class):
                print("Class %s:" %(self.class_names[j]))
                print(self.class_iou_weight[j])

    def save(self, sess):
        """ Save the trained model. """
        print("Saving model to %s" %self.save_dir)
        self.saver.save(sess, self.save_dir, self.global_step)

    def load(self, sess):
        """ Load the trained model. """
        print("Loading model...") 
        checkpoint = tf.train.get_checkpoint_state(self.save_dir) 
        if checkpoint is None: 
            print("Error: No saved model found. Please train first.") 
            sys.exit(0) 
        self.saver.restore(sess, checkpoint.model_checkpoint_path) 

    def load2(self, data_path, session, ignore_missing=True):
        """ Load the pretrained CNN model. """
        print("Loading basic model from %s..." %data_path)
        data_dict = np.load(data_path).item()
        count = 0
        miss_count = 0
        for op_name in data_dict:
            with tf.variable_scope(op_name, reuse=True):
                for param_name, data in data_dict[op_name].iteritems():
                    try:
                        var = tf.get_variable(param_name)
                        session.run(var.assign(data))
                        count += 1
                       #print("Variable %s:%s loaded" %(op_name, param_name))
                    except ValueError:
                        miss_count += 1
                       #print("Variable %s:%s missed" %(op_name, param_name))
                        if not ignore_missing:
                            raise
        print("%d variables loaded. %d variables missed." %(count, miss_count))

