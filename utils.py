# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import math
import pprint
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
import PIL
from PIL import Image
import random
from scipy import ndimage
from cv2 import cv2
import glob
import time
pp = pprint.PrettyPrinter()
get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

def imread(path):
   img = cv2.imread(path)
   im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float)
   return im_rgb

def load_deviation_data(batch_idx, data_load, phase, deviation):
    EVS_batch = []
    SVS_batch = []
    CVS_batch = []
    cur_CVS_batch = []
    cur_EVS_batch = []
    cur_SVS_batch = []
    path = data_load
    path_EVS = os.path.join(path, phase+'/EVS')
    path_SVS = os.path.join(path, phase + '/SVS')
    path_CVS = os.path.join(path, phase + '/CVS')
    for i in range(batch_idx[0], batch_idx[1]):
        idx = i
        print(idx)
        img_EVS = []
        img_SVS = []
        img_CVS = []
        for j in range(i - deviation + 1, i):
            jdx = j
            img_EVS_ = imread(os.path.join(path_EVS, 'EVS%d.bmp'%(jdx)))
            img_SVS_ = imread(os.path.join(path_SVS, 'SVS%d.bmp'%(jdx)))
            img_CVS_ = imread(os.path.join(path_CVS,  'CVS%d.bmp'%(jdx)))
            img_EVS.append(img_EVS_)
            img_SVS.append(img_SVS_)
            img_CVS.append(img_CVS_)
        for j in range(i, i + deviation):
            jdx = j
            img_EVS_ = imread(os.path.join(path_EVS, 'EVS%d.bmp'%(jdx)))
            img_SVS_ = imread(os.path.join(path_SVS, 'SVS%d.bmp'%(jdx)))
            img_CVS_ = imread(os.path.join(path_CVS, 'CVS%d.bmp'%(jdx)))
            img_EVS.append(img_EVS_)
            img_SVS.append(img_SVS_)
            img_CVS.append(img_CVS_)
        EVS = np.concatenate((img_EVS), axis=2)
        SVS = np.concatenate((img_SVS), axis=2)
        CVS = np.concatenate((img_CVS), axis=2)
        img_cur_CVS_ = imread(os.path.join(path_CVS, 'CVS%d.bmp'%(idx)))
        img_cur_SVS_ = imread(os.path.join(path_SVS, 'SVS%d.bmp'%(idx)))
        img_cur_EVS_ = imread(os.path.join(path_EVS, 'EVS%d.bmp'%(idx)))
        EVS_batch.append(EVS)
        SVS_batch.append(SVS)
        CVS_batch.append(CVS)

        cur_CVS_batch.append(img_cur_CVS_)
        cur_SVS_batch.append(img_cur_SVS_)
        cur_EVS_batch.append(img_cur_EVS_)
    EVS_batch,SVS_batch,CVS_batch = prepross(EVS_batch,SVS_batch,CVS_batch, phase)
    cur_EVS_batch,cur_SVS_batch,cur_CVS_batch = prepross(cur_EVS_batch,cur_SVS_batch,cur_CVS_batch, phase)
    return EVS_batch, SVS_batch, CVS_batch, cur_CVS_batch, cur_EVS_batch, cur_SVS_batch

def canny(img):
    canny = cv2.Canny(np.uint8(img), 500, 1000)
    return np.reshape(canny, (canny.shape[0], canny.shape[1], 1))


def optical_flow(imgs, dst='./capture_folder'):
    for idx, file in enumerate(imgs):
        copyfile(file, dst + '/' + str(idx) + '.bmp')
    feature_params = dict(maxCorners = 100, qualityLevel = 0.3, minDistance = 7, blockSize = 7)
    lk_params = dict(winSize = (15,15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    cap = cv2.VideoCapture(dst + "/%01d.bmp")
    color = np.random.randint(0,255,(100,3))
    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)

    while(1):
        ret,frame = cap.read()
        if frame is None:
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        # Select good points
        good_new = p1[st==1]
        good_old = p0[st==1]
        # draw the tracks
        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
            frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
        img = cv2.add(frame,mask)
        cv2.imshow('frame',img)
        #k = cv2.waitKey(30) & 0xff
        #if k == 27:
        #    break
        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1,1,2)
    cv2.destroyAllWindows()
    cap.release()
    return mask

def load_data(batch_idx,data_load,phase):
    img_IR = []
    img_TV = []
    img_EN = []
    
    path = data_load
    path_IR = os.path.join(path, phase+'/EVS')
    path_TV = os.path.join(path, phase+'/SVS')
    path_EN = os.path.join(path, phase+'/CVS')
    # a_IR = glob.glob(os.path.join(path_IR,'*.bmp'))
    # a_TV = glob.glob(os.path.join(path_TV,'*.bmp'))
    # a_EN = glob.glob(os.path.join(path_EN,'*.bmp'))

    for i in range(batch_idx[0],batch_idx[1]):
        idx = i
        #if idx > 750:
        #    idx = idx + 461
        print(idx)
        img_IR_ = imread(os.path.join(path_IR, 'EVS%d.bmp' %idx))
        img_TV_ = imread(os.path.join(path_TV, 'SVS%d.bmp' %idx))
        img_EN_ = imread(os.path.join(path_EN, 'CVS%d.bmp' %idx))
        # img_IR_ = imread(a_IR[i])
        # img_TV_ = imread(a_TV[i])
        # img_EN_ = imread(a_EN[i])
        img_IR.append(img_IR_)
        img_TV.append(img_TV_)
        img_EN.append(img_EN_)
    img_IR,img_TV,img_EN = prepross(img_IR,img_TV,img_EN, phase)
    return img_IR, img_TV, img_EN

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img

def save_images(images, size, image_path,sample_dir,phase):
    print('image path is:', sample_dir)
    return imsave(inverse_transform(images), size, image_path,sample_dir,phase)

def imsave(images, size, path,sample_dir,phase):
    shape = images.shape
    if phase == 'train':
        _,img_name = os.path.split(path)
        images = np.reshape(images[0], [shape[1], shape[1], shape[3]])
        images = np.clip(images,0,1)
        return plt.imsave(sample_dir+'/'+img_name, images, format='png', cmap='gray')
    else:
        path,img_name = os.path.split(path)
        images = np.reshape(images, [1, shape[1], shape[1], shape[3]], 3)
        images = np.clip(images,0,1)
        return plt.imsave(os.path.join(path,'./'  + img_name), images[0], format='png', cmap='gray')

def inverse_transform(images):
    return (images+1.)/2.

def prepross(img_IR,img_TV,img_EN, phase, flip=True):
    img_A = np.array(img_IR)
    img_B = np.array(img_TV)
    img_C = np.array(img_EN)

#    if flip and np.random.random() > 0.5:
#        img_A = np.fliplr(img_A)
#        img_B = np.fliplr(img_B)
#        img_C = np.fliplr(img_C)

#    if flip and np.random.random() > 0.5:
#        img_A = np.flipud(img_A)
#        img_B = np.flipud(img_B)
#        img_C = np.flipud(img_C)
    if phase == 'train':
        change_img_A = np.random.randint(1, 8)
        change_img_B = np.random.randint(1, 8)

    #    if np.random.random() > 0.5:
    #       img_A = np.fliplr(img_A)
    #       img_B = np.fliplr(img_B)
    #      img_C = np.fliplr(img_C)
    #
    #    if np.random.random() > 0.5:
    #        img_B = np.flipud(img_B)
    #       img_C = np.flipud(img_C)
    #   shift_x = np.random.randint(0, 3)
    #   shift_y = np.random.randint(0, 3)
    #   img_size = img_A.shape[1]
    #   img_A = img_A[0,shift_x:-1,shift_y:-1]
    #   img_A = scipy.misc.imresize(img_A,(img_size, img_size))
    #
        shift_x = np.random.randint(0, 3)
        shift_y = np.random.randint(0, 3)
        img_size = img_B.shape[1]

        #img_B = img_B[0,shift_x:-1,shift_y:-1]
        #mg_B = scipy.misc.imresize(img_B,(img_size, img_size))

        if ((change_img_A == 1) or (change_img_A > 3)):
            brightness_A = -75 + 150 * np.random.random()
            img_A = img_A + brightness_A
            img_A = np.clip(img_A, 0, 255)

        if ((change_img_B == 1) or (change_img_B > 3)):
            brightness_B = -75 + 150 * np.random.random()
            img_B = img_B + brightness_B
            img_B = np.clip(img_B, 0, 255)

        if ((change_img_A == 2) or (change_img_A > 4)):
            sigma_blur_A = 2 * np.random.random()
            img_A = ndimage.gaussian_filter(img_A, sigma_blur_A)

        if ((change_img_B == 2) or (change_img_B > 4)):
            sigma_blur_B = 2 * np.random.random()       
            img_B = ndimage.gaussian_filter(img_B, sigma_blur_B)

        mean = 0.0
        if ((change_img_A == 3) or (change_img_A > 5)):
            std_devian_A = 20 * np.random.random()
            img_A = img_A + np.random.normal(mean, std_devian_A, img_A.shape)
            img_A = np.clip(img_A, 0, 255)

        if ((change_img_B == 3) or (change_img_B > 5)):
            std_devian_B = 20 * np.random.random()
            img_B = img_B + np.random.normal(mean, std_devian_B, img_B.shape)
            img_B = np.clip(img_B, 0, 255)

    img_A = img_A / 127.5 - 1.
    img_B = img_B / 127.5 - 1.
    img_C = img_C / 127.5 - 1.
    return img_A,img_B,img_C

    
