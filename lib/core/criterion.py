# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
from torch.nn import functional as F

import numpy as np
import math

from . import pyfft as fft
import logging
from . import common
logger = logging.getLogger(__name__)

def spatial_pyramid_pool(previous_conv, num_sample, previous_conv_size, out_pool_size):
    '''
    previous_conv: a tensor vector of previous convolution layer
    num_sample: an int number of image in the batch
    previous_conv_size: an int vector [height, width] of the matrix features size of previous convolution layer
    out_pool_size: a int vector of expected output size of max pooling layer
    
    returns: a tensor vector with shape [1 x n] is the concentration of multi-level pooling
    '''    
    #print(previous_conv.size())
    for i in range(len(out_pool_size)):
        #print(previous_conv_size)
        h_wid = int(math.ceil(previous_conv_size[0] / out_pool_size[i]))
        w_wid = int(math.ceil(previous_conv_size[1] / out_pool_size[i]))
        h_pad = math.ceil((h_wid*out_pool_size[i] - previous_conv_size[0] + 1)/2)
        w_pad = math.ceil((w_wid*out_pool_size[i] - previous_conv_size[1] + 1)/2)
        avgpool = nn.AvgPool2d((h_wid, w_wid), stride=(h_wid, w_wid), padding=(h_pad, w_pad))
        x = avgpool(previous_conv)
        #print(x.shape)
        b, c, h, w = x.size(0), x.size(1), x.size(2), x.size(3)
        x = (x/torch.norm(x,p=2,dim=1).view(b, 1, h, w)).view(b, c, h*w)
        #print(x.shape)
        if(i == 0):
            spp = x
            #print("spp size:",spp.size())
        else:
            spp = torch.cat((spp,x), 2)
            #print("size:",spp.size())
    return spp

def psnr(img1, img2):
   mse = np.mean( (img1/255. - img2/255.) ** 2 )
   if mse < 1.0e-10:
      return 100
   PIXEL_MAX = 1
   return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

class focal_loss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=0.25):
        super(focal_loss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, preds, targets):
        '''
        https://github.com/c0nn3r/RetinaNet/blob/master/focal_loss.py
        :param preds: [N,C]
        :param targets:[N]
        :return: focal-loss
        '''
        logpt = -F.cross_entropy(preds, targets)
        pt = torch.exp(logpt)
        focal_loss = -self.alpha * (1 - pt) ** self.gamma * logpt
        return focal_loss

class GHMC_loss(torch.nn.Module):
    def __init__(self, bins=10, momentum=0, use_sigmiod=True, loss_weight=1.0):
        super(GHMC_loss, self).__init__()
        self.bins = bins
        self.momentum = momentum
        self.edges = [float(x) / bins for x in range(bins + 1)]
        self.edges[-1] += 1e-6
        if momentum > 0:
            self.acc_sum = [0.0 for _ in range(bins)]
        self.use_sigmoid = use_sigmiod
        self.loss_weight = loss_weight

    def forward(self, pred, target, label_weight):
        '''

        :param pred:[batch_num, class_num]:
        :param target:[batch_num, class_num]:Binary class target for each sample.
        :param label_weight:[batch_num, class_num]: the value is 1 if the sample is valid and 0 if ignored.
        :return: GHMC_Loss
        '''
        if not self.use_sigmoid:
            raise NotImplementedError
        target, label_weight = target.float(), label_weight.float()
        edges = self.edges
        mmt = self.momentum
        weights = torch.zeros_like(pred)

        # gradient length
        g = torch.abs(pred.sigmoid().detach() - target)
        valid = label_weight > 0
        total = max(valid.float().sum().item(), 1.0)
        n = 0  # the number of valid bins

        for i in range(self.bins):
            inds = (g >= edges[i]) & (g <= edges[i + 1]) & valid
            num_in_bins = inds.sum().item()
            if num_in_bins > 0:
                if mmt > 0:
                    self.acc_sum[i] = mmt * self.acc_sum[i] + (1 - mmt) * num_in_bins
                    weights[inds] = total / self.acc_sum[i]
                else:
                    weights[inds] = total / num_in_bins
                n += 1

        if n > 0:
            weights = weights / n

        loss = F.binary_cross_entropy_with_logits(pred, target, weights, reduction='sum') / total

        return loss * self.loss_weight	

class CrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, weight=None):
        super(CrossEntropy, self).__init__()
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(weight=weight, 
                                             ignore_index=ignore_label)

    def forward(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.upsample(
                    input=score, size=(h, w), mode='bilinear')

        loss = self.criterion(score, target)

        return loss

class OhemCrossEntropy(nn.Module): 
    def __init__(self, ignore_label=-1, thres=0.7, 
        min_kept=100000, weight=None): 
        super(OhemCrossEntropy, self).__init__() 
        self.thresh = thres
        self.min_kept = max(1, min_kept)
        self.ignore_label = ignore_label 
        self.criterion = nn.CrossEntropyLoss(weight=weight, 
                                             ignore_index=ignore_label, 
                                             reduction='none') 
    
    def forward(self, score, target, **kwargs):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.upsample(input=score, size=(h, w), mode='bilinear')
        pred = F.softmax(score, dim=1)
        pixel_losses = self.criterion(score, target).contiguous().view(-1)
        mask = target.contiguous().view(-1) != self.ignore_label         
        
        tmp_target = target.clone() 
        tmp_target[tmp_target == self.ignore_label] = 0 
        pred = pred.gather(1, tmp_target.unsqueeze(1)) 
        pred, ind = pred.contiguous().view(-1,)[mask].contiguous().sort()
        min_value = pred[min(self.min_kept, pred.numel() - 1)] 
        threshold = max(min_value, self.thresh) 
        
        pixel_losses = pixel_losses[mask][ind]
        pixel_losses = pixel_losses[pred < threshold] 
        return pixel_losses.mean()



class OhemSSRCrossEntropy(nn.Module): 
    def __init__(self, ignore_label=-1, thres=0.7, 
        min_kept=100000, weight=None, conv=common.default_conv): 
        super(OhemSSRCrossEntropy, self).__init__() 
        self.thresh = thres
        self.min_kept = max(1, min_kept)
        self.ignore_label = ignore_label 
        self.criterion = nn.CrossEntropyLoss(weight=weight, 
                                             ignore_index=ignore_label, 
                                             reduction='none')
        self.criterion_sr = nn.L1Loss()
        #self.init_weights = self.init_weights()
        #Super-resolution-subnet
        n_resblocks = 16
        n_feats = 64
        kernel_size_sr = 3 
        scale = 4
        act = nn.ReLU(True)
        # define head module
        m_head = [conv(3, n_feats, kernel_size_sr)]
        # define body module
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size_sr, act=act, res_scale=1
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size_sr))
        # define tail module
        self.head1 = nn.Sequential(*m_head)
        self.body1 = nn.Sequential(*m_body)
        self.tail1 = common.Upsampler(conv, scale, n_feats, act=False)
        self.tail2 = conv(n_feats, 3, kernel_size_sr)

        model_dict = self.state_dict()
        pretrained_dict_sr = torch.load('/scratch/prospero/jxie/code/srseg/EDSR/experiment/edsr_baseline_x4/model/model_latest.pt')
        pretrained_dict_sr = {k: v for k, v in pretrained_dict_sr.items()
                           if k in model_dict.keys()}
        #print(pretrained_dict_sr)
        model_dict.update(pretrained_dict_sr)
        self.load_state_dict(model_dict) 
    
    def forward(self, score, target, inimg, **kwargs):
        pb, pc, ph, pw = score.size(0), score.size(1), score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)

        with torch.no_grad():
            x = self.head1(score[:,7:10,:,:])
            res = self.body1(x)
            res += x
            res = self.tail1(res)
            x = self.tail2(res)
            #x = self.tail(res)
            #print(x.dtype)

        F_sr = spatial_pyramid_pool(res, int(pb), [int(512), int(512)], [32, 16, 8, 4, 2, 1])
        F_cls = spatial_pyramid_pool(score[:,10:74,:,:], int(pb), [int(ph), int(pw)], [32, 16, 8, 4, 2, 1])

        similarity_cls = torch.bmm(F_cls.permute(0,2,1), F_cls)
        similarity_sr = torch.bmm(F_sr.permute(0,2,1), F_sr)
        #print(similarity_sr.shape)

        similarity_loss = torch.mean(torch.abs(similarity_cls-similarity_sr))/(1365*1365)

        rh, rw = inimg.size(2), inimg.size(3)
        if ph != rh or pw != rw:
            inimg = F.upsample(
                    input=inimg, size=(ph, pw), mode='nearest')

        if ph != h or pw != w:
            cls = F.upsample(input=score[:,0:7,:,:], size=(h, w), mode='bilinear')
        pred = F.softmax(cls, dim=1)
        pixel_losses = self.criterion(cls, target).contiguous().view(-1)
        mask = target.contiguous().view(-1) != self.ignore_label         
        
        tmp_target = target.clone() 
        tmp_target[tmp_target == self.ignore_label] = 0 
        pred = pred.gather(1, tmp_target.unsqueeze(1)) 
        pred, ind = pred.contiguous().view(-1,)[mask].contiguous().sort()
        min_value = pred[min(self.min_kept, pred.numel() - 1)] 
        threshold = max(min_value, self.thresh) 
        
        pixel_losses = pixel_losses[mask][ind]
        pixel_losses = pixel_losses[pred < threshold] 

        loss = pixel_losses.mean() + 0.1*self.criterion_sr(score[:,7:10,:,:], inimg) + similarity_loss

        return loss



class SSRCrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, weight=None, conv=common.default_conv):
        super(SSRCrossEntropy, self).__init__()
        self.ignore_label = ignore_label
        self.criterion_cls = nn.CrossEntropyLoss(weight=weight, 
                                             ignore_index=ignore_label)
        #self.criterion_cls = focal_loss()
        #self.criterion_cls = GHMC_loss()
        self.criterion_sr = nn.L1Loss()
        #self.init_weights = self.init_weights()
        #Super-resolution-subnet
        n_resblocks = 16
        n_feats = 64
        kernel_size_sr = 3 
        scale = 4
        act = nn.ReLU(True)
        # define head module
        m_head = [conv(3, n_feats, kernel_size_sr)]
        # define body module
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size_sr, act=act, res_scale=1
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size_sr))
        # define tail module
        self.head1 = nn.Sequential(*m_head)
        self.body1 = nn.Sequential(*m_body)
        self.tail1 = common.Upsampler(conv, scale, n_feats, act=False)
        self.tail2 = conv(n_feats, 3, kernel_size_sr)

        model_dict = self.state_dict()
        pretrained_dict_sr = torch.load('/scratch/prospero/jxie/code/srseg/EDSR/experiment/edsr_baseline_x4/model/model_latest.pt')
        pretrained_dict_sr = {k: v for k, v in pretrained_dict_sr.items()
                           if k in model_dict.keys()}
        #print(pretrained_dict_sr)
        model_dict.update(pretrained_dict_sr)
        self.load_state_dict(model_dict)

    def forward(self, score, target, inimg):
        pb, pc, ph, pw = score.size(0), score.size(1), score.size(2), score.size(3)
        #print(score.shape)
        #print(target.shape)
        with torch.no_grad():
            x = self.head1(score[:,7:10,:,:])
            res = self.body1(x)
            res += x
            res = self.tail1(res)
            x = self.tail2(res)
            #x = self.tail(res)
            #print(x.dtype)
        #print(res.shape)
        F_sr = spatial_pyramid_pool(res, int(pb), [int(512), int(512)], [32, 16, 8, 4, 2, 1])
        F_cls = spatial_pyramid_pool(score[:,10:74,:,:], int(pb), [int(ph), int(pw)], [32, 16, 8, 4, 2, 1])

        similarity_cls = torch.bmm(F_cls.permute(0,2,1), F_cls)
        similarity_sr = torch.bmm(F_sr.permute(0,2,1), F_sr)
        #print(similarity_sr.shape)

        similarity_loss = torch.mean(torch.abs(similarity_cls-similarity_sr))/(1365*1365)

        h, w = target.size(1), target.size(2)
        rh, rw = inimg.size(2), inimg.size(3)
        if ph != h or pw != w:
            cls = F.upsample(
                    input=score[:,0:7,:,:], size=(h, w), mode='bilinear')
        if ph != rh or pw != rw:
            inimg = F.upsample(
                    input=inimg, size=(ph, pw), mode='nearest')


        loss = self.criterion_cls(cls, target) + 0.1*self.criterion_sr(score[:,7:10,:,:], inimg) + similarity_loss
        #loss = self.criterion_cls(cls, target)

        return loss
