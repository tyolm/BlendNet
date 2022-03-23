import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import os
import math
import argparse
import scipy as sp
import scipy.stats

import task_generator as tg
import sys
sys.path.append(os.getcwd())
from net.blend import Blend
from net.conv4 import CNNEncoder


parser = argparse.ArgumentParser(description="One Shot Visual Recognition")
parser.add_argument("-f","--feature_dim",type = int, default = 64)
parser.add_argument("-r","--relation_dim",type = int, default = 8)
parser.add_argument("-w","--class_num",type = int, default = 5)
parser.add_argument("-s","--sample_num_per_class",type = int, default = 1)
parser.add_argument("-b","--batch_num_per_class",type = int, default = 15)
parser.add_argument("-e","--episode",type = int, default= 500000)
parser.add_argument("-t","--test_episode", type = int, default = 600)
parser.add_argument("-l","--learning_rate", type = float, default = 0.001)
parser.add_argument("-g","--gpu",type=int, default=0)
parser.add_argument("-u","--hidden_unit",type=int,default=10)
args = parser.parse_args()

# Hyper Parameters
FEATURE_DIM = args.feature_dim
RELATION_DIM = args.relation_dim
CLASS_NUM = args.class_num
SAMPLE_NUM_PER_CLASS = 1
BATCH_NUM_PER_CLASS = args.batch_num_per_class
EPISODE = args.episode
TEST_EPISODE = args.test_episode
LEARNING_RATE = args.learning_rate
GPU = args.gpu
HIDDEN_UNIT = args.hidden_unit

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m, h

def reduce(features):
    s0 = features.size(0)
    s1 = features.size(1)
    s2 = features.size(2)
    l = int(s0 / CLASS_NUM)
    f = []
    for c in range(CLASS_NUM):
        feature = features[l*c: l*(c+1)].view(l*s1, s2*s2)
        mean = torch.mean(feature, dim=0)
        feature_m = feature - mean.unsqueeze(0).repeat(l*s1, 1)

        feature_mt = torch.transpose(feature_m, 0, 1)
        u, s, v = torch.svd(feature_mt.double(), some=False)
        u = u.float()
        uu = torch.transpose(u[:, :19*19], 0, 1)
        feature_r = torch.matmul(uu, feature_mt) # [15*64, 19*19]
        f.append(feature_r.view(l, s1, 19, 19))

    res = torch.cat([i for i in f])
    return res


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())

def main():
    acc_pool = []
    # Step 1: init data folders
    print("init data folders")
    # init character folders for dataset construction
    metatrain_folders,metatest_folders = tg.mini_imagenet_folders()

    # Step 2: init neural networks
    print("init neural networks")

    feature_encoder = CNNEncoder()
    relation_network = Blend()

    feature_encoder.apply(weights_init)
    relation_network.apply(weights_init)

    feature_encoder.cuda(GPU)
    relation_network.cuda(GPU)

    feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(),lr=LEARNING_RATE)
    feature_encoder_scheduler = StepLR(feature_encoder_optim,step_size=100000,gamma=0.5)
    relation_network_optim = torch.optim.Adam(relation_network.parameters(),lr=LEARNING_RATE)
    relation_network_scheduler = StepLR(relation_network_optim,step_size=100000,gamma=0.5)

    if os.path.exists(str("D:\\nut\\code\\blendnet_win\\model\\miniimagenet\\mini_conv_aug_blend_encoder_" + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")):
        feature_encoder.load_state_dict(torch.load(str("D:\\nut\\code\\blendnet_win\\model\\miniimagenet\\mini_conv_aug_blend_encoder_" + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")))
        print("load feature encoder success")
    if os.path.exists(str("D:\\nut\\code\\blendnet_win\\model\\miniimagenet\\mini_conv_aug_blendnet_"+ str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")):
        relation_network.load_state_dict(torch.load(str("D:\\nut\\code\blendnet_win\\model\\miniimagenet\\mini_conv_aug_blendnet_"+ str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")))
        print("load relation network success")

    # Step 3: build graph
    print("Training...")

    last_accuracy = 0.0

    for episode in range(EPISODE):

        feature_encoder_scheduler.step(episode)
        relation_network_scheduler.step(episode)

        # init dataset
        # sample_dataloader is to obtain previous samples for compare
        # batch_dataloader is to batch samples for training
        task = tg.MiniImagenetTask(metatrain_folders,CLASS_NUM,SAMPLE_NUM_PER_CLASS,BATCH_NUM_PER_CLASS)
        sample_dataloader = tg.get_mini_imagenet_data_loader(task,num_per_class=SAMPLE_NUM_PER_CLASS,split="train",shuffle=False)
        batch_dataloader = tg.get_mini_imagenet_data_loader(task,num_per_class=BATCH_NUM_PER_CLASS,split="test",shuffle=True)

        # sample datas
        samples,sample_labels = sample_dataloader.__iter__().next()
        batches,batch_labels = batch_dataloader.__iter__().next()

        # image augment
        import imgaug.augmenters as iaa
        seq = iaa.Sequential([
            iaa.Fliplr(1),
            iaa.CropAndPad(percent=(-0.25, 0.25)),
            iaa.Rotate((-20, 20))
        ])
        batches_np = batches.numpy().transpose(0, 2, 3, 1)
        batches_aug_img = seq.augment_images(images=batches_np)
        batches_aug = torch.from_numpy(batches_aug_img.transpose(0, 3, 1, 2))

        # calculate features
        # sample_features = feature_encoder(Variable(samples).cuda(GPU)) # [5, 64, 42, 42]
        # batch_features = feature_encoder(Variable(batches).cuda(GPU)) # [75, 64, 42, 42]
        smp_f1, smp_f2, smp_f3, smp_f4 = feature_encoder(Variable(samples).cuda(GPU))
        bco_f1, bco_f2, bco_f3, bco_f4 = feature_encoder(Variable(batches).cuda(GPU))
        bca_f1, bca_f2, bca_f3, bca_f4 = feature_encoder(Variable(batches_aug).cuda(GPU))
        bc_f1 = (bco_f1 + bca_f1) / 2
        bc_f2 = (bco_f2 + bca_f2) / 2
        bc_f3 = (bco_f3 + bca_f3) / 2
        bc_f4 = (bco_f4 + bca_f4) / 2


#         print('sample_feature', sample_features.shape, 'batch_features', batch_features.shape)

        # calculate relations
        # each batch sample link to every samples to calculate relations
        # to form a 100x128 matrix for relation network
        # sample_features_ext = sample_features.unsqueeze(0).repeat(BATCH_NUM_PER_CLASS*CLASS_NUM,1,1,1,1) # [75, 5, 64, 19, 19]
        # batch_features_ext = batch_features.unsqueeze(0).repeat(SAMPLE_NUM_PER_CLASS*CLASS_NUM,1,1,1,1) # [5, 75, 64, 19, 19]
        # batch_features_ext = torch.transpose(batch_features_ext,0,1) # [75, 5, 64, 19, 19]
        # relation_pairs = torch.cat((sample_features_ext,batch_features_ext),2).view(-1,FEATURE_DIM*2,19,19) # [375, 128, 19, 19]
        # relations = relation_network(relation_pairs).view(-1,CLASS_NUM*SAMPLE_NUM_PER_CLASS) # [75, 5]


        # mse = nn.MSELoss().cuda(GPU)
        # one_hot_labels = Variable(torch.zeros(BATCH_NUM_PER_CLASS*CLASS_NUM, CLASS_NUM).scatter_(1, batch_labels.view(-1,1).long(), 1)).cuda(GPU)
        # loss = mse(relations,one_hot_labels)


        mse = nn.MSELoss().cuda(GPU)
        one_hot_labels = Variable(torch.zeros(BATCH_NUM_PER_CLASS*CLASS_NUM, CLASS_NUM).scatter_(1, batch_labels.view(-1,1).long(), 1)).cuda(GPU)

        smp_f1_ext = smp_f1.unsqueeze(0).repeat(BATCH_NUM_PER_CLASS*CLASS_NUM,1,1,1,1)
        bc_f1_ext = bc_f1.unsqueeze(0).repeat(SAMPLE_NUM_PER_CLASS*CLASS_NUM,1,1,1,1)
        bc_f1_ext = torch.transpose(bc_f1_ext,0,1)
        relation_pairs1 = torch.cat((smp_f1_ext, bc_f1_ext), 2).view(-1, FEATURE_DIM*2, 42, 42)
        relations1 = relation_network(relation_pairs1).view(-1,CLASS_NUM*SAMPLE_NUM_PER_CLASS)

        loss1 = mse(relations1, one_hot_labels)


        smp_f2_ext = smp_f2.unsqueeze(0).repeat(BATCH_NUM_PER_CLASS*CLASS_NUM,1,1,1,1)
        bc_f2_ext = bc_f2.unsqueeze(0).repeat(SAMPLE_NUM_PER_CLASS*CLASS_NUM,1,1,1,1)
        bc_f2_ext = torch.transpose(bc_f2_ext,0,1)
        relation_pairs2 = torch.cat((smp_f2_ext, bc_f2_ext), 2).view(-1, FEATURE_DIM*2, 21, 21)
        
        relations2 = relation_network(relation_pairs2).view(-1,CLASS_NUM*SAMPLE_NUM_PER_CLASS)

        loss2 = mse(relations2, one_hot_labels)


        smp_f3_ext = smp_f3.unsqueeze(0).repeat(BATCH_NUM_PER_CLASS*CLASS_NUM,1,1,1,1)
        bc_f3_ext = bc_f3.unsqueeze(0).repeat(SAMPLE_NUM_PER_CLASS*CLASS_NUM,1,1,1,1)
        bc_f3_ext = torch.transpose(bc_f3_ext,0,1)
        relation_pairs3 = torch.cat((smp_f3_ext, bc_f3_ext), 2).view(-1, FEATURE_DIM*2, 10, 10)
        relations3 = relation_network(relation_pairs3).view(-1,CLASS_NUM*SAMPLE_NUM_PER_CLASS)

        loss3 = mse(relations3, one_hot_labels)


        smp_f4_ext = smp_f4.unsqueeze(0).repeat(BATCH_NUM_PER_CLASS*CLASS_NUM,1,1,1,1)
        bc_f4_ext = bc_f4.unsqueeze(0).repeat(SAMPLE_NUM_PER_CLASS*CLASS_NUM,1,1,1,1)
        bc_f4_ext = torch.transpose(bc_f4_ext,0,1)
        relation_pairs4 = torch.cat((smp_f4_ext, bc_f4_ext), 2).view(-1, FEATURE_DIM*2, 5, 5)
        relations4 = relation_network(relation_pairs4).view(-1,CLASS_NUM*SAMPLE_NUM_PER_CLASS)

        loss4 = mse(relations4, one_hot_labels)


        loss = loss1 + loss2 + loss3 + loss4


        # training

        feature_encoder.zero_grad()
        relation_network.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm(feature_encoder.parameters(),0.5)
        torch.nn.utils.clip_grad_norm(relation_network.parameters(),0.5)

        feature_encoder_optim.step()
        relation_network_optim.step()


        if (episode+1)%100 == 0:
                print("episode:",episode+1,"loss",loss.item())

        if episode%5000 == 0:

            # test
            print("Testing...")
            accuracies = []
            for i in range(TEST_EPISODE):
                total_rewards = 0
                counter = 0
                task = tg.MiniImagenetTask(metatest_folders,CLASS_NUM,1,15)
                sample_dataloader = tg.get_mini_imagenet_data_loader(task,num_per_class=1,split="train",shuffle=False)

                num_per_class = 3
                test_dataloader = tg.get_mini_imagenet_data_loader(task,num_per_class=num_per_class,split="test",shuffle=True)
                sample_images,sample_labels = sample_dataloader.__iter__().next()
                for test_images,test_labels in test_dataloader:
                    batch_size = test_labels.shape[0]
                    # calculate features
                    # sample_features = feature_encoder(Variable(sample_images).cuda(GPU)) # 5x64
                    # test_features = feature_encoder(Variable(test_images).cuda(GPU)) # 20x64

                    # test image augment
                    test_images_np = test_images.numpy().transpose(0, 2, 3, 1)
                    test_aug_img = seq.augment_images(images=test_images_np)
                    test_images_aug = torch.from_numpy(test_aug_img.transpose(0, 3, 1, 2))

                    smp_f1, smp_f2, smp_f3, smp_f4 = feature_encoder(Variable(sample_images).cuda(GPU))
                    tso_f1, tso_f2, tso_f3, tso_f4 = feature_encoder(Variable(test_images).cuda(GPU))
                    tsa_f1, tsa_f2, tsa_f3, tsa_f4 = feature_encoder(Variable(test_images_aug).cuda(GPU))

                    ts_f1 = (tso_f1 + tsa_f1) / 2
                    ts_f2 = (tso_f2 + tsa_f2) / 2
                    ts_f3 = (tso_f3 + tsa_f3) / 2
                    ts_f4 = (tso_f4 + tsa_f4) / 2

                    # calculate relations
                    # each batch sample link to every samples to calculate relations
                    # to form a 100x128 matrix for relation network
                    # sample_features_ext = sample_features.unsqueeze(0).repeat(batch_size,1,1,1,1)
                    # test_features_ext = test_features.unsqueeze(0).repeat(1*CLASS_NUM,1,1,1,1)
                    # test_features_ext = torch.transpose(test_features_ext,0,1)


                    # relation_pairs = torch.cat((sample_features_ext,test_features_ext),2).view(-1,FEATURE_DIM*2,19,19)
                    # relations = relation_network(relation_pairs).view(-1,CLASS_NUM)

                    smp_f1_ext = smp_f1.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
                    ts_f1_ext = ts_f1.unsqueeze(0).repeat(1*CLASS_NUM, 1, 1, 1, 1)
                    ts_f1_ext = torch.transpose(ts_f1_ext, 0, 1)

                    relation_pairs1 = torch.cat((smp_f1_ext, ts_f1_ext), 2).view(-1, FEATURE_DIM*2, 42, 42)
                    relations1 = relation_network(relation_pairs1).view(-1, CLASS_NUM)


                    smp_f2_ext = smp_f2.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
                    ts_f2_ext = ts_f2.unsqueeze(0).repeat(1*CLASS_NUM, 1, 1, 1, 1)
                    ts_f2_ext = torch.transpose(ts_f2_ext, 0, 1)

                    relation_pairs2 = torch.cat((smp_f2_ext, ts_f2_ext), 2).view(-1, FEATURE_DIM*2, 21, 21)
                    relations2 = relation_network(relation_pairs2).view(-1, CLASS_NUM)


                    smp_f3_ext = smp_f3.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
                    ts_f3_ext = ts_f3.unsqueeze(0).repeat(1*CLASS_NUM, 1, 1, 1, 1)
                    ts_f3_ext = torch.transpose(ts_f3_ext, 0, 1)

                    relation_pairs3 = torch.cat((smp_f3_ext, ts_f3_ext), 2).view(-1, FEATURE_DIM*2, 10, 10)
                    relations3 = relation_network(relation_pairs3).view(-1, CLASS_NUM)


                    smp_f4_ext = smp_f4.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
                    ts_f4_ext = ts_f4.unsqueeze(0).repeat(1*CLASS_NUM, 1, 1, 1, 1)
                    ts_f4_ext = torch.transpose(ts_f4_ext, 0, 1)

                    relation_pairs4 = torch.cat((smp_f4_ext, ts_f4_ext), 2).view(-1, FEATURE_DIM*2, 5, 5)
                    relations4 = relation_network(relation_pairs4).view(-1, CLASS_NUM)


                    relations_sum = torch.add(torch.add(torch.add(relations1, relations2), relations3), relations4)
                    relations_mean = relations_sum / 4

                    _,predict_labels = torch.max(relations_mean.data,1)

                    rewards = [1 if predict_labels[j]==test_labels[j] else 0 for j in range(batch_size)]

                    total_rewards += np.sum(rewards)
                    counter += batch_size
                accuracy = total_rewards/1.0/counter
                accuracies.append(accuracy)

            test_accuracy,h = mean_confidence_interval(accuracies)
            acc_pool.append(test_accuracy)
            # acc_pool.append(h)

            print("test accuracy:",test_accuracy,"h:",h)

            if test_accuracy > last_accuracy:

                # save networks
                torch.save(feature_encoder.state_dict(),str("D:\\nut\\code\\blendnet_win\\model\\miniimagenet\\mini_conv_aug_blend_encoder_" + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl"))
                torch.save(relation_network.state_dict(),str("D:\\nut\\code\\blendnet_win\\model\\miniimagenet\\mini_conv_aug_blendnet_"+ str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl"))

                print("save networks for episode:",episode)

                last_accuracy = test_accuracy

    print(acc_pool)
    print(last_accuracy)


if __name__ == '__main__':
    main()

