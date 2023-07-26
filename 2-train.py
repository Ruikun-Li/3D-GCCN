# Train for 3D-GCNN
# Our method is backbone-agnostic, the encoder and decoder of CNN part can be replaced by any more powerful segmentation network.
# Ruikun Li

import os
import time
import numpy as np
import torch.nn as nn
import torch
from torch.utils import data
from net_model import Encoder3D, GAT3D, Decoder3D, trans_cnn_feature
from utils import cal_dice_loss, NewDiceLoss, MyDataset, check_and_create_path, Logger
import sys
sys.path.append('./')

data_dir = '../Image_folder/'
label_dir = '../Label_folder/'
graph_dir = '3d_graph/'
checkpoint_dir = 'checkpoints/temp/'
encoder_dir = checkpoint_dir + 'encoder/'
gnn_dir = checkpoint_dir + 'gnn/'
decoder_dir = checkpoint_dir + 'decoder/'
check_and_create_path(encoder_dir)
check_and_create_path(gnn_dir)
check_and_create_path(decoder_dir)
GPU0 = 'cuda:0'
GPU1 = 'cuda:0'
OLD_EPOCH = 0
num_class = 1
sys.stdout = Logger(checkpoint_dir + 'log.txt')


def train_net(encoder, gnn, decoder, epochs=500, batch_size=1, learning_rate=0.001, fold_index=0, fold_num=4):
    # Extract every fold from total list
    patient_list = os.listdir(data_dir)
    train_list = []
    val_list = []
    for index_patient, patient in enumerate(patient_list):
        if index_patient % fold_num != fold_index:
            train_list.append(patient)
        else:
            val_list.append(patient)
    print(len(train_list), len(val_list), val_list)

    train_set = MyDataset(train_list, data_dir, label_dir, graph_dir, valid_flag=False)
    train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)
    valid_set = MyDataset(val_list, data_dir, label_dir, graph_dir, valid_flag=True)
    valid_loader = data.DataLoader(valid_set, batch_size=1, shuffle=False, num_workers=4)

    optimizer = torch.optim.Adam(
        [{'params': encoder.parameters()}, {'params': gnn.parameters()}, {'params': decoder.parameters()}],
        lr=learning_rate, weight_decay=0.0005)

    criterion = NewDiceLoss()
    # criterion2 = nn.BCELoss()
    criterion_gnn = nn.BCELoss()

    print('''Starting training:
            Epochs:{}
            Batch size:{}
            Learning rate:{}
            Training size:{}
            Validation size:{}
            '''.format(epochs, batch_size, learning_rate, len(train_list), len(val_list)))

    best_val_loss = np.inf
    best_encoder_path = 'checkpoints/best_encoder-'
    best_gnn_path = 'checkpoints/best_gnn-'
    best_decoder_path = 'checkpoints/best_decoder-'
    bad_epoch = 0
    for epoch in range(OLD_EPOCH, OLD_EPOCH + epochs):
        time1 = time.time()
        # Train
        encoder.train()
        gnn.train()
        decoder.train()
        # Set learning rate
        training_lr = learning_rate * (0.7 ** (epoch // 50))
        training_lr_gnn = learning_rate * (0.7 ** (epoch // 50))
        for index, param_group in enumerate(optimizer.param_groups):
            if index == 1:  # 1 means gnn
                param_group['lr'] = training_lr_gnn
            else:
                param_group['lr'] = training_lr

        train_loss_gnn = np.zeros([num_class])
        train_loss_cnn = np.zeros([num_class])
        count = 0
        for index, (img, label, patch_nodes, adj_gnn, label_gnn) in enumerate(train_loader):
            img = img.to(GPU0)
            label = label.to(GPU1)
            adj_gnn = adj_gnn.to(GPU1)
            label_gnn = label_gnn.to(GPU1)

            f1, f2, f3, f4 = encoder(img)
            f1 = f1.to(GPU1)
            f2 = f2.to(GPU1)
            f3 = f3.to(GPU1)
            f4 = f4.to(GPU1)
            out_cnn, f_cnn = decoder(f1, f2, f3, f4)
            graph_gnn = trans_cnn_feature(f_cnn, patch_nodes)
            graph_gnn = graph_gnn.to(GPU1)
            out_gnn, _ = gnn(graph_gnn, adj_gnn)
            del f1, f2, f3, f4, img, graph_gnn, adj_gnn, f_cnn

            # Caculate the losses
            out_gnn = torch.transpose(out_gnn, 1, 2)
            label_gnn = torch.transpose(label_gnn, 1, 2)
            loss_i_gnn = criterion_gnn(out_gnn, label_gnn)
            train_loss_gnn[0] += loss_i_gnn.item()

            loss_i_cnn = criterion(out_cnn, label, beta=1)
            train_loss_cnn[0] += loss_i_cnn.item()

            loss = loss_i_cnn + loss_i_gnn

            count += 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            del loss, label, label_gnn, out_gnn, out_cnn
        train_loss_gnn = train_loss_gnn / count
        train_loss_cnn = train_loss_cnn / count

        # Validation
        encoder.eval()
        gnn.eval()
        decoder.eval()
        val_loss_cnn = np.zeros([num_class])
        count = 0
        for index, (img, label) in enumerate(valid_loader):
            # original version 0
            img.requires_grad_(requires_grad=False)
            img = img.to(GPU0)

            f1, f2, f3, f4 = encoder(img)
            f1 = f1.to(GPU1)
            f2 = f2.to(GPU1)
            f3 = f3.to(GPU1)
            f4 = f4.to(GPU1)
            out_cnn, _ = decoder(f1, f2, f3, f4)
            out_cnn = out_cnn.squeeze().cpu().detach().numpy()
            del f1, f2, f3, f4, img

            if np.max(out_cnn) > 1:
                print('11111111111111111111111111111111111')
            out_cnn = (out_cnn > 0.5).astype(np.uint8)
            label = label.squeeze().cpu().detach().numpy()
            val_loss_cnn[0] += cal_dice_loss(out_cnn, label)

            count += 1
            del label, out_cnn
        val_loss_cnn = val_loss_cnn / count
        time_cost = time.time() - time1

        # Print epoch loss
        np.set_printoptions(precision=3)
        print('Epoch {0}/{1} : Train:{2:.4f}, {3:.4f}, Valid: {4:.4f}'
              .format(epoch + 1, epochs + OLD_EPOCH, np.mean(train_loss_gnn), np.mean(train_loss_cnn), np.mean(val_loss_cnn)), time_cost)

        # Save the model
        val_loss = np.mean(val_loss_cnn)
        if np.mean(train_loss_cnn) * 0.5 < val_loss < best_val_loss:
            print('  Update...')
            if os.path.exists(best_encoder_path):
                os.remove(best_encoder_path)
            if os.path.exists(best_gnn_path):
                os.remove(best_gnn_path)
            if os.path.exists(best_decoder_path):
                os.remove(best_decoder_path)
            best_encoder_path = checkpoint_dir + 'best_cnn_encoder-' + '{0:.4f}.pth'.format(1 - val_loss)
            best_gnn_path = checkpoint_dir + 'best_gnn-' + '{0:.4f}.pth'.format(1 - val_loss)
            best_decoder_path = checkpoint_dir + 'best_cnn_decoder-' + '{0:.4f}.pth'.format(1 - val_loss)
            torch.save(encoder.state_dict(), best_encoder_path)
            torch.save(gnn.state_dict(), best_gnn_path)
            torch.save(decoder.state_dict(), best_decoder_path)
            best_val_loss = val_loss
            bad_epoch = 0
        else:
            bad_epoch += 1
            if bad_epoch == 100:
                print('******EXIT******: No improvement for 100 epochs!')
                sys.exit(0)
        if (epoch + 1) % 10 == 0:
            torch.save(encoder.state_dict(), encoder_dir + '{}.pth'.format(epoch + 1))
            torch.save(gnn.state_dict(), gnn_dir + '{}.pth'.format(epoch + 1))
            torch.save(decoder.state_dict(), decoder_dir + '{}.pth'.format(epoch + 1))
            print('Model {} saved.'.format(epoch + 1))


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True

    gccn_encoder = Encoder3D(channel_list=[1, 8, 16, 32, 64])
    gnn_net = GAT3D(8, 8, 1, dropout=0.2, alpha=0.2, n_heads=3)
    gccn_decoder = Decoder3D(channel_list=[64, 32, 16, 8, 1])

    gccn_encoder = gccn_encoder.to(GPU0)
    gnn_net = gnn_net.to(GPU1)
    gccn_decoder = gccn_decoder.to(GPU1)

    gccn_encoder_path = encoder_dir + str(OLD_EPOCH) + '.pth'
    gnn_path = gnn_dir + str(OLD_EPOCH) + '.pth'
    gccn_decoder_path = decoder_dir + str(OLD_EPOCH) + '.pth'
    if os.path.exists(gccn_encoder_path):
        gccn_encoder.load_state_dict(torch.load(gccn_encoder_path, map_location='cuda:1'))
        gnn_net.load_state_dict(torch.load(gnn_path, map_location='cuda:1'))
        gccn_decoder.load_state_dict(torch.load(gccn_decoder_path, map_location='cuda:1'))
        print('Model loaded from {} and {} and {}.'.format(gccn_encoder_path, gnn_path, gccn_decoder_path))
    else:
        print('Building a new model...')

    train_net(gccn_encoder, gnn_net, gccn_decoder)
    print('Train finished!\n')

