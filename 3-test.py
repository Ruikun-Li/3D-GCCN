# Test for 3D-GCNN
# Ruikun Li

import os
import time
import numpy as np
import torch
import SimpleITK as sitk
from net_model import Encoder3D, Decoder3D
from utils import check_and_create_path, load_data, hausdorff_95
import sys
sys.path.append('./')

test_dir = '../Image_folder/'
label_dir = '../Label_folder/'
out_dir = 'prediction/temp/'
check_and_create_path(out_dir)
encoder_path = 'checkpoints/temp/best_cnn_encoder-0.xxxx.pth'
decoder_path = 'checkpoints/temp/best_cnn_decoder-0.xxxx.pth'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
USE_GPU = 1
num_class = 1


if __name__ == '__main__':
    patient_list = os.listdir(test_dir)
    test_list = []
    for index in range(len(patient_list)):
        if index % 4 == 0:
            test_list.append(patient_list[index])
    print(test_list)
    print('Test on {} images.'.format(len(test_list)))

    net_encoder = Encoder3D(channel_list=[1, 8, 16, 32, 64])
    net_decoder = Decoder3D(channel_list=[64, 32, 16, 8, num_class])

    if USE_GPU:
        net_encoder.cuda()
        net_decoder.cuda()
        net_encoder.load_state_dict(torch.load(encoder_path, map_location='cuda:0'))
        net_decoder.load_state_dict(torch.load(decoder_path, map_location='cuda:0'))
        print('Model loaded with GPU.')

    net_encoder.eval()
    net_decoder.eval()

    total_loss = 0
    total_acc = 0
    total_sen = 0
    total_spe = 0
    total_hd95 = 0
    total_loss2 = 0
    total_acc2 = 0
    total_sen2 = 0
    total_spe2 = 0
    total_hd952 = 0
    total_time = 0
    for test_file in test_list:
        time000 = time.time()
        print('Predicting', test_file)
        data_path = test_dir + test_file + '/patient.nii.gz'
        label_path = label_dir + test_file + '/hepaticvessel.nii.gz'

        # load the data
        data = load_data(data_path)
        ori_data = sitk.ReadImage(data_path)
        ori_label = sitk.ReadImage(label_path)
        label = sitk.GetArrayFromImage(ori_label)

        # padding
        z_size = data.shape[0]
        padding_z = (16 - z_size % 16) % 16
        data = np.pad(data, ((padding_z // 2, padding_z - padding_z // 2), (0, 0), (0, 0)), 'constant')
        label = np.pad(label, ((padding_z // 2, padding_z - padding_z // 2), (0, 0), (0, 0)), 'constant')
        y_size = data.shape[1]
        padding_y = (16 - y_size % 16) % 16
        data = np.pad(data, ((0, 0), (padding_y // 2, padding_y - padding_y // 2), (0, 0)), 'constant')
        label = np.pad(label, ((0, 0), (padding_y // 2, padding_y - padding_y // 2), (0, 0)), 'constant')
        x_size = data.shape[2]
        padding_x = (16 - x_size % 16) % 16
        data = np.pad(data, ((0, 0), (0, 0), (padding_x // 2, padding_x - padding_x // 2)), 'constant')
        label = np.pad(label, ((0, 0), (0, 0), (padding_x // 2, padding_x - padding_x // 2)), 'constant')
        # print(data.shape)

        data = data[np.newaxis, np.newaxis, :, :, :].astype(np.float32)
        label = (label == 1).astype(np.float32)
        data = torch.from_numpy(data)
        data = data.cuda()

        feature1, feature2, feature3, feature4 = net_encoder(data)
        out, _ = net_decoder(feature1, feature2, feature3, feature4)
        del feature1, feature2, feature3, feature4

        pred_mask = out.squeeze().cpu().detach().numpy()

        pred_label = (pred_mask > 0.5).astype(np.uint8)
        del data, out, pred_mask
        total_time += time.time() - time000

        tp = np.sum(pred_label * label)
        tn = np.sum((1 - pred_label) * (1 - label))
        fp = np.sum(pred_label * (1 - label))
        fn = np.sum((1 - pred_label) * label)
        loss = 1 - ((2 * tp + 10e-4) / (2 * tp + fp + fn + 10e-4))
        acc = (tp + tn + 10e-4) / (tp + tn + fp + fn + 10e-4)
        sen = (tp + 10e-4) / (tp + fn + 10e-4)
        spe = (tn + 10e-4) / (tn + fp + 10e-4)
        hd95 = hausdorff_95(pred_label, label, spacing=np.array(ori_data.GetSpacing()[::-1]))
        print('before', loss, tp, tn, fp, fn, acc, sen, spe, hd95)
        total_loss += loss
        total_acc += acc
        total_sen += sen
        total_spe += spe
        total_hd95 += hd95

        # Post processing: This should be adjusted according to specific task
        from skimage import measure
        all_labels = measure.label(pred_label, background=0, connectivity=3)
        properties = measure.regionprops(all_labels)
        areas = np.array([prop.area for prop in properties])
        final_label = np.zeros(all_labels.shape)
        for i, area in enumerate(areas):
            if area > 200:  # This should be adjusted according to specific task
                final_label += (all_labels == (i + 1)).astype(np.float32)
        tp = np.sum(final_label * label)
        tn = np.sum((1 - final_label) * (1 - label))
        fp = np.sum(final_label * (1 - label))
        fn = np.sum((1 - final_label) * label)
        loss2 = 1 - ((2 * tp + 10e-4) / (2 * tp + fp + fn + 10e-4))
        acc2 = (tp + tn + 10e-4) / (tp + tn + fp + fn + 10e-4)
        sen2 = (tp + 10e-4) / (tp + fn + 10e-4)
        spe2 = (tn + 10e-4) / (tn + fp + 10e-4)
        hd952 = hausdorff_95(final_label, label)
        print('after', loss2, tp, tn, fp, fn, acc2, sen2, spe2, hd952, '\n')
        total_loss2 += loss2
        total_acc2 += acc2
        total_sen2 += sen2
        total_spe2 += spe2
        total_hd952 += hd952

        out_data = final_label
        out_path = out_dir + test_file + '_' + str(loss2)[:6] + '_' + str(hd952)[:7] + '.nii.gz'
        out_data = sitk.GetImageFromArray(out_data)
        out_data.SetSpacing(ori_data.GetSpacing())
        sitk.WriteImage(out_data, out_path)

    print('Dice:', 1 - total_loss / len(test_list))
    print('Acc:', total_acc / len(test_list))
    print('Sen:', total_sen / len(test_list))
    print('Spe:', total_spe / len(test_list))
    print('HD95:', total_hd95 / len(test_list))
    print(' ')
    print('Dice after post:', 1 - total_loss2 / len(test_list))
    print('Acc after post:', total_acc2 / len(test_list))
    print('Sen after post:', total_sen2 / len(test_list))
    print('Spe after post:', total_spe2 / len(test_list))
    print('HD95 after post:', total_hd952 / len(test_list))
    print(' ')
    print('Average Time:', total_time / len(test_list))
