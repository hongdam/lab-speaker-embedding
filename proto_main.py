from proto_dataloader import MelSpeakerID, MelCollate
from torch.utils.data import  DataLoader

from proto_visualization_lib import save_t_sne
from proto_model import ResNeXt

from datetime import datetime
import numpy as np

import torch

import torch.optim as optim
import torch.nn as nn


def train(num_classes=10, speaker_id_list=None):
    device = torch.device("cuda:1")

    root_dir = '../datasets/NIKL_pre/'
    meta_file = 'metadata_train.csv'
    dataloader = MelSpeakerID(meta_file, root_dir, speaker_id_list)
    collate_fn = MelCollate()

    dataloader = DataLoader(dataloader, batch_size=64,
                            shuffle=True, num_workers=4, collate_fn=collate_fn)

    model = ResNeXt(num_classes=num_classes, pretrained=True).to(device)
    opt = optim.Adam(model.parameters(), lr=0.00005, weight_decay=True)
    loss_fn = nn.CrossEntropyLoss()

    start_time = datetime.now().strftime("%m-%d-%H-%M_")

    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[1, 3, 6], gamma=0.25) # start from 0

    for e in range(10):
        for param_group in opt.param_groups:
            print("epoch {} lr {}".format(e, param_group['lr']))
        train_loss = 0
        correct = 0
        total = 0
        for iteration, batch in enumerate(dataloader):
            mels, labels = batch
            total += len(labels)
            outputs, _ = model(mels.to(device))

            opt.zero_grad()
            loss = loss_fn(outputs, torch.LongTensor(labels).to(device))
            loss.backward()
            opt.step()

            train_loss += loss.data.item()
            _, pred = torch.max(outputs.data, 1)
            correct += (pred == torch.LongTensor(labels).to(device)).sum().item()

            if iteration % 20 == 0:
                print(pred.data[:10].cpu().numpy(), end=' ')
                print('{} [{}/{}] Loss: {:.5f} {:.2f} {}'.format(e, iteration, len(dataloader), train_loss / 20, correct / total *100, correct))
                train_loss = 0
                correct = 0
                total=0

        # torch.save(model.state_dict(), './checkpoint/' + start_time + str(e))

        # inference(model, device,start_time + str(e), num_classes, speaker_id_list)
        scheduler.step()


def inference(model, device, model_name, num_classes=10, speaker_id_list=None):

    root_dir = '../datasets/NIKL_pre/'
    meta_file = 'metadata_test.csv'
    dataloader = MelSpeakerID(meta_file, root_dir, speaker_id_list)
    collate_fn = MelCollate()

    dataloader = DataLoader(dataloader, batch_size=50,
                            shuffle=True, num_workers=4, collate_fn=collate_fn)

    if model is None:
        device = torch.device("cuda:1")
        model = ResNeXt(num_classes=num_classes, pretrained=False).to(device)
        model.load_state_dict(torch.load('./checkpoint/' + model_name))

    model.eval()

    ys = np.array([])
    np_label = []

    for batch in dataloader:
        mel, speaker_id = batch
        out, vec = model(mel.to(device))

        vec = vec.data.cpu().numpy()

        ys = np.vstack([ys, vec]) if ys.size else vec
        np_label.extend(speaker_id)
        print(len(np_label))

    np.save('./infer/test_f_vec.npy', ys)
    np.save('./infer/test_f_label.npy', np_label)

    save_t_sne(model_name)

    model.train()


if __name__ == '__main__':
    # sp_idxs = [0, 1, 2, 4, 22, 81, 82, 83, 85, 80]
    speaker_id_list = [0, 1, 2, 4, 5, 6, 7, 8, 9, 22, 81, 82, 83, 85, 80, 84, 85, 86, 87, 88]
    train(num_classes=20, speaker_id_list=speaker_id_list)
