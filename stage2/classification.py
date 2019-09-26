import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from loader import *
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import cv2
import argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser()
parser.add_argument('--ct_combination_2_path', default='/home/salight/PE/dataset_original/combination-3/ct_numpy/ct_clean', type=str, metavar='PATH1',
                    help='path to ct numpy of PE129')
parser.add_argument('--ct_challenge_path', default='/home/salight/PE/dataset_original/challenge/pe_np_resampled/ct_clean', type=str, metavar='PATH2',
                    help='path to another ct numpy of challenge')
parser.add_argument('--ckpt_save_dir', default='./joint_try', type=str, metavar='ckpt',
                    help='path for you to save ckpts while training')
parser.add_argument('--train_csv', default='/home/salight/PE/reduction-challenge/new/train-pca-3x.csv', type=str, metavar='train_csv',
                    help='path for you to save your inference results on the candidates of stage 1')
parser.add_argument('--pca', default=True, type=bool, metavar='bool_pca',
                    help='must mactch train_csv')
parser.add_argument('--stage_1_prediction_csv', default='/home/salight/PE/reduction-challenge/new/prediction-25-new.csv', type=str, metavar='candidates_csv',
                    help='processed testing candidates with or without pca')
parser.add_argument('--save_csv', default='./after_fp_classification.csv', type=str, metavar='result',
                    help='path for you to save your inference results on the candidates of stage 1')
parser.add_argument('--test', default=0, type=int, metavar='test', help='if test')



def load(df, df_test, ct_combination_2_path, ct_challenge_path, pca=True):
    transformed_train_dataset = PeDataset(df = df, ct_combination_2_path=ct_combination_2_path,ct_challenge_path=ct_challenge_path,
                                          threshold=120, pca=pca,
                                          transform=transforms.Compose([
                                              ToTensor(),
                                              # RandomCrop(24)
                                          ]))

    trainloader = DataLoader(transformed_train_dataset, batch_size=64,
                             shuffle=True, num_workers=4)

    transformed_val_dataset = PeDataset(df = df_test, ct_combination_2_path=ct_combination_2_path,ct_challenge_path=ct_challenge_path,
                                          threshold=120, pca=pca,
                                          transform=transforms.Compose([
                                              ToTensor(),
                                              # RandomCrop(24)
                                          ]))

    valloader = DataLoader(transformed_val_dataset, batch_size=64,
                             shuffle=True, num_workers=4)
    return trainloader,valloader

class MiccaiNet(nn.Module):
    def __init__(self):
        super(MiccaiNet,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(8*8*32, 128)  #24->6
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(-1, 8*8*32)  #24->6
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes,momentum=0.3)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes,momentum=0.3)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)

        return out

class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.in_planes = 16

        self.conv1 = conv3x3(3, 16)
        self.bn1 = nn.BatchNorm2d(16,momentum=0.3)
        self.layer1 = self._make_layer(BasicBlock, 16, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 32, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 64, 2, stride=2)
        self.linear = nn.Linear(64 * BasicBlock.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out

softmax = nn.Sequential(
          nn.Softmax()
        )

criterion = nn.CrossEntropyLoss()


def adjust_learning_rate(optimizer, epoch, init_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = init_lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        print(lr)
        param_group['lr'] = lr

def train(ckpt_save_dir,net,df,df_test, ct_combination_2_path, ct_challenge_path,init_lr,finetune_checkpoint=None, pca=True):
    if not os.path.exists(ckpt_save_dir):
        os.mkdir(ckpt_save_dir)
    losses = []
    recalls = []
    precisions = []
    f1s = []
    max_recall = 0.4
    trainloader, valloader = load(df, df_test,ct_combination_2_path, ct_challenge_path)
    #load checkpoint to finetune
    if finetune_checkpoint is not None:
        net.load_state_dict(torch.load(finetune_checkpoint))
    for epoch in range(200):  # loop over the dataset multiple times
        running_loss = 0.0
        adjust_learning_rate(optimizer, epoch,init_lr)
        for i, data in enumerate(trainloader):
            print("inloader",i)
            # get the inputs
            net.train()
            inputs, labels, names = data['image'],data['label'],data['file_name']
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            if i % 20 == 19:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
                losses.append(running_loss / 100)
                running_loss = 0.0

        if epoch>2:
            correct = 0
            total = 0
            tp = 0
            all_p = 0
            all_pred_p = 0
            net.eval()
            with torch.no_grad():
                for data in trainloader:
                    images, labels, names = data['image'],data['label'],data['file_name']
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = net(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    all_p += (labels == 1).sum().item()
                    all_pred_p += (predicted == 1).sum().item()
                    tp += ((predicted == 1) & (predicted == labels)).sum().item()

            print('train:  recall = %d %%   precision = %d %%' % (tp / all_p * 100, tp / all_pred_p * 100))
            print('Accuracy of the network on the  train images: %d %%' % (100 * correct / total))

    print('Finished Training')

def test(checkpoint, net, df, df_test, ct_combination_2_path, ct_challenge_path, pca=True):

    trainloader, valloader = load(df, df_test, ct_combination_2_path, ct_challenge_path)
    net.load_state_dict(torch.load(checkpoint))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    all_names = []
    all_predicted = []
    all_labels = []

    tp_names = []
    fp_names = []
    tn_names = []
    fn_names = []

    correct = 0
    total = 0
    tp = 0
    all_p = 0
    all_pred_p = 0
    net.eval()

    with torch.no_grad():
        for data in valloader:
            images, labels, names = data['image'], data['label'],data['file_name']
            all_labels.extend(labels)
            all_names.extend(names)
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)

            for i in range(len(images)):
                if predicted[i] == labels[i] and labels[i] == 1:
                    tp_names.append(names[i])
                elif predicted[i] == labels[i] and labels[i] == 0:
                    tn_names.append(names[i])
                elif predicted[i] != labels[i] and labels[i] == 1:
                    fn_names.append(names[i])
                elif predicted[i] != labels[i] and labels[i] == 0:
                    fp_names.append(names[i])

            probabilities = softmax(outputs).cpu().numpy()[:,1].tolist()
            # all_predicted.extend(predicted.cpu().numpy().tolist())
            all_predicted.extend(probabilities)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_p += (labels == 1).sum().item()
            all_pred_p += (predicted == 1).sum().item()
            tp += ((predicted == 1) & (predicted == labels)).sum().item()
        recall = tp / all_p
        precision = tp / all_pred_p
        f1 = 2 * precision * recall / (precision + recall)
        print(tp, all_p, all_pred_p)
        print('prediction:  recall = %d %%   precision = %d %% f1=%d %%' % (recall * 100, precision * 100, f1 * 100))

    df = pd.DataFrame({'file_name':all_names,'pred':all_predicted})
    df.sort_values(by='file_name',inplace=True)
    return df


def show_2D_img(img_save_path,path_2D,tp_names, fp_names, tn_names, fn_names):
    names = os.listdir(path_2D)
    names.sort()
    if not os.path.exists(img_save_path):
        os.mkdir(img_save_path)
        os.mkdir(os.path.join(img_save_path, 'tp'))
        os.mkdir(os.path.join(img_save_path, 'fp'))
        os.mkdir(os.path.join(img_save_path, 'tn'))
        os.mkdir(os.path.join(img_save_path, 'fn'))
    for name in names:
        folder = None
        if name in tp_names:
            folder = 'tp'
        elif name in fp_names:
            folder = 'fp'
        elif name in tn_names:
            folder = 'tn'
        elif name in fn_names:
            folder = 'fn'
        arr = np.load( os.path.join(path_2D,name) )
        cv2.imwrite( os.path.join(img_save_path,folder,name + "-0.jpg"), arr[:, :, 0])
        cv2.imwrite( os.path.join(img_save_path,folder,name + "-1.jpg"), arr[:, :, 1])
        cv2.imwrite( os.path.join(img_save_path,folder,name + "-2.jpg"), arr[:, :, 2])
        print(name,'saved in ',folder)


if __name__ == "__main__":
    args = parser.parse_args()
    ct_combination_2_path = args.ct_combination_2_path
    ct_challenge_path = args.ct_challenge_path
    save_csv = args.save_csv
    train_csv = args.train_csv
    stage_1_prediction_csv = args.stage_1_prediction_csv
    ckpt_save_dir = args.ckpt_save_dir

    net = ResNet18(num_classes=2)
    # net = MiccaiNet()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    learning_rate = 0.005
    optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.99, weight_decay=1e-2)
    net.to(device)
    
    df = pd.read_csv(train_csv)
    df_test = pd.read_csv(stage_1_prediction_csv)
    if args.test == 0:
    	train(ckpt_save_dir, net, df, df_test, ct_combination_2_path, ct_challenge_path, learning_rate, args.pca)
    else:
    	checkpoint = './checkpoints/VGG11/r=0.9383-p=0.0707-f=0.1315-epoch6.pt'
    	df = test(checkpoint,net, df, df_test, ct_combination_2_path, ct_challenge_path, args.pca)
    	df.to_csv(save_csv, index=False)




