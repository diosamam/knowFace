import torch
import torch.nn as nn
from torch import optim, tensor
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image
import PIL.ImageOps


import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

#定义一些超参
train_batch_size = 32        #训练时batch_size批量大小
train_number_epochs = 50     #训练的epoch迭代次数
# train_number_epochs = 5    #训练的epoch迭代次数


def show_plot(iteration,loss):
    #绘制损失变化图
    plt.plot(iteration,loss)
    plt.savefig('./ORL_faces_pytorch-main/src/data/orl_faces/subject01/result_2.png')


class SiameseNetworkDataset_test(Dataset):
    def __init__(self, imageFolderDataset, transform=None, should_invert=True):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform
        self.should_invert = should_invert

    def __getitem__(self, index):
        img_tuple = self.imageFolderDataset.imgs[index]

        img = Image.open(img_tuple[0]).convert("L")
        if self.should_invert:
            img = PIL.ImageOps.invert(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, img_tuple[1]

    def __len__(self):
        return len(self.imageFolderDataset.imgs)



    # def __init__(self, imageFolderDataset, transform=None, should_invert=True):
    #     self.imageFolderDataset = imageFolderDataset
    #     self.transform = transform
    #     self.should_invert = should_invert
    #
    # def __getitem__(self, index):
    #     img0_tuple = random.choice(self.imageFolderDataset.imgs)  # Randomly choose one image
    #     img1_tuple = random.choice(self.get_other_images(img0_tuple))  # Randomly choose another image from the remaining
    #
    #     img0 = Image.open(img0_tuple[0]).convert("L")  # Load and convert the first image to grayscale
    #     img1 = Image.open(img1_tuple[0]).convert("L")  # Load and convert the second image to grayscale
    #
    #
    #     label = int(img1_tuple[1] != img0_tuple[1])  # Calculate the label based on whether the images belong to the same class
    #
    #     return img0, img1, torch.tensor([label], dtype=torch.float32), img0_tuple[1], img1_tuple[1]
    #
    # def __len__(self):
    #     return len(self.imageFolderDataset.imgs)
    #
    # def get_other_images(self, img_tuple):
    #     # Get a list of other images (29 images) excluding the current image
    #     other_images = [img for img in self.imageFolderDataset.imgs if img != img_tuple]
    #     return other_images[:29]  # Return only 29 images (excluding the current image)
# ---------------------------------------------------
#
# import os
# import random
# import torchvision.datasets as datasets
#
# # Set the path to the ORL dataset directory
# data_dir = "./ORL_faces_pytorch-main/src/data/orl/"
#
# # Define the number of training images per subject
# num_train_images = 7
#
# # Create empty lists for training and testing datasets
# train_dataset = []
# test_dataset = []
#
# # Iterate over each subject in the dataset
# for subject in os.listdir(data_dir):
#     subject_dir = os.path.join(data_dir, subject)
#
#     # Get the list of image files for the current subject
#     image_files = os.listdir(subject_dir)
#
#     # Shuffle the image files randomly
#     random.shuffle(image_files)
#
#     # Split the image files into training and testing sets
#     train_images = image_files[:num_train_images]
#     test_images = image_files[num_train_images:]
#
#     # Add the image file paths to the respective datasets
#     for train_image in train_images:
#         train_dataset.append((os.path.join(subject_dir, train_image), int(subject[1:])))
#     for test_image in test_images:
#         test_dataset.append((os.path.join(subject_dir, test_image), int(subject[1:])))
#
# # Create the training and testing ImageFolder datasets
# train_data = datasets.ImageFolder(data_dir, train_dataset)
# test_data = datasets.ImageFolder(data_dir, test_dataset)
#
# # Print the number of samples in the training and testing datasets
# print("Number of training samples:", len(train_data))
# print("Number of testing samples:", len(test_data))


# ---------------------------------------------------
# import torch
# import torch.nn as nn
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader, Dataset
# from PIL import Image
import os

# Define the dataset class
# class ORLFacesDataset(Dataset):
#     def __init__(self, data_dir, train=True, transform=None):
#         self.data_dir = data_dir
#         self.train = train
#         self.transform = transform
#         self.images = []
#
#         self._load_dataset()
#
#     def _load_dataset(self):
#         for person in range(1, 41):  # There are 40 people in the dataset
#             person_dir = os.path.join(self.data_dir, f"s{person}")
#             person_images = sorted(os.listdir(person_dir))
#             if self.train:
#                 self.images.extend([os.path.join(person_dir, img) for img in person_images[:7]])
#             else:
#                 self.images.extend([os.path.join(person_dir, img) for img in person_images[:3]])
#
#     def __getitem__(self, index):
#         image_path = self.images[index]
#         img = Image.open(image_path).convert("L")
#
#         if self.transform is not None:
#             img = self.transform(img)
#
#         label = int(image_path.split("/")[-2].replace("s", ""))  # Extract the label from the image path
#
#         return img, label
#
#     def __len__(self):
#         return len(self.images)


# Set the data directory and create the transforms
# data_dir = "./ORL_faces_pytorch-main/src/data/orl/"
# transform = transforms.Compose([
#     transforms.Resize((100, 100)),
#     transforms.ToTensor()
# ])
#
# # Create the training and testing datasets
# train_dataset = ORLFacesDataset(data_dir, train=True, transform=transform)
# test_dataset = ORLFacesDataset(data_dir, train=False, transform=transform)
#
# # Create the data loaders
# train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
#
# # Print the number of training and testing samples
# print("Number of training samples:", len(train_dataset))
# print("Number of testing samples:", len(test_dataset))


# ---------------------------------------------------

# 自定义Dataset类，__getitem__(self,index)每次返回(img1, img2, 0/1,img1的类别, img2的类别)
class SiameseNetworkDataset(Dataset):

    def __init__(self, imageFolderDataset, transform=None, should_invert=True):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform
        self.should_invert = should_invert

    def __getitem__(self, index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)  # 37个类别中任选一个
        should_get_same_class = random.randint(0, 1)  # 保证同类样本约占一半
        if should_get_same_class:
            while True:
                # 直到找到同一类别
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] == img1_tuple[1]:
                    break
        else:
            while True:
                # 直到找到非同一类别
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] != img1_tuple[1]:
                    break
        # 将img0_tuple和img1_tuple对应的图像文件路径加载为PIL.Image.Image对象，并将它们转换为灰度图像
        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])
        img0 = img0.convert("L")# 转换为灰度图像
        img1 = img1.convert("L")

        if self.should_invert: # 反转图像像素
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None: # 应用指定的图像转换
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1, torch.from_numpy(np.array([int(img1_tuple[1] != img0_tuple[1])], dtype=np.float32)),img0_tuple[1],img1_tuple[1]

    def __len__(self):
        return len(self.imageFolderDataset.imgs)

#
# 定义文件dataset
training_dir = "./ORL_faces_pytorch-main/src/data/orl_faces/train/"  # 训练集地址
folder_dataset = torchvision.datasets.ImageFolder(root=training_dir)

# 定义图像dataset
transform = transforms.Compose([transforms.Resize((100, 100)),# 将图像调整为100x100的大小
                                transforms.ToTensor()])# 将图像转换为张量
# 创建SiameseNetworkDataset对象，将图像数据集和图像转换操作作为参数传递给它
siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset,
                                        transform=transform,
                                        should_invert=False)
#
# 定义图像dataloader
train_dataloader = DataLoader(siamese_dataset,
                              shuffle=True,
                              batch_size=train_batch_size)



# 搭建模型，卷积神经网络（CNN）架构
class SiameseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # 卷积神经网络，由多个卷积层、ReLU激活函数和批归一化层组成
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 4, kernel_size=3),#输入通道数为1，输出通道数为4，卷积核大小为3x3。
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),

            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),

            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
        )
        # 全连接网络，由多个线性层和ReLU激活函数组成
        self.fc1 = nn.Sequential(
            nn.Linear(8 * 100 * 100, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 5))

    # 在一个输入上前向传播，并返回相应的输出
    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    # 在两个输入上前向传播，并返回相应的输出
    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


# 自定义ContrastiveLoss，继承自torch.nn.Module
class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        # 分别计算了同类样本和异类样本之间的损失
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive


net = SiameseNetwork().cuda()  # 定义模型且移至GPU
criterion = ContrastiveLoss()  # 定义损失函数
# 定义优化器，设置学习速率，反向传播过程中根据计算出的梯度更新模型的参数，从而最小化损失函数并提高模型的性能
optimizer = optim.Adam(net.parameters(), lr=0.0005)
counter = []
loss_history = []#损失值
iteration_number = 0

# 开始训练
for epoch in range(0, train_number_epochs):
    for i, data in enumerate(train_dataloader, 0):
        img0, img1, label,_,_ = data
        # img0维度为torch.Size([32, 1, 100, 100])，32是batch，label为torch.Size([32, 1])
        img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()  # 数据移至GPU
        optimizer.zero_grad()#在向后传递之前将模型参数的梯度归零
        output1, output2 = net(img0, img1)
        loss_contrastive = criterion(output1, output2, label)#对比损耗
        loss_contrastive.backward()#反向传播是通过调用来计算相对于模型参数的损失梯度来执行的
        optimizer.step()
        if i % 10 == 0:
            iteration_number += 10
            counter.append(iteration_number)
            loss_history.append(loss_contrastive.item())
    print("Epoch number: {} , Current loss: {:.4f}\n".format(epoch, loss_contrastive.item()))#打印当前epoch和损失值

show_plot(counter, loss_history)


# 保存模型
torch.save(net.state_dict(), './model.pth')
