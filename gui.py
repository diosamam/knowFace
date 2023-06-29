import torch
import torchvision

import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset

import torchvision.utils
import numpy as np

import tkinter as tk
from PIL import ImageTk, Image

from ORL_faces_gpu import SiameseNetworkDataset, SiameseNetworkDataset_test, SiameseNetwork

# from ORL_faces_gpu import net
#定义测试的dataset和dataloader

#定义文件dataset
testing_dir = "./ORL_faces_pytorch-main/src/data/orl_faces/test/"  #测试集地址
folder_dataset_test = torchvision.datasets.ImageFolder(root=testing_dir)

#定义图像dataset
transform_test = transforms.Compose([transforms.Resize((100,100)),
                                     transforms.ToTensor()])
siamese_dataset_test = SiameseNetworkDataset_test(imageFolderDataset=folder_dataset_test,
                                        transform=transform_test,
                                        should_invert=False)

#定义图像dataloader
test_dataloader = DataLoader(siamese_dataset_test,
                            shuffle=True,
                            batch_size=1)


#生成对比图像

dataiter = iter(test_dataloader)
# x0,_,_,x0label,_ = next(dataiter)
x0,x0label = next(dataiter)

threshold = 0.9  # 距离阈值，欧氏距离大于0.9就认为不是同一张人脸

# 在测试集上计算距离并进行划分
predictions = []
labels = []
f=[]



# 创建Tkinter窗口
window = tk.Toplevel()

# 定义窗口的尺寸
window.geometry("600x600")

# 创建标签以显示图像
image_label = tk.Label(window)
image_label.place(x=110, y=0)

# 计算F1值
def calculate_f1_score(true_positives, false_positives, false_negatives):
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score

# 显示图像的更新

# 计算和显示混淆矩阵、准确率、精确率和召回率,
def show_result(labels ,predictions):
    predictions = np.array(predictions)
    labels = np.array(labels)
    # labels = predictions

    # 计算混淆矩阵
    confusion_matrix = np.zeros((2, 2))
    confusion_matrix[0, 0] = np.sum(np.logical_and(predictions == 0, labels == 0))
    confusion_matrix[0, 1] = np.sum(np.logical_and(predictions == 1, labels == 0))
    confusion_matrix[1, 0] = np.sum(np.logical_and(predictions == 0, labels == 1))
    confusion_matrix[1, 1] = np.sum(np.logical_and(predictions == 1, labels == 1))

    # 计算准确率、精确率和召回率
    accuracy = np.sum(predictions == labels) / len(labels)
    precision = confusion_matrix[1, 1] / (confusion_matrix[1, 1] + confusion_matrix[0, 1])
    recall = confusion_matrix[1, 1] / (confusion_matrix[1, 1] + confusion_matrix[1, 0])
    f1=calculate_f1_score( confusion_matrix[1, 1],  (confusion_matrix[1, 1] + confusion_matrix[0, 1]), (confusion_matrix[1, 1] + confusion_matrix[1, 0]))
    text = "Confusion Matrix:\n" + str( confusion_matrix )+ "\n" + "Accuracy: {:.4f}".format(
        accuracy) + "\n" + "Precision: {:.4f}".format(precision) + "\n" + "Recall: {:.4f}".format(recall)+ "\n" +"F1: {:.4f}".format(f1)
    label_text.config(text=text)  # 更新label的文本内容
    # 显示损失变化图
    image_loss = Image.open('./ORL_faces_pytorch-main/src/data/orl_faces/subject01/result_2.png')
    image_loss = image_loss.resize((110, 105))
    tk_image_loss = ImageTk.PhotoImage(image_loss)
    image_label_loss.configure(image=tk_image_loss)
    image_label_loss.image = tk_image_loss
    print(accuracy)
    print(precision)
    print(recall)
    print(f1)
    print()

# 显示识别错误的人脸
def show_unidentify():
    # Open the image file
    image2 = Image.open('./ORL_faces_pytorch-main/src/data/orl_faces/subject01/result_1.png')

    image2 = image2.resize((110, 105))
    tk_image2 = ImageTk.PhotoImage(image2)
    image_label2.configure(image=tk_image2)
    image_label2.image = tk_image2

# 按一下按钮就会刷新，从测试集中随机选择一张和要识别的人脸拼接在一起显示出来
def update_image():
    # 打开图片文件
    image = Image.open('./ORL_faces_pytorch-main/src/data/orl_faces/subject01/result_0.png')

    # 调整图像大小
    image = image.resize((210, 105))
    global tk_image
    #将图像转换为Tkinter格式
    tk_image = ImageTk.PhotoImage(image)

    # 更新图像标签
    image_label.configure(image=tk_image)
    image_label.image = tk_image


# 处理按钮点击事件的方法
def next_image():
    try:

        #测试数据集中获取下一批数据样本。
        # 这里使用dataiter作为迭代器来获取数据集中的样本。
        # _表示忽略的变量，x1是输入的第二个图像，x1label是第二个图像的标签。
        x1, x1label = next(dataiter)
        # 将第一张图像x0和第二张图像x1在0维度上进行拼接，创建一个合并的图像。这样可以将这两张图像一起显示。
        concatenated = torch.cat((x0, x1), 0)
        # 加载模型
        net = SiameseNetwork().cuda()
        net.load_state_dict(torch.load('./model.pth'))
        net = net.cuda()
        # 将拼接的图像x0和x1分别作为输入传递给神经网络net
        output1, output2 = net(x0.cuda(), x1.cuda())
        # 使用欧氏距离函数F.pairwise_distance计算output1和output2之间的欧氏距离
        euclidean_distance = F.pairwise_distance(output1, output2)
        # 将欧氏距离与预先定义的阈值threshold进行比较，并将比较结果转换为0/1,0代表是同一个人脸，1代表不是同一个人脸，
        # 然后将其添加到predictions列表中。predictions用于存储模型的预测结果。
        predictions.append(int(euclidean_distance > threshold))
        # 存储真实的标签
        if x0label == x1label:
            labels.append(0)
        else:
            labels.append(1)
        f.append(euclidean_distance.item())
        # 将拼接的图像concatenated保存为本地文件
        image_path = "./ORL_faces_pytorch-main/src/data/orl_faces/subject01/result_0.png"
        torchvision.utils.save_image(concatenated, image_path)
        # gui中显示欧氏距离
        label.config(text="两张人脸图片的欧氏距离为 {:.2f}".format(euclidean_distance.item()))  # 更新标签的文本内容
        # 保存错误识别的图像并且输出
        if x0label == x1label and euclidean_distance > threshold:
            image_path_false = "./ORL_faces_pytorch-main/src/data/orl_faces/subject01/result_1.png"
            torchvision.utils.save_image(x1, image_path_false)
            show_unidentify()
        # 测试完测试集，输出测试结果
        if labels.__len__() == 29:
            show_result(labels, predictions)
        # 更新要对比的图像，将下一对图像显示在gui中
        update_image()

    except StopIteration:
        pass


label = tk.Label(window, text="两张人脸图片的欧氏距离为:")
label.place(x=100, y=120)
# 创建一个按钮来显示下一张图像
next_button = tk.Button(window, text="Next Image", command=next_image)
next_button.place(x=170, y=140)  # 将按钮放置在x=170，y=140的位置

label_txet_fasle = tk.Label(window, text="识别错误的人脸图片:")
label_txet_fasle.place(x=100, y=170)

image_label2 = tk.Label(window)
image_label2.place(x=150, y=200)



label_txet_result = tk.Label(window, text="混淆矩阵、准确率、精确率、召回率:")
label_txet_result.place(x=100, y=320)

# 创建Label小部件，并设置文本内容
label_text = tk.Label(window)
label_text.place(x=150, y=340)

label_txet_loss = tk.Label(window, text="损失变化图:")
label_txet_loss.place(x=300, y=170)
image_label_loss = tk.Label(window)
image_label_loss.place(x=380, y=200)

# 开始Tkinter事件循环
window.mainloop()

