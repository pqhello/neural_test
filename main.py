import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
from torchvision import datasets
import torchvision
from torch.utils.data import DataLoader
import torch


from MultillayerPerceptron import MultillayerPerceptron
if __name__ == "__main__":
    train_dataset = datasets.MNIST(root='./neural_test/data/MNIST',train=True,transform=torchvision.transforms.ToTensor(),download=True)
    eval_dataset = datasets.MNIST(root='./neural_test/data/MNIST',train=False,transform=torchvision.transforms.ToTensor(),download=True)
    # data = pd.read_csv("./neural_test/data/MNIST/mnist_train.csv")
    dataloaders = DataLoader(train_dataset,batch_size=64,shuffle=True)
    #取出数据
    # print(dataloaders.dataset)
    num_to_show = 25
    num_cells = math.ceil(math.sqrt(num_to_show))
    # plt.figure(figsize=(10, 10))
    # for i in range(num_to_show):
    # # for data in dataloaders.dataset:
    #     (image, label) = dataloaders.dataset[i]
    #     # print(np.shape(image)) # torch.Size([1, 28, 28])
    #     plt.subplot(num_cells, num_cells, i + 1)
    #     print(np.squeeze(image).shape) # (28, 28)
    #     plt.imshow(np.squeeze(image), cmap='gray')
    #     plt.title(f'Label: {label}')
    #     # plt.axis('off')
    # plt.subplots_adjust(wspace=0.3, hspace=0.4)
    # plt.show()
#获取全部训练数据的标签和图像,选择前800个样本作为训练数据，前200个样本作为验证数据
    train_labels = train_dataset.targets[:800]

    # train_labels = [label for _, label in dataloaders.dataset[:800]]
    # train_images = dataloaders.dataset[:800][0].tolist()
    train_images = train_dataset.data[:800]
    print(train_labels.shape) #(800, 28, 28)
    print(f"Total number of training samples: {len(train_labels)}") #60000
#获取全部验证数据的标签和图像
    eval_dataloader = DataLoader(eval_dataset, batch_size=64, shuffle=False)
    # eval_labels = eval_dataloader.dataset[:200][1].tolist()
    # eval_images = eval_dataloader.dataset[:200][0].tolist()
    eval_labels = eval_dataset.targets[:200]
    eval_images = eval_dataset.data[:200]
    # print(f"Total number of evaluation samples: {len(eval_labels)}")#10000
    layers= [784,25,10]
    normalize_data = True#归一化
    max_iterations = 100#迭代次数
    alpha = 0.01#学习率
    train_images = train_dataset.data[:800].reshape(800, -1).float()
    print(train_images.shape) #(800, 784)
    MultillayerPerceptron = MultillayerPerceptron(train_images, train_labels, layers, normalize_data)
    (thetas,cost_history) = MultillayerPerceptron.train(max_iterations,alpha)
    plt.plot(range(len(cost_history)), cost_history)
    plt.xlabel('cost_history')
    plt.show()



