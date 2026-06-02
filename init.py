import numpy as np
from utils.features import prepare_for_training
from utils.hypothesis import sigmod,sigmoid_gradient

class MultillayerPerceptron:
    def __init__(self,data,labels,layers,normalize_data=False):
        data_processed = prepare_for_training(data,normalize_data=normalize_data)[0]
        self.data = data_processed
        self.labels = labels
        self.layers = layers #784*25*10
        self.normalize_data = normalize_data
        self.thetas = MultillayerPerceptron.thetas_init(layers)

    def train(self,max_iteration=1000,alpha=0.1):
        #为了方便参数更新，所以将参数矩阵拉长
        unrolled_thetas=MultillayerPerceptron.thetas_unroll(self.thetas)
        (optimized_thetas,cost_history) = MultillayerPerceptron.gradient_descent(self.data,self.labels,unrolled_thetas,self.layers,max_iteration,alpha)
        self.thetas = MultillayerPerceptron.thetas_roll(optimized_thetas,self.layers)
        return self.thetas,cost_history

    @staticmethod
    def gradient_descent(data,labels,unrolled_theta,layers,max_iteration,alpha):
        '''
        1.计算当前的损失值
        2.计算梯度值
        3.进行梯度值的更新
        '''
        optimized_thetas = unrolled_theta
        cost_history = []
        for _ in range(max_iteration):
            #计算损失需要前向传播走一次，所以参数要变回矩阵形式
            cost = MultillayerPerceptron.cost_function(data,labels,MultillayerPerceptron.thetas_roll(optimized_thetas,layers),layers)
            cost_history.append(cost)
            theta_gradient = MultillayerPerceptron.gradient_step(data,labels,optimized_thetas,layers)
            optimized_thetas = optimized_thetas - alpha*theta_gradient
        return optimized_thetas,cost_history
    @staticmethod
    def gradient_step(data,labels,optimized_thetas,layers):
        thetas = MultillayerPerceptron.thetas_roll(optimized_thetas,layers)
        thetas_rolled_gradients = MultillayerPerceptron.back_propagaion(data,labels,thetas,layers)
        thetas_unrolled_gradients = MultillayerPerceptron.thetas_unroll(thetas_rolled_gradients)
        return thetas_unrolled_gradients
    
    @staticmethod
    def back_propagaion(data,labels,thetas,layers):
        num_layers =len(layers)
        (num_examples,num_features) = data.shape
        num_lables_types = layers[-1]
        deltas = {}

        for layer_index in range(num_layers-1):
            in_count = layers[layer_index]
            out_count = layers[layer_index+1]
            deltas[layer_index]= np.zeros((out_count,in_count+1))
        for example_index in range(num_examples):
            layers_inputs = {}
            layers_activations = {}
            layers_activation = data[example_index,:].reshape((num_features,1))
            layers_activations[0] = layers_activation

            for index in range(num_layers-1):
                layer_theta = thetas[index]
                layer_input = np.dot(layer_theta,layers_activation)
                layers_activation = np.vstack((np.array((1)),sigmod(layer_input)))
                layers_inputs[index+1] = layer_input#后一层的计算结果
                layers_activations[index+1] = layers_activation
            output_layer_action = layers_activation[1:,:]

            delta = {}
            #标签处理
            bitwise_label = np.zeros((num_lables_types,1))
            bitwise_label[labels[example_index[0]]] = 1
            #计算输出层和真实值之间的差异
            delta[num_layers-1] = output_layer_action - bitwise_label
            #遍历循环 L-1,L-2..2
            for layers_index in range(num_layers-2,0,-1):
                layer_theta = thetas[layer_index]
                next_delta = delta[layer_index+1]
                layer_input = layers_inputs[layer_index]
                layer_input = np.vstack((np.array((1)),layer_input))
                #按照公式进行计算
                np.dot(layer_theta.T,next_delta)*sigmoid_gradient(layer_input)
                #去掉偏执项
                delta[layers_index] = delta[layers_index][1:,:]
            #计算每一层的梯度值
            for layer_index in range(num_layers-1):
                delta_layer = delta[layer_index+1]
                layer_activation = layers_activations[layer_index].T
                layer_delta = delta[layer_index]+np.dot(delta_layer,layer_activation) 
                deltas[layer_index] = deltas[layer_index]+layer_delta
            for layer_index in range(num_layers-1):
                deltas[layer_index] = deltas[layer_index]/num_examples









    @staticmethod  
    def cost_function(data,labels,thetas,layers):
        num_layers = len(layers)
        num_examples =data.shape[0]
        num_labels = layers[-1]

        predictions = MultillayerPerceptron.feedforward_propagaion(data,thetas,layers)
        #制作向量标签
        bitwise_labels=np.zeros((num_examples,num_labels))
        for example_index in range(num_examples):
            bitwise_labels[example_index,labels[example_index][0]]= 1
        bit_set_cost = np.sum(np.log(predictions[bitwise_labels==1]))
        bit_not_set_cost = np.sum(np.log(1-predictions[bitwise_labels==0]))
        cost = (bit_set_cost+bit_not_set_cost)*(-1/num_examples)
        return cost
    @staticmethod
    def feedforward_propagaion(data,thetas,layers):
        num_layers = len(layers)
        num_examples =data.shape[0]
        in_layer_activation = data
        for layer_index in range(num_layers-1):
            theta =thetas[layer_index]
            out_layer_activation = sigmod(np.dot(in_layer_activation,theta.T))
            out_layer_activation = np.hstack((np.ones((num_examples,1)),out_layer_activation))
            in_layer_activation = out_layer_activation
        return in_layer_activation[:,1:]
    @staticmethod
    def thetas_roll(unrollede_thetas,layers):
        num_layers = len(layers)
        thetas = {}
        unrolled_shift = 0
        for layer_index in range(num_layers-1):
            in_count = layers[layer_index]
            out_count = layers[layer_index+1]

            thetas_width = in_count
            thetas_height = out_count
            thetas_volume = thetas_width*thetas_height
            start_index = unrolled_shift
            end_index = unrolled_shift+thetas_volume
            layer_theta_unrolled =unrollede_thetas[start_index,end_index]
            thetas[layer_index] = layer_theta_unrolled.reshape((thetas_height,thetas_width))
            unrolled_shift = unrolled_shift+thetas_volume
        return thetas



    @staticmethod
    def thetas_unroll(thetas):
        num_theta= len(thetas)
        unrolled_theta = np.array([])
        for thetas_index in range(num_theta):
            unrolled_theta=np.hstack((unrolled_theta,thetas[thetas_index].flatten()))
        return thetas

    @staticmethod
    def thetas_init(layers):
        '''
        根据传进来的layers，初始化特征值矩阵
        '''
        num_layers = len(layers)
        thetas = {}
        for layer_init in range(num_layers-1):
            in_count = layers[layer_init]
            out_count = layers[layer_init+1]
            thetas[layer_init]=np.random.rand(out_count,in_count+1)*0.05#考虑偏执参数所以在in_count加1，*0.05是希望随机的特征值能小一点
        return thetas
    
