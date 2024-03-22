import numpy as np
import gzip
import pickle
import random
import matplotlib.pyplot as plt
from PIL import Image
import random


lr = 0.1  
lamda=10.0
num=50000

def load_data():
    f = gzip.open('mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = pickle.load(f,encoding='bytes')
    f.close()
    return train_set, valid_set, test_set

def sigmoid(x):
    return 1/(1+np.exp(-1*x))

def sigmoid_prime(x):
    return sigmoid(x)*(1-sigmoid(x))

class layer():
    def __init__(self,input_size,output_size):
        self.input_size=input_size
        self.output_size=output_size
        self.input=np.zeros(shape=[input_size])
        self.weight=np.random.uniform(-1,1,[input_size,output_size])
        self.bias=np.random.uniform(-1,1,[output_size])
        self.delta=[]
        self.weight_delta=[]

    def forward(self,data):
        self.input=data
        return sigmoid(np.dot(data,self.weight)+self.bias)
    
    def backword(self,delta):
        self.delta.append(delta)
        weight_delta=np.dot(self.input.reshape(len(self.input),1),delta.reshape(1,len(delta)))+lamda*self.weight/num
        self.weight_delta.append(weight_delta)
        return np.dot(self.weight,delta)*(self.input*(1-self.input))
    
    def renew(self):
        self.weight_delta=np.asarray(self.weight_delta)
        self.delta=np.asarray(self.delta)
        self.weight_delta=np.mean(self.weight_delta,axis=0)
        self.bias_delta=np.mean(self.delta,axis=0)
        self.weight=(1-lamda*lr/num)*self.weight-lr*self.weight_delta
        self.bias-=lr*self.bias_delta
        self.delta=[]
        self.weight_delta=[]

def costFunction(output,expect):
    exp=np.zeros(shape=[10])
    exp[expect]=1
    return np.sum((output-expect)**2)/2



class net():
    def __init__(self,layer_num):
        self.layer_num=layer_num
        self.layer=[]
        self.layer_len=len(layer_num)-1
        for i in range(self.layer_len):
            self.layer.append(layer(layer_num[i],layer_num[i+1]))
        pass

    def __forward(self,data):
        for i in range(self.layer_len):
            data=self.layer[i].forward(data)
        return data
    
    def __backword(self,delta):
        for i in range(self.layer_len):
            delta=self.layer[-i-1].backword(delta)
        pass
    
    def __renew(self):
        for l in self.layer:
            l.renew()

    def __last_delta(self,data,expect):
        return (data-expect)

    def test(self,test_data):
        d=self.__forward(test_data)
        return np.argmax(d)
    
    def train(self,epoche,batch_size,ts,tes,filename="net.data"):
        maximumAccuracy=0
        for j in range(epoche):
            test_data=[]
            for i in range(len(ts[0])):
                test_data.append([ts[0][i],ts[1][i]])
            random.shuffle(test_data)
            for i,d in enumerate(test_data):
                batch_expect=np.zeros(shape=[self.layer_num[-1]])
                batch_expect[d[1]]=1
                delta=self.__last_delta(self.__forward(d[0]),batch_expect)
                self.__backword(delta)
                if (i+1)%batch_size==0:
                    self.__renew()
            accuracy=self.test(tes)
            print(j,accuracy)
            if accuracy>maximumAccuracy:
                maximumAccuracy=accuracy
                print("save")
                self.save(filename)

    def train_max(self,epoche,batch_size,ts,tes,filename="net.data"):
        maximumAccuracy=0
        for j in range(epoche):
            test_data=[]
            for i in range(len(ts[0])):
                arr=Image.fromarray(np.uint8(ts[0][i].reshape(28,28)*255))
                r=random.randint(0,6)
                p=random.randint(-5,5)
                if r==0:
                    arr=arr.rotate(10)
                elif r==1:
                    arr=arr.rotate(-20)
                elif r==2:
                    arr=arr.rotate(20)
                elif r==3:
                    arr=arr.rotate(-10)
                if random.randint(0,1)==0:
                    arr=arr.transform((28,28),Image.AFFINE,(1,0,p,0,1,0),Image.BICUBIC)
                arr=np.asarray(arr).reshape(784)/255.0
                test_data.append([arr,ts[1][i]]) 
            random.shuffle(test_data)
            for i,d in enumerate(test_data):
                batch_expect=np.zeros(shape=[self.layer_num[-1]])
                batch_expect[d[1]]=1
                delta=self.__last_delta(self.__forward(d[0]),batch_expect)
                self.__backword(delta)
                if (i+1)%batch_size==0:
                    self.__renew()
            accuracy=self.test(tes)
            print(j,accuracy)
            if accuracy>maximumAccuracy:
                maximumAccuracy=accuracy
                print("save")
                self.save(filename)

    def test(self,test_data):
        num=0
        for i in range(len(test_data[0])):
            d=self.__forward(test_data[0][i])
            if np.argmax(d)==test_data[1][i]:
                num+=1
        return num
    
    def use(self,data):
        return np.argmax(self.__forward(data))

    def save(self,filename="net.data"):
        with open(filename,"wb") as f:
            pickle.dump(self,f)

def load(filename="net.data"):
    with open(filename,"rb") as f:
        return pickle.load(f,encoding='bytes')  
        

if __name__=="__main__":
    '''
    #数据训练
    ts, vs, tes = load_data()
    t=net([784,100,10])             #这里是定义网络结构
    t.train(30,10,ts,tes,"net.data")
    '''
    
    ts, vs, tes = load_data()
    i=plt.imread("testimagine/2.png")       #这里是读取图片
    i=i[:,:,0]
    i=1-i
    plt.imshow(i,cmap='gray')
    plt.show()
    t=load("net.data")
    ans=t.use(i.reshape(784))
    print(ans)
    
    
    
    """ts, vs, tes = load_data()
    t=net([784,100,30,10])
    t.train_max(60,30,ts,tes,"netmax_r_30.data")"""
    