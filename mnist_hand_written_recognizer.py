from keras.datasets import mnist
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import to_categorical

import plot_utils

if __name__ == '__main__':
    # X_train为60000*28*28的数据, Y_train为60000元素的数组,每个元素代表标签
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

    # print("X_Train.shape" + str(X_train.shape))
    # print("Y_Train.shape" + str(Y_train.shape))

    X_train = X_train.reshape(60000, 784) / 255.0
    X_test = X_test.reshape(10000, 784) / 255.0

    # 将数组元素转为one-hot型数据,标注结果的标签数据
    Y_train = to_categorical(Y_train, 10)
    Y_test = to_categorical(Y_train, 10)

    # 设置神经网络节点层数和每层数量
    model = Sequential()
    model.add(Dense(units=256, activation='relu', input_dim=784))
    model.add(Dense(units=256, activation='relu'))
    model.add(Dense(units=256, activation='relu'))
    model.add(Dense(units=10, activation='sigmoid'))

    # mean_squared_error
    model.compile(loss='categorical_crossentropy', optimizer=SGD(learning_rate=0.05), metrics=['accuracy'])

    model.fit(X_train, Y_train, epochs=2, batch_size=128)

    pres = model.predict(X)

    plot_utils.show_scatter_surface(X, Y, model)

    print(model.get_weights())

# 测试
# loss, accuracy = model.evaluate(X_test, Y_test)
# print("Loss"+str(loss))
# print("Accuracy"+str(accuracy))