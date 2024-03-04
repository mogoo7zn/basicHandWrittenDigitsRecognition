"""this project is based on Lenet5 idea"""

from keras.datasets import mnist
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import to_categorical
from keras.layers import Conv2D
from keras.layers import AveragePooling2D
from keras.layers import Flatten


if __name__ == '__main__':
    # X_train为60000*28*28的数据, Y_train为60000元素的数组,每个元素代表标签
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

    X_train = X_train.reshape(60000, 28, 28, 1) / 255.0
    X_test = X_test.reshape(10000, 28, 28, 1) / 255.0

    # 将数组元素转为one-hot型数据,标注结果的标签数据
    Y_train = to_categorical(Y_train, 10)
    Y_test = to_categorical(Y_train, 10)

    # 使用same模式卷积
    model = Sequential()
    model.add(Conv2D(filters=6, kernel_size=(5, 5), strides=(1, 1), input_shape=(28, 28, 1),
                     padding='valid', activation='relu'))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1),
                     padding='valid', activation='relu'))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(units=120, activation='relu'))
    model.add(Dense(units=84, activation='relu'))
    model.add(Dense(units=10, activation='softmax'))

    # 训练
    model.compile(loss='categorical_crossentropy', optimizer=SGD(learning_rate=0.05), metrics=['accuracy'])
    model.fit(X_train, Y_train, epochs=5, batch_size=128)

    # 评估测试集
    loss, accuracy = model.evaluate(X_test, Y_test)
    print("Loss"+str(loss))
    print("Accuracy"+str(accuracy))