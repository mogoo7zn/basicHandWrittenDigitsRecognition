import numpy
from keras import layers
from tensorflow import keras

inputs = keras.Input(shape=(28, 28, 1,), name="digits")
# 中间层
x = layers.Conv2D(filters=6, kernel_size=(5, 5), strides=(1, 1), input_shape=(28, 28, 1),
                  padding='valid', activation='relu', name='CNN_1')(inputs)
x = layers.AveragePooling2D(pool_size=(2, 2), name='pooling_1')(x)
x = layers.Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1),
                  padding='valid', activation='relu', name='CNN_2')(x)
x = layers.AveragePooling2D(pool_size=(2, 2), name='pooling_2')(x)
x = layers.Flatten()(x)
x = layers.Dense(units=120, activation="relu", name="dense_1")(x)
x = layers.Dense(units=84, activation="relu", name="dense_2")(x)

outputs = layers.Dense(10, activation="softmax", name="predictions")(x)

model = keras.Model(inputs=inputs, outputs=outputs)

(x_train, train_labels), (x_test, test_labels) = keras.datasets.mnist.load_data()

# 数据预处理
x_train = x_train.reshape(60000, 28, 28, 1).astype("float32") / 255
x_test = x_test.reshape(10000, 28, 28, 1).astype("float32") / 255

train_labels = train_labels.astype("float32")
test_labels = test_labels.astype("float32")

# 选择末尾10000个作为测试集
x_val = x_train[-10000:]
y_val = train_labels[-10000:]
x_train = x_train[:-10000]
train_labels = train_labels[:-10000]

model.compile(
    # 优化器
    optimizer=keras.optimizers.SGD(learning_rate=0.05),
    # 损失函数
    loss=keras.losses.SparseCategoricalCrossentropy(),
    # 监视的矩阵
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

print("Fit model on training data")
history = model.fit(
    x_train,
    train_labels,
    batch_size=128,
    epochs=2,
    # 在数据集的每段最后10000个，改为监视训练效果的验证集
    validation_data=(x_val, y_val),
)

print(history.history)

# 在测试集上验证模型
print("Evaluate on test data")
results = model.evaluate(x_test, test_labels, batch_size=128)
print("test loss, test acc:", results)

# 对已标签的字符做预测
print("Generate predictions for 3 samples")
predictions = model.predict(x_test[:3])
print("predictions shape:", predictions.shape)
numpy.array(predictions)
print("predictions of each test case:\n", predictions)
print("actual labels of first three numbers:", test_labels[:3])
print("predicted results:", predictions.argmax(axis=1))
