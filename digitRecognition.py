import numpy
from tensorflow import keras
from keras import layers

inputs = keras.Input(shape=(784,), name="digits")
# 中间层
x = layers.Dense(256, activation="relu", name="dense_1")(inputs)
x = layers.Dense(256, activation="relu", name="dense_2")(x)
# x = layers.Dense(256, activation="relu", name="dense_3")(x)
# x = layers.Dense(256, activation="relu", name="dense_4")(x)

outputs = layers.Dense(10, activation="softmax", name="predictions")(x)

model = keras.Model(inputs=inputs, outputs=outputs)

(x_train, train_labels), (x_test, test_labels) = keras.datasets.mnist.load_data()

# 数据预处理
x_train = x_train.reshape(60000, 784).astype("float32") / 255
x_test = x_test.reshape(10000, 784).astype("float32") / 255

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
