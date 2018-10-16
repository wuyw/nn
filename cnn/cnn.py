from keras.layers import Dense,Activation,Convolution2D,MaxPooling2D,Flatten,Input
from keras.models import Model
from keras.datasets import mnist
from keras.optimizers import Adam
from keras.utils import np_utils


(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(-1, 1,28, 28)/255.
X_test = X_test.reshape(-1, 1,28, 28)/255.
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

input_img = Input(shape=(1,28,28))

cnnLayer = Convolution2D(
    filters=32,
    kernel_size=5,
    strides=1,
    padding='same',
    data_format='channels_first')(input_img)
cnnLayer = Activation('relu')(cnnLayer)
cnnLayer = MaxPooling2D(pool_size=2,
                        strides=2,
                        padding='same',
                        data_format='channels_first')(cnnLayer)


cnnLayer = Convolution2D(64, 5,
                         strides=1,
                         padding='same',
                         data_format='channels_first')(cnnLayer)
cnnLayer = Activation('relu')(cnnLayer)
cnnLayer = MaxPooling2D(2, 2,
                        'same',
                        data_format='channels_first')(cnnLayer)
cnnLayer = Flatten()(cnnLayer)
cnnLayer = Dense(1024)(cnnLayer)
cnnLayer = Activation('relu')(cnnLayer)

cnnLayer = Dense(10)(cnnLayer)
cnnLayer = Activation('softmax')(cnnLayer)


cnn = Model(input=input_img, output=cnnLayer)


adam = Adam(lr=1e-4)

cnn.compile(optimizer=adam,
            loss='categorical_crossentropy',
            metrics=['accuracy'])

cnn.fit(X_train, y_train,
                epochs=1,
                batch_size=256,
                shuffle=True)

print('\nTesting ------------')
# Evaluate the model with the metrics we defined earlier
loss, accuracy = cnn.evaluate(X_test, y_test)

print('\ntest loss: ', loss)
print('\ntest accuracy: ', accuracy)