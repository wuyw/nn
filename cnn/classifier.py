from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.optimizers import RMSprop

(X_train,Y_train),(X_test,Y_test) = mnist.load_data()


X_train = X_train.reshape(X_train.shape[0],-1)/255.
X_test = X_test.reshape(X_test.shape[0],-1)/255.
Y_train = np_utils.to_categorical(Y_train,num_classes=10)
Y_test = np_utils.to_categorical(Y_test,num_classes=10)

print(X_train[1].shape)
print(Y_train[:3])

mode = Sequential([Dense(32,input_dim=784),
                   Activation('relu'),
                   Dense(10),
                   Activation('softmax')])

rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

mode.compile(optimizer=rmsprop,
             loss='categorical_crossentropy',
             metrics=['accuracy'])
mode.fit(X_train,Y_train,epochs=100,batch_size=32)

loss,accuracy = mode.evaluate(X_test,Y_test)


print('test loss: ',loss)
print('test accuracy: ',accuracy)

