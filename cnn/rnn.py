from keras.layers import SimpleRNN,Activation,Dense,Input
from keras.models import Model,load_model
from keras.optimizers import Adam
from keras.datasets import mnist
from keras.utils import np_utils


(X_train, y_train), (X_test, y_test) = mnist.load_data()

# data pre-processing
X_train = X_train.reshape(-1, 28, 28) / 255.      # normalize
X_test = X_test.reshape(-1, 28, 28) / 255.        # normalize
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

OUTPUT_SIZE = 10
CELL_SIZE = 50
LR = 0.01

input_img = Input(shape=(28,28))

rnn = SimpleRNN(
    output_dim=CELL_SIZE,
    unroll=True)(input_img)
rnn = Dense(OUTPUT_SIZE)(rnn)
rnn = Activation('softmax')(rnn)
adam = Adam(LR)

rnnModel = Model(input=input_img, output=rnn)

rnnModel.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# train
rnnModel.fit(X_train,
             y_train,
             epochs=1,
             batch_size=256,
             shuffle=True)



loss,accuracy = rnnModel.evaluate(X_test,y_test)

print(loss)
print(accuracy)
# model = load_model()

# rnnModel.fit(X_test, y_test,
#                 epochs=1,
#                 batch_size=256,
#                 shuffle=True)
