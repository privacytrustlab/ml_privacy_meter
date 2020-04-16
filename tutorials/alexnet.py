import ml_privacy_meter
import tensorflow as tf
keras = tf.keras
keraslayers = tf.compat.v1.keras.layers 
import numpy as np
import os

def classification_cnn(input_shape):
  """
  AlexNet:
  Described in: http://arxiv.org/pdf/1404.5997v2.pdf
  Parameters from:
  github.com/akrizhevsky/cuda-convnet2/blob/master/layers/
  """
  # Creating initializer, optimizer and the regularizer ops
  initializer = tf.compat.v1.keras.initializers.random_normal(0.0, 0.01)
  regularizer = tf.keras.regularizers.l2(5e-4)

  inputshape = (input_shape[0], input_shape[1], input_shape[2],)

  #Creating the model
  model = tf.compat.v1.keras.Sequential( 
        [
          keraslayers.Conv2D(
              64, 11, 4, 
              padding='same', 
              activation=tf.nn.relu,
              kernel_initializer=initializer,
              kernel_regularizer=regularizer,
              input_shape = inputshape,
              data_format = 'channels_last'
            ),
          keraslayers.MaxPooling2D(
              2, 2, padding='valid' 
            ),
          keraslayers.Conv2D(
              192, 5, 
              padding='same', 
              kernel_initializer=initializer,
              kernel_regularizer=regularizer,
              activation=tf.nn.relu
            ),
          keraslayers.MaxPooling2D(
              2, 2, padding='valid'
            ),
          keraslayers.Conv2D(
              384, 3, 
              padding='same', 
              kernel_initializer=initializer,
              kernel_regularizer=regularizer,
              activation=tf.nn.relu
            ),
          keraslayers.Conv2D(
              256, 3, 
              padding='same', 
              kernel_initializer=initializer,
              kernel_regularizer=regularizer,
              activation=tf.nn.relu
            ),    
          keraslayers.Conv2D(
              256, 3, 
              padding='same', 
              kernel_initializer=initializer,
              kernel_regularizer=regularizer,
              activation=tf.nn.relu
            ),    
          keraslayers.MaxPooling2D(
              2, 2, padding='valid'
            ),
          keraslayers.Flatten(),
          keraslayers.Dropout(0.3),
          keraslayers.Dense(
              100,
              kernel_initializer=initializer,
              kernel_regularizer=regularizer,
              activation=tf.nn.softmax
            )
         ]
      )
  return model

input_shape = (32, 32, 3)
cmodel = classification_cnn(input_shape)
advreg_wrapper = ml_privacy_meter.defense.advreg.initialize()

# Get the datahandler ('tf.data.Dataset' instance)
dataset_path = "datasets/cifar100.txt"

def scheduler(epoch):
    lr = 0.0001
    if epoch > 25:
        lr = 0.00001
    elif epoch > 60:
        lr = 0.000001
    print('Using learning rate', lr)
    return lr


if __name__ == '__main__':
    callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
    datahandler = advreg_wrapper.get_datahandler(dataset_path=dataset_path, 
                                                 batch_size=128, 
                                                 training_size=30000,
                                                 input_shape=input_shape,
                                                 normalization=True)


    features, labels = datahandler.train_features, datahandler.train_labels
    print('Train features', np.array(datahandler.train_features).shape, np.array(datahandler.train_labels).shape)
    opt = keras.optimizers.Adam(learning_rate=0.0001)

    cmodel.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    size = len(features)
    num_classes = 100
    
    reshape_input = (len(features),) + (self.input_shape[2], self.input_shape[0], self.input_shape[1])
    features = np.transpose(np.reshape(features, reshape_input), (0, 2, 3, 1))

    features_train = features[int(0.2 * size) :]
    features_test = features[: int(0.2 * size)]

    features_train = datahandler.normalize(features_train)
    features_train = datahandler.normalize(features_test)

    labels_train = labels[int(0.2 * size) :]
    labels_test = labels[: int(0.2 * size)]

    labels_train = keras.utils.to_categorical(labels_train, num_classes)
    labels_test = keras.utils.to_categorical(labels_test, num_classes)
    print(cmodel.summary())

    cmodel.fit(features_train, labels_train,
              batch_size=128,
              epochs=100,
              validation_data=(features_test, labels_test),
              shuffle=True, callbacks=[callback])

    model_path = os.path.join('cifar100_model')
    cmodel.save(model_path)
    print('Saved trained model at %s ' % model_path)
