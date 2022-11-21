import numpy as np
import tensorflow as tf
import sys
from privacy_meter.audit import Audit, MetricEnum
from privacy_meter.audit_report import ROCCurveReport, SignalHistogramReport
from privacy_meter.constants import InferenceGame
from privacy_meter.dataset import Dataset
from privacy_meter.information_source import InformationSource
from privacy_meter.model import TensorflowModel
from privacy_meter import audit_report
import time
from privacy_meter.audit import Audit, MetricEnum
from privacy_meter.audit_report import ROCCurveReport, SignalHistogramReport
from privacy_meter.constants import InferenceGame
from privacy_meter.dataset import Dataset
from privacy_meter.information_source import InformationSource
from privacy_meter.model import TensorflowModel
import privacy_meter.hypothesis_test as prtest
from privacy_meter.metric import PopulationMetric,GroupPopulationMetric
from privacy_meter.information_source_signal import ModelLoss



def preprocess_cifar10_dataset():
    input_shape, num_classes = (32, 32, 3), 10

    # split the data between train and test sets
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    # convert labels into one hot vectors
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    return x_train, y_train, x_test, y_test, input_shape, num_classes


def get_tensorflow_cnn_classifier(input_shape, num_classes, regularizer):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu',
                                     input_shape=input_shape, kernel_regularizer=regularizer))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu',
                                     kernel_regularizer=regularizer))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    return model


if __name__ == '__main__':
    # for training the target model
    num_train_points = 500
    num_test_points = 500
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    optim_fn = 'adam'
    epochs = 25
    batch_size = 64
    regularizer_penalty = 0.01
    regularizer = tf.keras.regularizers.l2(l=regularizer_penalty)
    # for the population metric
    num_population_points = 10000
    fpr_tolerance_list = list(np.linspace(0,1,300))
    
    
    x_train_all, y_train_all, x_test_all, y_test_all, input_shape, num_classes = preprocess_cifar10_dataset()
    x_train, y_train = x_train_all[:num_train_points], y_train_all[:num_train_points]
    x_test, y_test = x_test_all[:num_test_points], y_test_all[:num_test_points]
    x_population = x_train_all[num_train_points:(num_train_points + num_population_points)]
    y_population = y_train_all[num_train_points:(num_train_points + num_population_points)]
    
    # create the target model's dataset
    train_ds = {'x': x_train, 'y': y_train,'g':y_train.argmax(axis=1)}
    test_ds = {'x': x_test, 'y': y_test,'g':y_test.argmax(axis=1)}
    target_dataset = Dataset(
        data_dict={'train': train_ds, 'test': test_ds},
        default_input='x', default_output='y', default_group='g',
    )

    # create the reference dataset
    population_ds = {'x': x_population, 'y': y_population, 'g':y_population.argmax(axis=1)}
    reference_dataset = Dataset(
        # this is the default mapping that a Metric will look for
        # in a reference dataset
        data_dict={'train': population_ds},
        default_input='x', default_output='y',default_group='g'
    )
    x = target_dataset.get_feature('train', '<default_input>')
    y = target_dataset.get_feature('train', '<default_output>')
    model = get_tensorflow_cnn_classifier(input_shape, num_classes, regularizer)
    model.summary()
    model.compile(optimizer=optim_fn, loss=loss_fn, metrics=['accuracy'])
    model.fit(x, y, batch_size=batch_size, epochs=epochs, verbose=2)
    
    
    target_model = TensorflowModel(model_obj=model, loss_fn=loss_fn)
    
    start_time = time.time()
    target_info_source = InformationSource(
    models=[target_model], 
    datasets=[target_dataset]
    )

    reference_info_source = InformationSource(
        models=[target_model],
        datasets=[reference_dataset]
    )
    metric = PopulationMetric(
    target_info_source = target_info_source,
    reference_info_source = reference_info_source,
    signals = [ModelLoss()],
    hypothesis_test_func = prtest.linear_itp_threshold_func
    )
    metric2 = GroupPopulationMetric(
        target_info_source = target_info_source,
        reference_info_source = reference_info_source,
        signals = [ModelLoss()],
        hypothesis_test_func = prtest.logit_rescale_threshold_func
    )
    
    audit_obj = Audit(
    metrics=[metric,metric2],
    inference_game_type=InferenceGame.PRIVACY_LOSS_MODEL,
    target_info_sources=target_info_source,
    reference_info_sources=reference_info_source,
    fpr_tolerances=fpr_tolerance_list,
    logs_directory_names=['pop','group_pop']
    )
    audit_obj.prepare()
    audit_results = audit_obj.run()
    population_results = audit_results[0]
    group_population_results = audit_results[1]
    # for result in audit_results:
    #     print(result)
    
    print('uses {}'.format(time.time()-start_time))
    # This instruction won't be needed once the tool is on pip
    audit_report.REPORT_FILES_DIR = 'privacy_meter/report_files'
    ROCCurveReport.generate_report(
    metric_result=population_results,
    inference_game_type=InferenceGame.PRIVACY_LOSS_MODEL,
    show=True,
    filename='population.png'
    )
    SignalHistogramReport.generate_report(
    metric_result=population_results[0],
    inference_game_type=InferenceGame.PRIVACY_LOSS_MODEL,
    show=True,
    filename='population_his.png'
    )
    
    ROCCurveReport.generate_report(
    metric_result=group_population_results,
    inference_game_type=InferenceGame.PRIVACY_LOSS_MODEL,
    show=True,
    filename='group.png'
    )
    SignalHistogramReport.generate_report(
    metric_result=group_population_results[0],
    inference_game_type=InferenceGame.PRIVACY_LOSS_MODEL,
    show=True,
    filename='group_hist.png'
    )