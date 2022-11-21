import numpy as np
import tensorflow as tf
from privacy_meter.audit import Audit, MetricEnum
from privacy_meter.audit_report import ROCCurveReport, SignalHistogramReport
from privacy_meter.constants import InferenceGame
from privacy_meter.dataset import Dataset
from privacy_meter.information_source import InformationSource
from privacy_meter.model import TensorflowModel
import privacy_meter.hypothesis_test as prtest
from privacy_meter.metric import ReferenceMetric
from privacy_meter.information_source_signal import ModelLoss
from privacy_meter import audit_report
import time
from privacy_meter import audit_report

def preprocess_cifar100_dataset():
    input_shape, num_classes = (32, 32, 3), 100

    # split the data between train and test sets
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()

    # scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    # convert labels into one hot vectors
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    return x_train, y_train, x_test, y_test, input_shape, num_classes


def get_tensorflow_cnn_classifier(input_shape, num_classes, regularizer):
    # TODO: change model architecture
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
    logdir = "testing_3"

    seed = 1234
    np.random.seed(seed)
    rng = np.random.default_rng(seed=seed)
    tf.compat.v1.set_random_seed(seed)
    # for training the dummy models
    num_points_per_train_split = 5000
    num_points_per_test_split = 1000
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    optim_fn = 'adam'
    epochs = 10
    batch_size = 64
    regularizer_penalty = 0.01
    regularizer = tf.keras.regularizers.l2(l=regularizer_penalty)
    # for the dummy reference metric
    num_reference_models = 10
    fpr_tolerance_list = list(np.logspace(-5,0,100))

    x_train_all, y_train_all, x_test_all, y_test_all, input_shape, num_classes = preprocess_cifar100_dataset()

    # create the dummy target model's dataset
    dataset = Dataset(
        data_dict={
            'train': {'x': x_train_all, 'y': y_train_all},
            'test': {'x': x_test_all, 'y': y_test_all}
        },
        default_input='x',
        default_output='y'
    )


    datasets_list = dataset.subdivide(
        num_splits=num_reference_models + 1,
        delete_original=True,
        in_place=False,
        return_results=True,
        method='hybrid',
        split_size={'train': num_points_per_train_split, 'test': num_points_per_test_split}
    )




    x = datasets_list[0].get_feature('train', '<default_input>')
    y = datasets_list[0].get_feature('train', '<default_output>')
    model = get_tensorflow_cnn_classifier(input_shape, num_classes, regularizer)
    model.summary()
    model.compile(optimizer=optim_fn, loss=loss_fn, metrics=['accuracy'])
    model.fit(x, y, batch_size=batch_size, epochs=epochs, verbose=2)




    target_model = TensorflowModel(model_obj=model, loss_fn=loss_fn)


    reference_models = []
    for model_idx in range(num_reference_models):
        print(f"Training dummy model {model_idx}...")
        reference_model = get_tensorflow_cnn_classifier(input_shape, num_classes, regularizer)
        reference_model.compile(optimizer=optim_fn, loss=loss_fn, metrics=['accuracy'])
        reference_model.fit(
            datasets_list[model_idx + 1].get_feature('train', '<default_input>'),
            datasets_list[model_idx + 1].get_feature('train', '<default_output>'),
            batch_size=batch_size,
            epochs=epochs,
            verbose=2
        )
        reference_models.append(
            TensorflowModel(model_obj=reference_model, loss_fn=loss_fn)
        )

    start_time = time.time()
    target_info_source = InformationSource(
        models=[target_model],
        datasets=[datasets_list[0]]
    )

    reference_info_source = InformationSource(
        models=reference_models,
        datasets=[datasets_list[0]] # we use the same dataset for the reference models
    )


    linear_itp_reference_metric = ReferenceMetric(
        target_info_source = target_info_source,
        reference_info_source = reference_info_source,
        signals = [ModelLoss()],
        hypothesis_test_func = prtest.linear_itp_threshold_func
    )
    logit_rescale_reference_metric = ReferenceMetric(
        target_info_source = target_info_source,
        reference_info_source = reference_info_source,
        signals = [ModelLoss()],
        hypothesis_test_func = prtest.logit_rescale_threshold_func
    )
    min_linear_logit_reference_metric = ReferenceMetric(
        target_info_source = target_info_source,
        reference_info_source = reference_info_source,
        signals = [ModelLoss()],
        hypothesis_test_func = prtest.min_linear_logit_threshold_func
    )
    audit_obj = Audit(
        metrics=[linear_itp_reference_metric, logit_rescale_reference_metric, min_linear_logit_reference_metric],
        inference_game_type=InferenceGame.PRIVACY_LOSS_MODEL,
        target_info_sources=target_info_source,
        reference_info_sources=reference_info_source,
        fpr_tolerances=np.linspace(0,1,1000).tolist(),
        logs_directory_names= [logdir, logdir, logdir]
    )
    audit_obj.prepare()


    audit_results = audit_obj.run()
    linear_itp_audit_results = audit_results[0]
    logit_rescale_audit_results = audit_results[1]
    min_linear_logit_audit_results = audit_results[2]
    print(linear_itp_audit_results[0].accuracy, logit_rescale_audit_results[0].accuracy, min_linear_logit_audit_results[0].accuracy)
    print('uses {}'.format(time.time()-start_time))
    
    
    audit_report.REPORT_FILES_DIR = 'privacy_meter/report_files'
    ROCCurveReport.generate_report(
    metric_result=linear_itp_audit_results,
    inference_game_type=InferenceGame.PRIVACY_LOSS_MODEL,
    show=True,
    filename = logdir+'/linear_itp_roc_curve.png'
    )
    ROCCurveReport.generate_report(
    metric_result=logit_rescale_audit_results,
    inference_game_type=InferenceGame.PRIVACY_LOSS_MODEL,
    show=True,
    filename = logdir+'/logit_rescale_audit_results.png'
    )
    ROCCurveReport.generate_report(
    metric_result=min_linear_logit_audit_results,
    inference_game_type=InferenceGame.PRIVACY_LOSS_MODEL,
    show=True,
    filename = logdir+'/min_linear_logit_audit_results.png'
    )