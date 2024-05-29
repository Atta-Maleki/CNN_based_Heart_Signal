from scipy.io import loadmat
import numpy as np
import tensorflow as tf
import re
from tensorflow.keras.models import Model
from tensorflow.python.keras.engine import data_adapter
from statistics import mean
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import confusion_matrix


class CWTDataGen(tf.keras.utils.Sequence):
    def __init__(self, list_IDs, labels, batch_size=16, dim=(9000,), n_channels=1,
                 n_classes=3, max_classes=4, image_dir='signal_images2/', shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.max_classes = max_classes
        self.image_dir = image_dir
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.list_IDs))
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        # self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        # y = np.empty((self.batch_size), dtype=int)
        y = np.empty((self.batch_size, self.max_classes))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Read signal image
            cwt = np.load(self.image_dir + ID + '.npy', allow_pickle=True)
            # Store cwt image
            X[i,] = cwt

            # Store class
            # y[i] = self.labels[ID]
            y[i,] = self.labels[ID]

        # return X, y
        # return X, tf.keras.utils.to_categorical(y, num_classes=self.n_classes)
        if self.n_classes != self.max_classes:
            return X, y[:, :-(self.max_classes-self.n_classes)]
        return X, y


# Callback for computing f1 score
class AdditionalMetrics(Callback):
    def __init__(self, val_data, best_val_scores, model_name, save_mode='saved_model'):
        'Initialization'
        self.val_f1s = []
        self.val_f1s_plus_noisy = []
        self.best_epoch = 0
        self.batch_num = 0
        self.epoch_num = 0
        self.val_data = val_data
        self.model_name = model_name
        self.save_mode = save_mode
        self.best_val_scores = best_val_scores

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_num = epoch + 1

    def on_test_begin(self, logs=None):
        self.batch_num = 0

    def on_test_batch_end(self, batch, logs=None):
        self.batch_num += 1

    def on_test_end(self, logs=None):
        final_predictions = []
        final_labels = []

        for i in range(self.batch_num):
            # Get arrays from generator
            X_test = self.val_data[i][0]
            y_test = self.val_data[i][1]

            # Generate predictions
            final_predictions += np.argmax(self.model.predict(X_test), axis=1).tolist()
            final_labels += np.argmax(y_test, axis=1).tolist()

        # Create confusion matrix
        cm = confusion_matrix(final_labels, final_predictions)

        # Calculate precision, recall, and F1 score
        p_af = cm[0, 0] / np.sum(cm[:, 0])
        r_af = cm[0, 0] / np.sum(cm[0, :])
        f1_af = 2 * (p_af * r_af) / (p_af + r_af)
        p_normal = cm[1, 1] / np.sum(cm[:, 1])
        r_normal = cm[1, 1] / np.sum(cm[1, :])
        f1_normal = 2 * (p_normal * r_normal) / (p_normal + r_normal)
        p_other = cm[2, 2] / np.sum(cm[:, 2])
        r_other = cm[2, 2] / np.sum(cm[2, :])
        f1_other = 2 * (p_other * r_other) / (p_other + r_other)
        p_noisy = cm[3, 3] / np.sum(cm[:, 3])
        r_noisy = cm[3, 3] / np.sum(cm[3, :])
        f1_noisy = 2 * (p_noisy * r_noisy) / (p_noisy + r_noisy)

        # Print metrics
        print(f"F1 Scores: AF = {f1_af}, Normal = {f1_normal}, Other = {f1_other}, Noisy = {f1_noisy}")
        avg_f1 = (f1_af + f1_normal + f1_other) / 3
        avg_f1_plus_noisy = (f1_af + f1_normal + f1_other + f1_noisy) / 4
        print(f"Avg_f1: {avg_f1}")

        # Saving the best model so far
        if avg_f1 > max(self.val_f1s, default=0.0):
            if self.save_mode == 'h5':
                self.model.save(f'{self.model_name}_{self.epoch_num}.h5')
            else:
                self.model.save(f'{self.model_name}_{self.epoch_num}')
            # Updating best epoch
            self.best_epoch = self.epoch_num

        # Updating f1_score list
        if np.isnan(avg_f1):
            self.val_f1s.append(0.0)
        else:
            self.val_f1s.append(avg_f1)

        if np.isnan(avg_f1_plus_noisy):
            self.val_f1s_plus_noisy.append(0.0)
        else:
            self.val_f1s_plus_noisy.append(avg_f1_plus_noisy)

        # Early stopping
        if self.epoch_num > 10 + self.best_epoch and self.epoch_num > 35:
            print(f"Training stopped at epoch {self.epoch_num} by early stopping")
            self.model.stop_training = True

    def on_train_end(self, logs=None):
        self.best_val_scores.append(max(self.val_f1s, default=0.0))
        # Save f1_scores for 3 and 4 classes
        np.save(f"{self.model_name}_F1s.npy", self.val_f1s)
        np.save(f"{self.model_name}_F1s_plus_noisy.npy", self.val_f1s_plus_noisy)


# Callback for saving models, F1 scores and early stopping.
class SaveModels(Callback):
    def __init__(self, best_val_scores, model_name, save_mode='saved_model'):
        'Initialization'
        self.val_f1s_3_classes = []
        self.val_f1s_4_classes = []
        self.best_epoch_3_classes = 0
        self.best_epoch_4_classes = 0
        self.model_name = model_name
        self.save_mode = save_mode
        self.best_val_scores = best_val_scores

    def save_model(self, epoch_num, class_type, f1_score, mode):
        if mode == 'saved_model':
            self.model.save(f'{self.model_name}_{epoch_num}_{class_type}_{f1_score}')
        else:
            self.model.save(f'{self.model_name}_{epoch_num}_{class_type}_{f1_score}.h5')

    def on_epoch_end(self, epoch, logs=None):
        # Get and print average F1 for 3 and 4 classes.
        avg_f1_3_classes = logs["val_avg_F1_3_classes"]
        avg_f1_4_classes = logs["val_avg_F1_4_classes"]
        already_saved = False
        print("")
        print(f"Avg_f1_3: {avg_f1_3_classes}, Avg_f1_4: {avg_f1_4_classes}")

        # Save the best model so far.
        if avg_f1_3_classes > max(self.val_f1s_3_classes, default=0.0):
            # Define class type of the model.
            class_type = "3C"
            if avg_f1_4_classes > max(self.val_f1s_4_classes, default=0.0):
                class_type = "3+4C"
            # Save the model.
            self.save_model(epoch, class_type, avg_f1_3_classes, self.save_mode)
            already_saved = True
            # Update the best epoch.
            self.best_epoch_3_classes = epoch

        if avg_f1_4_classes > max(self.val_f1s_4_classes, default=0.0):
            if not already_saved:
                class_type = "4C"
                self.save_model(epoch, class_type, avg_f1_3_classes, self.save_mode)
            self.best_epoch_4_classes = epoch

        # Updating f1_score lists.
        self.val_f1s_3_classes.append(avg_f1_3_classes)
        self.val_f1s_4_classes.append(avg_f1_4_classes)

        # Early stopping.
        if epoch > 10 + self.best_epoch_3_classes and epoch > 50:
            print(f"Training stopped after epoch {epoch} by early stopping mechanism")
            self.model.stop_training = True

    def on_train_end(self, logs=None):
        # Save the best val F1 for the current fold.
        self.best_val_scores.append(max(self.val_f1s_3_classes, default=0.0))


# Confusion matrix class for F1 metrics.
class ConfusionMatrixMetric(tf.keras.metrics.Metric):
    """
    A custom Keras metric to compute the running average of the confusion matrix
    """
    def __init__(self, num_classes, name='confusion_matrix_metric', **kwargs):
        super().__init__(name=name, **kwargs)  # handles base args (e.g., dtype)
        self.num_classes = num_classes
        self.total_cm = self.add_weight("total", shape=(num_classes, num_classes), initializer="zeros")

    def get_config(self):
        config = super().get_config()
        config['num_classes'] = self.num_classes
        return config

    def reset_state(self):
        for s in self.variables:
            s.assign(tf.zeros(shape=s.shape))

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.total_cm.assign_add(self.confusion_matrix(y_true, y_pred))
        return self.total_cm

    def result(self):
        return self.process_confusion_matrix()

    def confusion_matrix(self, y_true, y_pred):
        """
        Make a confusion matrix
        """
        y_true = tf.math.argmax(y_true, axis=1)
        y_pred = tf.math.argmax(y_pred, axis=1)
        cm = tf.math.confusion_matrix(y_true, y_pred, dtype=tf.float32, num_classes=self.num_classes)
        return cm

    def process_confusion_matrix(self):
        """
        Returns precision, recall and f1 along with overall accuracy
        """
        cm = self.total_cm
        diag_part = tf.linalg.diag_part(cm)
        precision = diag_part / (tf.reduce_sum(cm, 0) + tf.constant(1e-15))
        recall = diag_part / (tf.reduce_sum(cm, 1) + tf.constant(1e-15))
        f1 = 2 * precision * recall / (precision + recall + tf.constant(1e-15))
        return precision, recall, f1

    def fill_output(self, output):
        """
        Fill in metrics
        """
        results = self.result()
        for i in range(self.num_classes):
            output[f'precision_{i}'] = results[0][i]
            output[f'recall_{i}'] = results[1][i]
            output[f'F1_{i}'] = results[2][i]
        # Computing average f1 for 3 and 4 classes.
        output['avg_F1_3_classes'] = tf.math.reduce_mean(results[2][:3], axis=0)
        output['avg_F1_4_classes'] = tf.math.reduce_mean(results[2], axis=0)


# Custom model class for F1 metrics.
class CustomModel(tf.keras.Model):
    def train_step(self, data):
        # Unpack the data.
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        # Run forward pass.
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(
                y, y_pred, sample_weight, regularization_losses=self.losses)
        if self.loss and y is None:
            raise TypeError(
                f'Target data is missing. Your model has `loss`: {self.loss}, '
                'and therefore expects target data to be passed in `fit()`.')
        # Run backwards pass.
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        self.compiled_metrics.update_state(y, y_pred, sample_weight)

        # Collect metrics to return.
        return_metrics = {}
        for metric in self.metrics[:-1]:
            result = metric.result()
            if isinstance(result, dict):
                return_metrics.update(result)
            else:
                return_metrics[metric.name] = result
        if 'confusion_matrix_metric' in self.metrics_names:
            self.metrics[-1].fill_output(return_metrics)
        return return_metrics

    def test_step(self, data):
        # Unpack the data.
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        # Run forward pass.
        y_pred = self(x, training=False)
        # Updates stateful loss metrics.
        self.compiled_loss(
            y, y_pred, sample_weight, regularization_losses=self.losses)
        self.compiled_metrics.update_state(y, y_pred, sample_weight)

        # Collect metrics to return.
        return_metrics = {}
        for metric in self.metrics[:-1]:
            result = metric.result()
            if isinstance(result, dict):
                return_metrics.update(result)
            else:
                return_metrics[metric.name] = result
        if 'confusion_matrix_metric' in self.metrics_names:
            self.metrics[-1].fill_output(return_metrics)
        return return_metrics


# Function for inserting and removing layers
def insert_layer_nonseq(model, layer_regex, insert_layer_factory,
                        insert_layer_name=None, position='after'):
    # Auxiliary dictionary to describe the network graph
    network_dict = {'input_layers_of': {}, 'new_output_tensor_of': {}}

    # Set the input layers of each layer
    for layer in model.layers:
        for node in layer._outbound_nodes:
            layer_name = node.outbound_layer.name
            if layer_name not in network_dict['input_layers_of']:
                network_dict['input_layers_of'].update(
                    {layer_name: [layer.name]})
            else:
                network_dict['input_layers_of'][layer_name].append(layer.name)

    # Set the output tensor of the input layer
    network_dict['new_output_tensor_of'].update(
        {model.layers[0].name: model.input})

    # Iterate over all layers after the input
    model_outputs = []
    for layer in model.layers[1:]:

        # Determine input tensors
        layer_input = [network_dict['new_output_tensor_of'][layer_aux]
                       for layer_aux in network_dict['input_layers_of'][layer.name]]
        if len(layer_input) == 1:
            layer_input = layer_input[0]

        # Insert layer if name matches the regular expression
        if re.match(layer_regex, layer.name):
            if position == 'replace':
                x = layer_input
            elif position == 'after':
                x = layer(layer_input)
            elif position == 'before':
                pass
            else:
                raise ValueError('position must be: before, after or replace')

            new_layer = insert_layer_factory()
            if insert_layer_name:
                new_layer.name = insert_layer_name
            else:
                new_layer.name = '{}_{}'.format(layer.name,
                                                new_layer.name)
            x = new_layer(x)
            print('New layer: {} Old layer: {} Type: {}'.format(new_layer.name,
                                                                layer.name, position))
            if position == 'before':
                x = layer(x)
        else:
            x = layer(layer_input)

        # Set new output tensor (the original one, or the one of the inserted
        # layer)
        network_dict['new_output_tensor_of'].update({layer.name: x})

        # Save tensor in output list if it is output in initial model
        if layer_name in model.output_names:
            model_outputs.append(x)

    return Model(inputs=model.inputs, outputs=model_outputs)

