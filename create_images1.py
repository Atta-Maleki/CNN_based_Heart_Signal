import pandas as pd
import signal_processing2

# data_dir = 'training2017_augmented/'
# image_dir = 'augmented_signals/'
#data_dir = 'training2017_augmented_relabeled/'
# image_dir = 'augmented_signals_relabeled/'
# data_dir = 'Cambell_data/'
# image_dir = 'Cambell_data_filtered/'

data_dir = 'C:/AFDetectionData/training2017/'
image_dir = 'C:/AFDetectionData/training2017/orig_signals/'

# For 2d cnn resampling
array_dir = 'signal_images/'

# classification = pd.read_csv('training2017/REFERENCE_RELABELED.csv')
# classification = pd.read_csv('training2017_augmented/REFERENCE_AUGMENTED.csv')
#classification = pd.read_csv('Cambell_data/REFERENCE_CAMBELL_2.csv')
classification = pd.read_csv('C:/AFDetectionData/training2017/REFERENCE.csv')

IDs = classification["ID"]
classes = classification["Label"]

ID_list = []
class_list = []

for i in range(len(classification.index)):
    if classes[i] != "Missing":
        ID_list.append(IDs[i])
        class_list.append(classes[i])

downsample_res = 9000

#signal_processing.data_augment(ID_list, class_list, data_dir, nsr_copies=0, length_rand_mode=0, time_rand=0.1, polarity_flip=True, time_flip=False)
signal_processing2.create_filtered_signals(ID_list, data_dir, image_dir, filt_type=0, length_norm=9000)
#signal_processing2.downsample_signal(ID_list, image_dir, f'orig_data_filtered_{downsample_res}/', downsample_res)

# signal_processing.create_filtered_signals_cambell(ID_list, data_dir, image_dir, filt_type=0, length_norm=9000)

