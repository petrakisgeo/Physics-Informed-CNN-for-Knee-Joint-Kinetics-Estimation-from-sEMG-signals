# Imports made from gatherData module file

# import os
# import glob
# import time
# import pickle
#
# import pandas as pd
# import scipy.io as sio
# import matplotlib.pyplot as plt
#
# from sklearn.model_selection import train_test_split
#
# import tensorflow as tf
# from tensorflow import keras
# from keras import layers
import keras.callbacks

from gatherData import *


# def getTrainingData(workpath, subjects, training_parameters, trial_types=['stair']):
#     # Create training scenario. Choose which subjects, which trial types and which data types will get fed to the NN
#     # Separation between Input and Output features is made outside this function
#     # Initially, no inter-trial-type training scenarios will be held and we train the NN with just the 'stair' data
#
#     data_types = list(training_parameters.keys())
#     features = [i for sublist in list(training_parameters.values()) for i in sublist]
#
#     print("Gathering training data . . .")
#     start = time.time()
#     for _, subject in subjects.iterrows():
#         # Get subject info features to add to the input data
#         subject_age = subject['Age']
#         subject_weight = subject['Weight']
#         subject_height = subject['Height']
#         # Subject number is needed for file iteration
#         subject_num = subject['Subject']
#         for trial_type in trial_types:
#             # The training conditions will be always extracted for info for each trial_type
#             path_condition = os.path.join(workpath, "**", subject_num, "**", trial_type, "conditions", "*.mat")
#             condition_files = glob.glob(path_condition, recursive=True)
#             # Get dataframes of different data types to merge them (inner join) on the time column
#             datatype_dataframes = []
#             for data in data_types:
#                 # Get data from all trials (for this type of data, eg 'emg' data) and concatenate them into a single dataframe
#                 trial_dataframes = []
#                 path_data = os.path.join(workpath, "**", subject_num, "**", trial_type, data, "*.csv")
#                 data_files = glob.glob(path_data, recursive=True)
#                 if len(data_files) != len(condition_files):
#                     print('Current Data Type:', data)
#                     raise Exception("Data files and Condition files not the same")
#                 for data_file, condition_file in zip(data_files, condition_files):
#                     # Load data file in dataframe together with the corresponding condition file
#                     data = pd.read_csv(data_file)
#                     mat_data = sio.loadmat(condition_file)
#                     # Add information about the specific trial
#                     #
#                     data['stairHeight'] = mat_data['stairHeight'][0][0]
#                     trial_dataframes.append(data)
#                 # Concatenate data from all the trials for this data type and pass it to the upper dataframe list
#                 alltrials_frame = pd.concat(trial_dataframes)
#                 datatype_dataframes.append(alltrials_frame)
#             # Horizontal concatenation of all the dataframes (by default have same number of columns and same ordering)
#             # We drop the two overlapping columns for every dataframe except the first one
#             # (Because joining on Header (time) is problematic due to rounding errors)
#             # Instead we concatenate horizontally
#
#             for i in range(1, len(datatype_dataframes)):
#                 datatype_dataframes[i].drop(['Header', 'stairHeight'], axis=1, inplace=True)
#             input_dataframe = pd.concat(datatype_dataframes, axis=1)
#             # Drop unnecessary columns
#             input_dataframe = input_dataframe.drop(
#                 columns=[i for i in input_dataframe.columns if i not in features and i != 'Header'])
#
#             # Output is always going to be Inverse Dynamics / Kinematics data.
#             path_id = os.path.join(workpath, "**", subject_num, "**", trial_type, 'id', "*.csv")
#             out_files = glob.glob(path_id, recursive=True)
#             output_dataframe = pd.concat([pd.read_csv(i) for i in out_files], axis=0)
#
#             print('\tJoining input and output dataframes . . .')
#             start2 = time.time()
#             final_dataframe = pd.merge(input_dataframe, output_dataframe, on='Header', how='inner')
#             # ID/IK data are sampled in lower frequencies than the original data. This will result in duplicate rows.
#             final_dataframe = final_dataframe.drop_duplicates()
#             print(f'\tDone: (time elapsed:{time.time() - start2:.2f}s)')
#
#             # Finally we add the subject characteristics
#             final_dataframe['age'] = subject_age
#             final_dataframe['height'] = subject_height
#             final_dataframe['weight'] = subject_weight
#             # Should save the training data in each subject folder but keep that for later use for inter-subject training
#             #
#             #
#             print(f'Done (time elapsed: {time.time() - start:.2f}s)')
#     return final_dataframe

def splitByRatio(starting_list, ratio=None, number=None):
    if ratio:
        ind = math.ceil(ratio * len(starting_list))
        first_part = starting_list[:ind]
        second_part = starting_list[ind:]
    if number:
        first_part = starting_list[:-number]
        second_part = starting_list[-number:]
    return first_part, second_part


# Watch out. ratio of trials not ratio of data
def leaveTrialsOut(training_dataframe, r=0.8, val_data_ratio=None):
    grouped_by_subj = training_dataframe.groupby(['subject_num'])
    # For every subject, keep a percentage of trials as train and the rest as test
    subjects = list(grouped_by_subj.groups.keys())
    trials_by_subj = [grouped_by_subj.get_group(subject) for subject in subjects]
    train_trials = []
    test_trials = []
    val_trials = []
    for subject_trial_data in trials_by_subj:
        # For each subject, find the beginning of each trial
        subject_trial_data.loc[:, 'timedeltas'] = subject_trial_data['Header'].diff()
        # print(training_dataframe['timedeltas'].tail())
        trial_starts = [0] + list(subject_trial_data[abs(subject_trial_data['timedeltas']) > 1].index)
        print(len(trial_starts))
        # Split the beginnings (trials) into train and test
        train, test = splitByRatio(trial_starts, ratio=r)
        test_df = subject_trial_data.iloc[test[0]:]
        test_trials.append(test_df)
        if val_data_ratio:
            train, val = splitByRatio(train, ratio=val_data_ratio)
            train_df = subject_trial_data.iloc[:val[0]]
            train_trials.append(train_df)
            val_df = subject_trial_data.iloc[val[0]:]
            val_trials.append(val_df)
        else:
            train_df = subject_trial_data.iloc[:test[0]]
            train_trials.append(train_df)
    train_df = pd.concat(train_trials)
    test_df = pd.concat(test_trials)
    val_df = pd.concat(val_trials)  # will be empty if val_data_ratio=None
    return train_df, test_df, val_df


def leaveSubjectsOut(training_dataframe, r=0.75):
    grouped = training_dataframe.groupby(['subject_num'])
    subjects = list(grouped.groups.keys())
    print(subjects)
    train_subjects, test_subjects = splitByRatio(subjects, ratio=r)
    print(train_subjects, test_subjects)
    training_group = [grouped.get_group(subjects) for subjects in train_subjects]
    test_group = [grouped.get_group(subjects) for subjects in test_subjects]
    train_df = pd.concat(training_group)
    test_df = pd.concat(test_group)
    return train_df, test_df


def getTrainTestData(training_dataframe, out_features, scale_data=None):
    train_data, test_data = train_test_split(training_dataframe, test_size=0.2, random_state=10)

    train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=10)

    train_input, train_output = getInputOutput(train_data)

    test_input, test_output = getInputOutput(test_data)

    val_input, val_output = getInputOutput(val_data)

    if scale_data:
        print("Data scaling enabled")
        train_input = sc.fit_transform(train_input)
        val_input = sc.transform(val_input)
        test_input = sc.transform(test_input)

    return train_input, train_output, val_input, val_output, test_input, test_output


def getInputOutput(training_dataframe):
    x = training_dataframe.drop(output_features, axis=1)
    y = training_dataframe[output_features]
    return x, y


def plotStats(stats):
    plt.figure()
    plt.plot(stats['mse'], color='blue', label='train')
    plt.plot(stats['val_mse'], color='red', label='validation')
    plt.title("MSE over epochs")
    plt.legend()
    plt.savefig("MSE.png")
    plt.show()

    plt.figure()
    plt.plot(stats['mae'], color='blue', label='train')
    plt.plot(stats['val_mae'], color='red', label='validation')
    plt.title("MAE over epochs")
    plt.legend()
    plt.savefig("MAE.png")
    plt.show()

    return


def trainModel(x_train, y_train, val_x, val_y, model, device, stats_file, model_file):
    opt = keras.optimizers.Adam(learning_rate=0.01)
    lossfunc = keras.losses.MeanSquaredError()
    escb = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    with tf.device(device):
        print(f'Training NN on {device}. . .')
        training_start = time.time()
        model.compile(optimizer=opt, loss=lossfunc, metrics=['mse', 'mae'])
        stats = model.fit(x=x_train, y=y_train, validation_data=(val_x, val_y), epochs=500, batch_size=64,
                          callbacks=escb)
        model.save(model_file)
        print(f'Model Trained (Time elapsed {time.time() - training_start:.2f}s)')

    with open(stats_file, "wb") as f:
        pickle.dump(stats.history, f)
    plotStats(stats.history)
    return stats


if __name__ == '__main__':
    cwd = os.getcwd()

    training_df = getTrainingData(cwd, subjects=subj_info, training_in=training_input, training_out=output_features,
                                  save_condition=False, dropNaN=False)
    # training_df = loadTrainingData(cwd, subj_info)
    training_df.dropna(inplace=True)
    training_df,_=dropDisplayFeatures(training_df)

    # LeaveTrialsOut and LeaveSubjectsOut here
    # training_df, test_df,validation_df = leaveTrialsOut(training_df,val_data_ratio=0.95)
    #
    # training_df.dropna(inplace=True)
    # test_df.dropna(inplace=True)
    # validation_df.dropna(inplace=True)
    #
    # training_df, _ = dropDisplayFeatures(training_df)
    # validation_df, _ = dropDisplayFeatures(validation_df)
    # test_df, _ = dropDisplayFeatures(test_df)
    #
    # x_train, y_train = getInputOutput(training_df)
    #
    # x_val, y_val = getInputOutput(validation_df)
    # x_test, y_test = getInputOutput(test_df)

    sc = MinMaxScaler()
    # x_train = sc.fit_transform(x_train)
    # x_val = sc.transform(x_val)
    # x_test = sc.transform(x_test)
    x_train, y_train, x_val, y_val, x_test, y_test = getTrainTestData(training_df, out_features=output_features)
    print(x_train.shape)

    n_rows, n_features = x_train.shape

    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        device = "/GPU:0"
    else:
        device = "/CPU:0"

    model = keras.Sequential(layers=[
        layers.InputLayer(input_shape=(n_features,)),  # Input
        layers.Dense(128, activation='relu'),  # Hidden Layer,
        layers.BatchNormalization(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(1, activation='linear')  # Output Regression layer
    ])
    # model= keras.Sequential(layers=[
    #     layers.Conv1D(filters=16,kernel_size=3,activation='relu',input_shape=(n_features,1),padding='same'),
    #     layers.MaxPool1D(pool_size=2,strides=2),
    #     layers.BatchNormalization(),
    #     layers.Dropout(0.5),
    #     layers.Flatten(),
    #     layers.Dense(128,activation='relu'),
    #     layers.BatchNormalization(),
    #     layers.Dropout(0.5),
    #     layers.Dense(128,activation='relu'),
    #     layers.Dropout(0.5),
    #     layers.Dense(1,activation='linear')
    # ])
    model_file_path = 'FFNN256_lto.h5'
    validation_stats_path = 'FFNN256_lto_val_stats.pkl'
    validation_stats = trainModel(x_train, y_train, x_val, y_val, model, device, model_file=model_file_path,
                                  stats_file=validation_stats_path)
    test_stats = model.evaluate(x_test, y_test)
    # Remember grid selection method
    # Train and Save model
    # model_file_path = 'FFNN256_intersubject_norm.h5'
    # validation_stats_path = 'FFN256_intersubject_norm_val_stats.pkl'
    # test_stats_path = 'FFN256_intersubject_norm_test_stats.pkl'
    # validation_stats = trainModel(x_train, y_train, val_x, val_y, model, device, model_file=model_file_path,
    #                               stats_file=validation_stats_path)
    # test_stats = model.evaluate(x_test, y_test)
    # print("Test loss:", test_stats[0])
    # print("Test MSE:", test_stats[1])
    #
    # start = time.time()
    # # Feature extraction to eliminate noise and overfitting in the data
    # inputs_initial = training_df.drop(output_features, axis=1)
    # scale = StandardScaler()
    # inputs_scaled = scale.fit_transform(inputs_initial)
    # pca = PCA(n_components=0.95)  # 95% of variance kept
    # inputs_pca_applied = pd.DataFrame(pca.fit_transform(inputs_scaled))
    # inputs_pca_applied.reset_index(drop=True, inplace=True)
    # print(inputs_pca_applied.shape)
    # print(training_df[output_features].shape)
    # # Re-apply the model
    # decomposed_data = pd.concat([inputs_pca_applied, training_df[output_features]], axis=1)
    #
    # x_train, y_train, val_x, val_y, x_test, y_test = getTrainTestData(decomposed_data, out_features=output_features)
    #
    # _, n_features = x_train.shape
    #
    # model = keras.Sequential(layers=[
    #     layers.InputLayer(input_shape=(n_features,)),  # Input
    #     layers.Dense(256, activation='relu'),  # Hidden Layer
    #     layers.Dense(128, activation='relu'),
    #     layers.Dense(64, activation='relu'),
    #     layers.Dense(1, activation='linear')  # Output Regression layer
    # ])
    # model_file_path = 'FFNN256_intersubject_FE.h5'
    # validation_stats_path = 'FFN256_intersubject_FE_val_stats.pkl'
    # test_stats_path = 'FFN256_intersubject_FE_test_stats.pkl'
    # validation_stats = trainModel(x_train, y_train, val_x, val_y, model, device, model_file=model_file_path,
    #                               stats_file=validation_stats_path)
    # test_stats = model.evaluate(x_test, y_test)
    # print("Test loss:", test_stats[0])
    # print("Test MSE:", test_stats[1])
    # with open(model_stats_path, 'rb') as f:
    #     stats_1 = pickle.load(f)

    # plotStats(stats_1)

    # arxiko -> dokimazoume gia ena atomo. inputs goniometers,ID,IMU accelerations, GCFs, (emg)-> Muscle forces (ID) or joint moments (ID)
    # logika ena for loop pou:
    # 1. for subj_info.Subject (epistrefei string) -> mpainei ston fakelo tou me vasi to string. krataei plhrofories gia subject
    # 2. fortwnei dedomena kai xwrizei se input output. feedarei NN
    # 3
