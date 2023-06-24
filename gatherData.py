# This file is neccessary for all other files so we gather all the imports here

import os
import glob
import time
import pickle
import math
import random

random.seed(100)

import pandas as pd
import numpy as np
import pandas.errors
import scipy.io as spio
from scipy.signal import filtfilt, butter
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

# import tensorflow as tf
# from tensorflow import keras
# from keras import layers

# global-default variables can be placed here....

training_input = {
    'emg': [
        'gastrocmed',
        'tibialisanterior',
        'soleus',
        'vastusmedialis',
        'vastuslateralis',
        'rectusfemoris',
        'bicepsfemoris',
        'semitendinosus',
    ]
    # 'gon': [
    #     'knee_sagittal','ankle_sagittal','hip_sagittal'
    # ],
    # 'ik': [
    #     'pelvis_tilt', 'pelvis_list', 'pelvis_rotation', 'pelvis_tx', 'pelvis_ty', 'pelvis_tz',
    #     'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r', 'knee_angle_r', 'ankle_angle_r',
    #     'subtalar_angle_r',
    #     'hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l', 'knee_angle_l',
    #     'ankle_angle_l', 'subtalar_angle_l'
    # ],
    # 'fp': [
    #     'FP1_vx', 'FP1_vy', 'FP1_vz', 'FP1_px', 'FP1_py', 'FP1_pz', 'FP1_moment_x',
    #     'FP2_vx', 'FP2_vy', 'FP2_vz', 'FP2_px', 'FP2_py', 'FP2_pz', 'FP2_moment_x',
    #     'FP3_vx', 'FP3_vy', 'FP3_vz', 'FP3_px', 'FP3_py', 'FP3_pz', 'FP3_moment_x',
    #     'FP4_vx', 'FP4_vy', 'FP4_vz', 'FP4_px', 'FP4_py', 'FP4_pz', 'FP4_moment_x',
    #     'FP5_vx', 'FP5_vy', 'FP5_vz', 'FP5_px', 'FP5_py', 'FP5_pz', 'FP5_moment_x',
    # ] # For stair trials
    # 'fp': [
    #     'Treadmill_R_vx', 'Treadmill_R_vy', 'Treadmill_R_vz', 'Treadmill_R_px', 'Treadmill_R_pz',
    #     'Treadmill_R_moment_x', 'Treadmill_R_moment_y', 'Treadmill_R_moment_z'
    # ]  # For treadmill trials
}
output_features = ["knee_angle_r_moment", 'knee_angle_r']
disp_features = ['HeelStrike', 'ToeOff', 'subject_num', 'Header']

subj_info = pd.read_csv("SubjectInfo.csv")


# subj_info.drop(index=[i for i in range(1, len(subj_info))], inplace=True)


def getTrainingData(workpath, subjects, training_in, training_out, trial_types=['stair'], trial_feature=['stairHeight'],
                    save_condition=False,
                    dropNaN=True):
    # Create training scenario. Choose which subjects, which trial types and which data types will get fed to the NN
    # Separation between Input and Output features is made outside this function
    # Initially, no inter-trial-type training scenarios will be held and we train the NN with just the 'stair' data

    data_types = list(training_in.keys())
    features = [i for sublist in list(training_in.values()) for i in sublist]
    disp_features = ['HeelStrike', 'ToeOff', 'Header']
    print("Gathering training data . . .")
    start = time.time()
    subject_dataframes = []
    for _, subject in subjects.iterrows():
        # Get subject info features to add to the input data
        subject_age = subject['Age']
        subject_weight = subject['Weight']
        subject_height = subject['Height']
        # Subject number is needed for file iteration
        subject_num = subject['Subject']
        trial_type_dataframes = []
        subject_file_path = os.path.join(workpath, "**", subject_num)
        for trial_type in trial_types:
            # The training conditions will be always extracted for info for each trial_type
            path_condition = os.path.join(subject_file_path, "**", trial_type, "conditions", "*.csv")
            condition_files = glob.glob(path_condition)
            # Get dataframes of different data types to merge them (inner join) on the time column
            type_dataframes = []
            for data_type in data_types:
                # Get data from all trials (for this type of data, eg 'emg' data) and concatenate them into a single dataframe
                trial_dataframes = []
                path_data = os.path.join(subject_file_path, "**", trial_type, data_type, "*.csv")
                data_files = glob.glob(path_data)
                if len(data_files) != len(condition_files):
                    print('Current Data Type:', data_type)
                    raise Exception("Data files and Condition files not the same")
                for data_file, condition_file in zip(data_files, condition_files):
                    # Load data file in dataframe together with the corresponding condition file
                    data_type = pd.read_csv(data_file)
                    # Add information about the specific trial
                    trial_condition = pd.read_csv(condition_file)
                    data_type[trial_feature] = trial_condition[trial_feature]
                    data_type = data_type[
                        1000 * data_type['Header'] % 5 == 0]  # Downsample to ID and IK sampling frequency
                    data_type.reset_index(inplace=True, drop=True)
                    trial_dataframes.append(data_type)
                # Concatenate data from all the trials for this data type and pass it to the upper dataframe list

                alltrials_frame = pd.concat(trial_dataframes)
                type_dataframes.append(alltrials_frame)

            # Horizontal concatenation of all the dataframes (by default have same number of columns and same ordering)
            # We drop the two overlapping columns for every dataframe except the first one
            # (Because joining on Header (time) is problematic due to rounding errors)
            # Instead we concatenate horizontally

            for i in range(1, len(type_dataframes)):
                type_dataframes[i].drop(['Header'] + trial_feature, axis=1, inplace=True)

            input_dataframe = pd.concat(type_dataframes, axis=1)

            # Drop unnecessary columns THIMISOU OTI EDW MPAINEI TO TRIAL CONDITION
            input_dataframe = input_dataframe.drop(
                columns=[i for i in input_dataframe.columns if
                         i not in features and i not in ['Header']])
            # ID/IK data are sampled in 1/5 the frequency. This means we want to keep the first of every 5 rows in our inputs
            # input_dataframe = input_dataframe[input_dataframe.index % 5 == 0]
            input_dataframe.reset_index(inplace=True, drop=True)
            # Output is always going to be Inverse Dynamics / Kinematics data.
            path_id = os.path.join(subject_file_path, "**", trial_type, 'id', "*.csv")
            id_files = glob.glob(path_id)
            id_frames = pd.concat([pd.read_csv(i) for i in id_files], axis=0)

            path_ik = os.path.join(subject_file_path, '**', trial_type, 'ik', "*.csv")
            ik_files = glob.glob(path_ik)
            ik_frames = pd.concat([pd.read_csv(i) for i in ik_files], axis=0)

            ik_frames.drop(columns=['Header'], inplace=True)
            output_dataframe = pd.concat([id_frames, ik_frames], axis=1)

            # Keep wanted output features

            output_dataframe.reset_index(drop=True, inplace=True)

            # Normalize with subject weight ! ! !
            # output_dataframe[output_features] /= float(subject_weight)

            # Add gait cycle % for display purposes
            path_gc = os.path.join(subject_file_path, "**", trial_type, 'gcRight', "*.csv")
            gcfiles = glob.glob(path_gc)

            gc_df = pd.concat([pd.read_csv(i) for i in gcfiles])

            gc_df.drop(columns=['Header'], inplace=True)
            gc_df.reset_index(inplace=True, drop=True)
            output_dataframe = pd.concat([gc_df, output_dataframe], axis=1)
            output_dataframe = output_dataframe[1000 * output_dataframe['Header'] % 5 == 0]
            output_dataframe.reset_index(drop=True, inplace=True)
            # To + disp features prokalei diplo Header sto apotelesma
            output_dataframe = output_dataframe[training_out + disp_features]
            output_dataframe = output_dataframe.iloc[:, :-1]
            trial_df = pd.concat([input_dataframe, output_dataframe], axis=1)
            if dropNaN:
                trial_df.dropna(inplace=True)  # Drop rows that knee ID is NaN

            trial_type_dataframes.append(trial_df)

        subject_df = pd.concat(trial_type_dataframes)
        # Finally we add the subject characteristics
        subject_df['subject_num'] = subject_num
        # subject_df['age'] = subject_age
        # subject_df['height'] = subject_height
        # subject_df['weight'] = subject_weight
        subject_dataframes.append(subject_df)
        # Should save the training data in each subject folder but keep that for later use for inter-subject training
        #
        #

        if save_condition:
            subject_df.to_csv("Training_Data" + subject['Subject'] + ".csv", index=False)
            print("Training Data saved")
    final_df = pd.concat(subject_dataframes)
    print(f'Done (time elapsed: {time.time() - start:.2f}s)')
    return final_df


def loadTrainingData(workpath, subjects):
    subject_dfs = []
    for _, subject in subjects.iterrows():
        filename = "Training_Data" + subject['Subject'] + ".csv"
        subject_dfs.append(pd.read_csv(filename))
    final = pd.concat(subject_dfs)
    return final


def dropDisplayFeatures(df):
    disp_features = ['HeelStrike', 'ToeOff', 'Header']
    # subject_related=['subject_num']
    # disp_features += subject_related
    disp_df = df[disp_features]
    training_df = df.drop(columns=disp_features, axis=1)

    return training_df, disp_df


def groupbyFeature(df, feature=None):
    if feature == None:
        return [df], ['Original - Not Grouped']  # In a list for iteration
    else:
        gb_object = df.groupby([feature])
        group_dataframes = [gb_object.get_group(group) for group in gb_object.groups]
        group_attributes = list(gb_object.groups.keys())  # The name/attr of each group
        return group_dataframes, group_attributes


def getGaitCycles(df, preprocess_EMG=False):
    cycles = []

    data_copy = df.copy()
    # Index issues because of drop NaN
    data_copy.reset_index(drop=True, inplace=True)
    # Differentiate to get starts-ends indices of each cycle in the dataframe
    data_copy['diffs'] = data_copy['HeelStrike'].diff()
    gc_start = [data_copy.index[0]] + data_copy[data_copy['diffs'] < 0].index.tolist()

    for i, start in enumerate(gc_start):
        if i + 1 < len(gc_start):
            end = gc_start[i + 1]  # The end of the cycle is just before the next cycle
        else:
            # end = data_copy.index[-1]  # the end of the cycle is the end of the dataframe
            continue  # problems with the last cycle
        cycle = data_copy.iloc[start:end]
        # Had cases where cycles were empty. Also no_of_rows>200 to skip incomplete gait cycles
        if not cycle.empty and cycle.shape[0] >= 200:
            cycles.append(cycle)

    if preprocess_EMG:
        print("Processing EMGs. . .")
        cycles = process_EMGs(cycles)
        print("Finished")

    interp_cycles = []

    common_x_axis = np.linspace(start=0.0, stop=100.0, num=100)
    average_cycle = pd.DataFrame({"avg_real": [0] * len(common_x_axis), "avg_pred": [0] * len(common_x_axis)})

    for i, cycle in enumerate(cycles):
        interp_cycle = pd.DataFrame()
        # Interpolate all columns according to HeelStrike %
        for column_name in cycle.columns:
            if column_name not in disp_features:
                interp_cycle[column_name] = np.interp(common_x_axis, cycle['HeelStrike'],
                                                      cycle[column_name])
            # interp_cycle['knee_r_moment_predictions'] = np.interp(common_x_axis, cycle['HeelStrike'],
            #                                                       cycle['knee_r_moment_predictions'])
        interp_cycle['x_common'] = common_x_axis

        #     average_cycle['avg_real'] += interp_cycle['knee_angle_r_moment']
        #     # average_cycle['avg_pred'] += interp_cycle['knee_r_moment_predictions']
        interp_cycles.append(interp_cycle)
    # average_cycle['avg_real'] = average_cycle['avg_real'] / len(cycles)
    # # average_cycle['avg_pred'] = average_cycle['avg_pred'] / len(cycles)
    # average_cycle['x_common'] = common_x_axis

    return cycles, interp_cycles, average_cycle


def process_EMGs(cycles):
    # Create lowpass butteworth 4th order filter for smoothing
    cutoff_freq = 6  # Hz
    sampling_rate = 200  # Hz
    normalized_cutoff = cutoff_freq / (0.5 * sampling_rate)
    processed_cycles = []
    b, a = butter(4, normalized_cutoff, btype='low', analog=False)
    for cycle in cycles:
        # Split EMG and Output

        EMG_raw = cycle.drop(output_features + disp_features, axis=1)
        rest_of_cycle = cycle[output_features + disp_features]
        # Rectify
        EMG_rectified = EMG_raw.abs()
        # Smooth each column seperately with the butteworth filter
        EMG_smoothed = pd.DataFrame()
        for column_name in EMG_rectified.columns:
            EMG_smoothed[column_name] = filtfilt(b, a, EMG_rectified[column_name])

        # Normalize. This resets the dataframe index.
        EMG_normalized = EMG_smoothed

        # EMG_normalized = (EMG_smoothed - EMG_smoothed.mean()) / EMG_smoothed.std()
        rest_of_cycle.reset_index(inplace=True, drop=True)
        # Update the original cycles list with the new cycle
        processed_cycles.append(pd.concat([EMG_normalized, rest_of_cycle], axis=1))

    return processed_cycles
