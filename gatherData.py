# This file is neccessary for all other files so we gather all the imports here

import os
import sys
import glob
import time
import pickle
import math
import random
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom

random.seed(100)

import pandas as pd
import numpy as np
import pandas.errors
import scipy.io as spio
from scipy.signal import filtfilt, butter
from scipy.stats import pearsonr
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style(style='darkgrid')
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
        # 'gracilis',
        # 'gluteusmedius',
        # 'rightexternaloblique'
    ],
    # 'ik': [
    #     'pelvis_tilt', 'pelvis_list', 'pelvis_rotation', 'pelvis_tx', 'pelvis_ty', 'pelvis_tz',
    #     'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r', 'knee_angle_r', 'ankle_angle_r',
    #     'subtalar_angle_r', 'mtp_angle_r',
    #     'hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l', 'knee_angle_l',
    #     'ankle_angle_l', 'subtalar_angle_l', 'mtp_angle_l', 'lumbar_extension',
    #     'lumbar_bending', 'lumbar_rotation'
    # ],
    # 'fp': [
    #     'FP1_vx', 'FP1_vy', 'FP1_vz', 'FP1_px', 'FP1_py', 'FP1_pz', 'FP1_moment_x',
    #     'FP2_vx', 'FP2_vy', 'FP2_vz', 'FP2_px', 'FP2_py', 'FP2_pz', 'FP2_moment_x',
    #     'FP3_vx', 'FP3_vy', 'FP3_vz', 'FP3_px', 'FP3_py', 'FP3_pz', 'FP3_moment_x',
    #     'FP4_vx', 'FP4_vy', 'FP4_vz', 'FP4_px', 'FP4_py', 'FP4_pz', 'FP4_moment_x',
    #     'FP5_vx', 'FP5_vy', 'FP5_vz', 'FP5_px', 'FP5_py', 'FP5_pz', 'FP5_moment_x',
    # ] # For stair trials
    # 'fp': [
    #     'Treadmill_R_vx', 'Treadmill_R_vy', 'Treadmill_R_vz', 'Treadmill_R_px', 'Treadmill_R_py', 'Treadmill_R_pz',
    #     'Treadmill_R_moment_x', 'Treadmill_R_moment_y', 'Treadmill_R_moment_z',
    #     'Treadmill_L_vx', 'Treadmill_L_vy', 'Treadmill_L_vz', 'Treadmill_L_px', 'Treadmill_L_py', 'Treadmill_L_pz',
    #     'Treadmill_L_moment_x', 'Treadmill_L_moment_y', 'Treadmill_L_moment_z'
    # ]  # For treadmill trials
}
output_features = ["knee_angle_r_moment", 'knee_angle_r']
disp_features = ['HeelStrike', 'ToeOff', 'subject_num', 'Header']

subj_info = pd.read_csv("SubjectInfo.csv")


# subj_info.drop(index=[i for i in range(0, 10)], inplace=True)


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
                    # trial_condition = pd.read_csv(condition_file)
                    # data_type[trial_feature] = trial_condition[trial_feature]
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
                type_dataframes[i].drop(['Header'], axis=1, inplace=True)

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
            # output_dataframe = id_frames
            # Keep wanted output features

            output_dataframe.reset_index(drop=True, inplace=True)

            # Normalize moment with subject weight ! ! !
            output_dataframe['knee_angle_r_moment'] /= float(subject_weight)

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
        subject_df['subject_weight'] = subject_weight
        subject_dataframes.append(subject_df)

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


def getGaitCycles(df, preprocess_EMG=False, p=1, dropHeader=False):
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

        # f, (ax1, ax2) = plt.subplots(2, 1)
        # ax1.plot(cycle['HeelStrike'], cycle['knee_angle_r_moment'])
        # ax2.plot(cycle['HeelStrike'], cycle['knee_angle_r'])
        # ax1.set_title("Knee moment (normalized)")
        # ax2.set_title("Knee angle (not normalized)")
        # ax2.set_xlabel("% of Gait Cycle completion")
        # ax1.set_ylabel("Nm/kg")
        # ax2.set_ylabel("degrees")
        # plt.tight_layout()
        # plt.show()
        if not cycle.empty and cycle.shape[0] >= 200:
            cycles.append(cycle)

    # Keep the % of cycles we want
    cycles = random.sample(cycles, int(p * len(cycles)))

    # Filter out dataframes with NaN values for stair trials
    if preprocess_EMG:
        print("Processing EMGs. . .")
        cycles = process_EMGs(cycles)
        print("Finished")

    interp_cycles = []
    features = []
    for l in training_input.values():
        features.extend(l)
    columns_to_interp = features + ['Header'] + output_features
    common_x_axis = np.linspace(start=0.0, stop=100.0, num=200)

    for i, cycle in enumerate(cycles):
        cycle = cycle.reset_index(drop=True)
        interp_cycle = pd.DataFrame()
        # Interpolate all columns according to HeelStrike %
        for column_name in columns_to_interp:
            to_interp = cycle[column_name]
            interp_cycle[column_name] = np.interp(common_x_axis, cycle['HeelStrike'],
                                                  cycle[column_name])
        interp_cycle['subject_num'] = cycle['subject_num'].iloc[:]
        interp_cycle['subject_weight'] = cycle['subject_weight'].iloc[:]

        interp_cycle['x_common'] = common_x_axis
        interp_cycles.append(interp_cycle)
        # Original vs Interpolated plot
        # f, (ax1, ax2) = plt.subplots(2, 1)
        # ax1.scatter(cycle['HeelStrike'], cycle['knee_angle_r'], s=5)
        # ax2.scatter(interp_cycle['x_common'], interp_cycle['knee_angle_r'], s=5)
        # ax1.set_title("Knee moment sample before interpolation (not normalized)")
        # ax2.set_title("After interpolation (n = 200 points)")
        # plt.tight_layout()
        # plt.show()
    # Distribution of gait cycle lengths plot
    # lengths = [len(cycle) for cycle in cycles]
    # sns.histplot(data=lengths)
    # plt.title("Distribution of gait cycle lengths")
    # plt.xlabel("timesteps")
    # plt.show()

    return cycles, interp_cycles


def process_EMGs(cycles):
    # Create lowpass butteworth 4th order filter for smoothing
    devices = training_input['emg']
    cutoff_freq = 6  # Hz
    sampling_rate = 200  # Hz
    normalized_cutoff = cutoff_freq / (0.5 * sampling_rate)
    processed_cycles = []
    b, a = butter(4, normalized_cutoff, btype='low', analog=False)
    for cycle in cycles:
        # Split EMG and Output

        EMG_raw = cycle[devices]
        rest_of_cycle = cycle.drop(columns=devices, axis=1)
        # Rectify
        EMG_rectified = EMG_raw.abs()
        # Smooth each column seperately with the butteworth filter
        EMG_smoothed = pd.DataFrame()

        for column_name in EMG_rectified.columns:
            EMG_smoothed[column_name] = filtfilt(b, a, EMG_rectified[column_name])

        rest_of_cycle.reset_index(inplace=True, drop=True)
        # Update the original cycles list with the new cycle
        processed_cycles.append(pd.concat([EMG_smoothed, rest_of_cycle], axis=1))

    return processed_cycles


# Opensim Batch Processing Functions
def create_inverse_dynamics_tool_xml(cwd, subject_num, angles_df):
    start = angles_df['time'].iloc[0]
    end = angles_df['time'].iloc[-1]
    # Path to subject specific model
    model_file_path = os.path.join(cwd, 'Epic_Lab_Dataset', subject_num, 'osimxml', subject_num + '.osim')
    # Create the root element
    root = ET.Element("OpenSimDocument")
    root.set("Version", "40000")

    # Create the InverseDynamicsTool element
    inverse_dynamics_tool = ET.SubElement(root, "InverseDynamicsTool")

    # Create child elements and set their text values
    results_directory = ET.SubElement(inverse_dynamics_tool, "results_directory")
    results_directory.text = os.getcwd()

    input_directory = ET.SubElement(inverse_dynamics_tool, "input_directory")

    model_file = ET.SubElement(inverse_dynamics_tool, "model_file")
    model_file.text = model_file_path

    time_range = ET.SubElement(inverse_dynamics_tool, "time_range")
    time_range.text = f'{start} {end}'

    forces_to_exclude = ET.SubElement(inverse_dynamics_tool, "forces_to_exclude")
    forces_to_exclude.text = "Muscles"

    external_loads_file = ET.SubElement(inverse_dynamics_tool, "external_loads_file")
    external_loads_file.text = 'ExtLoads.xml'

    coordinates_file = ET.SubElement(inverse_dynamics_tool, "coordinates_file")
    coordinates_file.text = os.path.join(cwd, 'angles.sto')

    lowpass_cutoff_frequency_for_coordinates = ET.SubElement(
        inverse_dynamics_tool, "lowpass_cutoff_frequency_for_coordinates"
    )
    lowpass_cutoff_frequency_for_coordinates.text = "-1"

    output_gen_force_file = ET.SubElement(inverse_dynamics_tool, "output_gen_force_file")
    output_gen_force_file.text = "opensim_output.sto"

    joints_to_report_body_forces = ET.SubElement(inverse_dynamics_tool, "joints_to_report_body_forces")

    output_body_forces_file = ET.SubElement(inverse_dynamics_tool, "output_body_forces_file")
    output_body_forces_file.text = "Unassigned"

    # Create the XML tree
    tree = ET.ElementTree(root)

    # Create a string representation of the XML
    xml_str = ET.tostring(root, encoding="utf-8")

    # Apply indents using minidom
    dom = minidom.parseString(xml_str)
    pretty_xml_str = dom.toprettyxml(indent="  ")

    # Write the XML file with indents
    with open("IDsetup.xml", "w", encoding="utf-8") as xml_file:
        xml_file.write(pretty_xml_str)


def write_sto_file(forces_df, file_name):
    # Open the file for writing
    with open(file_name, 'w') as file:
        # Write the header information
        file.write(f'{file_name}\n')
        file.write('version=1\n')
        file.write(f'nRows={len(forces_df)}\n')
        file.write(f'nColumns={len(forces_df.columns)}\n')
        file.write('inDegrees=yes\n')
        file.write('endheader\n')

        file.write('\t'.join(forces_df.columns) + '\n')

        # Write the data
        forces_df.to_csv(file, sep='\t', index=False, header=False)
    full_path = os.path.join(os.getcwd(), file_name)
    return full_path


def read_sto_file(sto_file_path):
    # Read the STO file as text
    with open(sto_file_path, 'r') as file:
        sto_lines = file.readlines()

    # Extract column names from the first line
    column_names = sto_lines[6].strip().split('\t')

    # Extract data rows
    data_rows = [line.strip().split('\t') for line in sto_lines[7:]]
    data_clean = []
    for row in data_rows:
        clean_row = [float(s) for s in row]
        data_clean.append(clean_row)
    # Create a DataFrame
    df = pd.DataFrame(data_clean, columns=column_names)

    return df


def filter_ID(id_df, cutoff_freq):
    # The scipy filtfilt function applies a filter twice, once forward and once reverse, to cancel out phase shift applied from the filter
    filtered = id_df.copy()
    k = math.pow(math.sqrt(2.0) - 1.0, -0.25)
    # f = id_df.iloc[:, 0].diff().mean()
    b, a = butter(2, k * cutoff_freq, fs=100.0)  # fs = 200 hz because 1/200 is 0.005 which is the sampling rate

    filtered.iloc[:] = filtfilt(b, a, filtered.iloc[:], axis=0)

    return filtered
