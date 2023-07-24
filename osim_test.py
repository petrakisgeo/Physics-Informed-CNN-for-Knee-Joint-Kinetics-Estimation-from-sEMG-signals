import os

import pandas as pd

from gatherData import *

import opensim as osim


if __name__ == '__main__':
    cwd = os.getcwd()

    # training_df = getTrainingData(cwd, subjects=subj_info, training_in=training_input, training_out=output_features,
    #                               trial_types=['treadmill'],
    #                               trial_feature=['Speed'], save_condition=True, dropNaN=False)
    training_df = loadTrainingData(cwd,subj_info)
    # training_df.dropna(inplace=True)
    # training_df.reset_index(drop=True, inplace=True)

    cycles, cycles_interp, _ = getGaitCycles(training_df, preprocess_EMG=False)
    for i in range(200):
        training_df = cycles_interp[i]
        # training_df = loadTrainingData(cwd, subj_info)
        force_plates = training_input['fp']
        ik_angles = training_input['ik']
        forces_df = training_df[['Header'] + force_plates]
        angles_df = training_df[['Header'] + ik_angles]
        forces_df = forces_df.rename(columns={'Header': 'time'})
        angles_df = angles_df.rename(columns={'Header': 'time'})

        subj_name = training_df['subject_num'].iloc[0]

        forces_xml_path = write_sto_file(forces_df, 'forces.sto')
        angles_sto_path = write_sto_file(angles_df, 'angles.sto')

        create_inverse_dynamics_tool_xml(cwd,subj_name, angles_df)
        IDtool = osim.InverseDynamicsTool('IDsetup.xml')
        IDtool.run()

        data = read_sto_file('opensim_output.sto')
        filtered_data = filter_ID(data, 6.0)
        # filtered_data = filter_ID(filtered_data, 2 * 4) # 1/fs = 0.005s
        f, (ax1, ax2, ax3) = plt.subplots(3, 1)
        ax1.plot(data['time'], data['knee_angle_r_moment'])
        ax1.set_title('Unfiltered knee moment')
        ax2.plot(filtered_data['time'], filtered_data['knee_angle_r_moment'])
        ax2.set_title('Filtered knee moment')
        ax3.plot(training_df['Header'], training_df['knee_angle_r_moment'])
        ax3.set_title('Their result')
        plt.tight_layout()
        plt.show()
        print("")
