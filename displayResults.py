from gatherData import *

if __name__ == '__main__':
    # model_path = "FFNN256.h5"
    # stats_path = "FFNN256.pkl"
    # model = keras.models.load_model(model_path)
    # print(model.summary())
    cwd = os.getcwd()
    # alldata = getTrainingData(cwd, subjects=subj_info, training_in=training_input, training_out=output_features,
    #                           trial_types=['stair'], trial_feature=['stairHeight'], save_condition=True, dropNaN=True)
    alldata = loadTrainingData(cwd, subj_info)
    # training, display = dropDisplayFeatures(alldata)
    # inp = training.drop(columns=output_features)


    cycles, interp_cycles, avg_of_cycles = getGaitCycles(alldata,preprocess_EMG=True)
    for cycle, interp_cycle in zip(cycles, interp_cycles):
        f, (ax1, ax2) = plt.subplots(2, 1)
        ax1.plot(cycle['HeelStrike'], cycle['gastrocmed'])
        plt.title('Raw data')
        ax2.plot(interp_cycle['x_common'], interp_cycle['gastrocmed'])
        plt.title('Interpolated N=100 points')
        plt.tight_layout()
        plt.show()
        print('something')

    # alldata.loc[:, 'knee_r_moment_predictions'] = model.predict(inp)
    # Grouping by features needs to be done at this point
    # group_feature = 'stairHeight'
    # groups, names = groupbyFeature(alldata, feature=group_feature)
    # avg_list = []
    # for group, name in zip(groups, names):
    #     group.reset_index(drop=True, inplace=True)
    #     cycles, interp_cycles, avg_of_cycles = getGaitCycles(group)
    #     avg_list.append(avg_of_cycles)
    #     # dfperfigure = len(cycles) // 4
    #     # print(len(cycles))
    #     # for i in range(dfperfigure):
    #     #     plt.figure(i)
    #     plt.figure()
    #     for gc in cycles:
    #         plt.subplot(2, 1, 1)
    #         plt.plot(gc['HeelStrike'], gc['knee_angle_r_moment'])
    #         plt.title(group_feature + str(name))
    #         plt.subplot(2, 1, 2)
    #         plt.plot(gc['HeelStrike'], gc['knee_r_moment_predictions'])
    #         plt.title(group_feature + str(name) + "Predictions")
    #     plt.tight_layout()
    #     plt.show()
    #
    #     plt.figure()
    #     for gc in interp_cycles:
    #         plt.subplot(2, 1, 1)
    #         plt.plot(gc['x_common'], gc['knee_angle_r_moment'])
    #         plt.title(group_feature + str(name) + "Interpolated")
    #         plt.subplot(2, 1, 2)
    #         plt.plot(gc['x_common'], gc['knee_r_moment_predictions'])
    #         plt.title("Predictions Interpolated")
    #     plt.tight_layout()
    #     plt.show()
    #     # Plot the avg of the inteprolated ones
    # plt.figure()
    # for avg_of_cycles in avg_list:
    #     plt.subplot(2, 1, 1)
    #     plt.plot(avg_of_cycles['x_common'], avg_of_cycles['avg_real'])
    #     plt.title("Average per " + group_feature)
    #     plt.subplot(2, 1, 2)
    #     plt.plot(avg_of_cycles['x_common'], avg_of_cycles['avg_pred'])
    #     plt.title("Average predictions per " + group_feature)
    # plt.tight_layout()
    # plt.show()
