from gatherData import *


def getInputOutput(training_dataframe):
    input_features = [i for sublist in list(training_input.values()) for i in sublist] + ['Header'] + ['subject_num']
    x = training_dataframe[input_features]
    y = training_dataframe[output_features]

    return x, y

def getInputOutputCycles(cycles):
    cycles_in = []
    cycles_out = []

    for cycle in cycles:
        x, y = getInputOutput(cycle)
        cycles_in.append(x)
        cycles_out.append(y)
    return cycles_in, cycles_out

def normalize_angles(y_train, y_val, y_test):
    # Fit to training data

    train_data = pd.concat(y_train)
    sc = StandardScaler()
    sc.fit(train_data['knee_angle_r'])

    def get_normalized_cycles(cycles, scale):
        norm_cycles = []
        for cycle in cycles:
            cycle['knee_angle_r'].iloc[:] = scale.transform(cycle['knee_angle_r'])
            norm_cycles.append(cycle)
        return norm_cycles

    norm_train = get_normalized_cycles(y_train, sc)
    norm_val = get_normalized_cycles(y_val, sc)
    norm_test = get_normalized_cycles(y_test, sc)

    return norm_train, norm_val, norm_test, sc

def getTrainTestCycles(all_cycles, scale_angles=None):
    def custom_train_test_split(cycles_list, test_size=0.2):
        cycles_list = random.sample(cycles_list, len(cycles_list))
        train_size = 1 - test_size
        split_index = int(len(cycles_list) * train_size)
        train_cycles = cycles_list[:split_index]
        test_cycles = cycles_list[split_index:]
        return train_cycles, test_cycles

    train_cycles, test_cycles = custom_train_test_split(all_cycles, test_size=0.2)

    train_cycles, val_cycles = custom_train_test_split(train_cycles, test_size=0.1)

    train_x, train_y = getInputOutputCycles(train_cycles)

    test_x, test_y = getInputOutputCycles(test_cycles)

    val_x, val_y = getInputOutputCycles(val_cycles)

    return train_x, train_y, val_x, val_y, test_x, test_y

if __name__ == '__main__':

    cwd = os.getcwd()
    alldata = getTrainingData(cwd, subjects=subj_info, training_in=training_input, training_out=output_features,
                              trial_types=['stair'], trial_feature=['Speed'], save_condition=True, dropNaN=True)

    # alldata = loadTrainingData(cwd, subj_info)

    cycles, interp_cycles, avg_of_cycles = getGaitCycles(alldata, preprocess_EMG=True)

    for cycle, interp_cycle in zip(cycles, interp_cycles):
        f, (ax1, ax2) = plt.subplots(2, 1)
        ax1.plot(cycle['HeelStrike'], cycle['knee_angle_r'], '.')
        ax1.set_title('Original raw data')
        ax2.plot(interp_cycle['x_common'], interp_cycle['knee_angle_r'], '.')
        ax2.set_title('Interpolated (linear) data, N=100 points')
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
