from gatherData import *

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Check if CUDA is available
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(torch.cuda.get_device_name(0))

import opensim as osim

import os
import sys


class HideOutput(object):
    '''
    A context manager that block stdout for its scope, usage:

    with HideOutput():
        os.system('ls -l')
    '''

    def __init__(self, *args, **kw):
        sys.stdout.flush()
        self._origstdout = sys.stdout
        self._oldstdout_fno = os.dup(sys.stdout.fileno())
        self._devnull = os.open(os.devnull, os.O_WRONLY)

    def __enter__(self):
        self._newstdout = os.dup(1)
        os.dup2(self._devnull, 1)
        os.close(self._devnull)
        sys.stdout = os.fdopen(self._newstdout, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._origstdout
        sys.stdout.flush()
        os.dup2(self._oldstdout_fno, 1)


def getTorchTensor(data):
    if isinstance(data, pd.DataFrame):
        return torch.tensor(data.values, dtype=torch.float32)
    else:
        return torch.tensor(data, dtype=torch.float32)


class MyDataset_PINN(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]

        emg_devices = training_input['emg']
        force_plates = ['Header'] + training_input['fp']
        ik = ['Header'] + training_input['ik']

        nn_input = sequence[emg_devices].copy()
        opensim_angles = sequence[ik].to_numpy()
        opensim_forces = sequence[force_plates].to_numpy()
        subj_num = sequence['subject_num'].iloc[0]
        subj_weight = sequence['subject_weight'].iloc[0]

        # All sequences are numpy arrays
        nn_input_tensor = getTorchTensor(nn_input)

        # All labels are dataframes
        nn_output_tensor = getTorchTensor(label)

        return (nn_input_tensor, nn_output_tensor, opensim_angles, opensim_forces, subj_num, subj_weight)


class CNN2(nn.Module):

    def __init__(self, sequence_length, n_features):
        super(CNN2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, n_features), padding=(0, 3))
        self.dropout1 = nn.Dropout(0.3)

        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1))
        self.dropout2 = nn.Dropout(0.3)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(5, 3))

        self.conv3 = nn.Conv2d(32, 64, kernel_size=(5, 5), padding=(0, 2))
        self.dropout3 = nn.Dropout(0.3)
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv4 = nn.Conv2d(64, 64, kernel_size=(3, 1))
        self.dropout4 = nn.Dropout(0.3)

        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3_1 = nn.Linear(512, sequence_length)
        self.fc3_2 = nn.Linear(512, sequence_length)
        self.relu = nn.ReLU()

    def forward(self, x):
        # b x 200 x 8
        x = x.view(x.shape[0], 1, x.shape[1], x.shape[2])
        # b x 1 x 200 x 8
        x = self.relu(self.conv1(x))
        # b x 32 x 200 x 7
        x = self.dropout1(x)
        x = self.relu(self.maxpool2(self.conv2(x)))
        # b x 32 x 40 x 2
        x = self.dropout2(x)
        x = self.relu(self.maxpool3(self.conv3(x)))
        # b x 64 x 18 x 1
        x = self.dropout3(x)
        x = self.relu(self.conv4(x))
        x = self.dropout4(x)
        # b x 64 x 16 x 1
        x = x.view(x.size(0), -1)
        # b x 1024
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x1 = self.fc3_1(x)
        x2 = self.fc3_2(x)
        return x1, x2


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, output_m, target_m, output_a, target_a, physics_result=None):
        mse1 = nn.MSELoss()(output_m, target_m)
        mse2 = nn.MSELoss()(output_a, target_a)
        if physics_result is not None:
            physics_result = physics_result.to(device)
            physics_mse = nn.MSELoss()(physics_result,
                                       target_m)  # THIMISOU TO ALLAKSES GIA NA TESTAREIS GWNIES KANONIKA THELEI OUTPUTM
        else:
            physics_mse = 0
        l = 1.0
        total_loss = mse1 + mse2 + l * physics_mse
        return total_loss, mse1, mse2, physics_mse


def process_batch_with_opensim_ID(cwd, angles_batch, forces_batch, subj_num, subj_weight, knee_angle_output_batch,
                                  moment_outputs_batch, moments_targets_batch, angle_scaler, moment_scaler):
    # Create the output dataframe.
    subj_weight = subj_weight.numpy()
    moments_per_sample = []

    for idx in range(angles_batch.shape[0]):
        # Make the numpy arrays to dataframes with their corresponding column names to write the .sto files correctly
        # If this creates delay, write_sto_file function can be edited
        angles_df = pd.DataFrame(angles_batch[idx, :, :], columns=['time'] + training_input['ik'])
        forces_df = pd.DataFrame(forces_batch[idx, :, :], columns=['time'] + training_input['fp'])
        output_knee_angle = knee_angle_output_batch[idx].cpu()
        output_knee_angle = output_knee_angle.detach().numpy()
        output_knee_angle = output_knee_angle.reshape(-1, 1)
        # Un-normalize knee angle to insert to opensim
        output_knee_angle = angle_scaler.inverse_transform(output_knee_angle)
        output_knee_angle = pd.DataFrame(output_knee_angle, columns=['knee_angle_r'])
        # Replace the IK calculated angle with the angle calculated by the CNN
        # f, (ax1, ax2) = plt.subplots(1, 2)
        # ax1.plot(angles_df['time'], output_knee_angle['knee_angle_r'], label='Unfiltered')

        angles_df['knee_angle_r'].iloc[:] = output_knee_angle['knee_angle_r'].iloc[:]
        angles_df['knee_angle_r'] = filter_ID(angles_df['knee_angle_r'], 8.0)

        # ax2.plot(angles_df['time'], angles_df['knee_angle_r'], label='Filtered')
        # ax1.set_title("Unfiltered")
        # ax1.set_ylabel("Degrees")
        # ax1.set_xlabel("Time")
        # ax2.set_title("LP filtered")
        # ax2.set_xlabel("Time")
        # plt.show()
        # Write the .sto files for IDtool input.
        _ = write_sto_file(angles_df, 'angles.sto')
        _ = write_sto_file(forces_df, 'forces.sto')
        create_inverse_dynamics_tool_xml(cwd, subj_num[idx], angles_df)
        # Run the ID simulation and read the results into a dataframe
        # with HideOutput():
        tool = osim.InverseDynamicsTool('IDsetup.xml')
        tool.run()
        del tool
        # Read the output from the .sto file, make it into a tensor and store it in a list. Normalize with bw and minmaxscale
        opensim_output = read_sto_file('opensim_output.sto')
        opensim_output['knee_angle_r_moment'] = filter_ID(opensim_output['knee_angle_r_moment'], 4.0)
        opensim_output['knee_angle_r_moment'] /= subj_weight[idx]
        opensim_output['knee_angle_r_moment'] = moment_scaler.transform(
            opensim_output['knee_angle_r_moment'].values.reshape(-1, 1))
        steps = len(opensim_output)
        # Deal with interpolated angles with consecutive equal values
        if (steps < 200):
            final_row = opensim_output.iloc[-1, :]
            missing_rows = pd.DataFrame([final_row] * (200 - steps))
            opensim_output = pd.concat([opensim_output, missing_rows])
        # nn_output = moment_outputs_batch[idx, :].cpu()
        real_output = moments_targets_batch[idx, :, 0].cpu()
        # Deal with outliers and non convergence of ID Tool
        if opensim_output['knee_angle_r_moment'].abs().max() > 3 * real_output.abs().max():
            opensim_output['knee_angle_r_moment'] = real_output[:]
        # plt.plot(opensim_output['time'], opensim_output['knee_angle_r_moment'], label='OpenSim output')

        moments_per_sample.append(getTorchTensor(opensim_output['knee_angle_r_moment'].copy()))
        # plt.plot(opensim_output['time'], nn_output.detach().numpy(), label='CNN output')
        # plt.plot(opensim_output['time'], real_output.detach().numpy(), label='Target')
        # plt.legend()
        # plt.title("Moments")
        # plt.xlabel("Time")
        # plt.ylabel("Nm/kg")
        # plt.show()
    # Stack the tensors into one tensor for all the batch. Right form to feed torch MSE loss function
    output_moments = torch.stack(moments_per_sample, dim=0)
    return output_moments


def splitByRatio(starting_list, ratio=None, number=None):
    if ratio:
        ind = math.ceil(ratio * len(starting_list)) - 1
        first_part = starting_list[:ind]
        second_part = starting_list[ind:]
    if number:
        first_part = starting_list[:-number]
        second_part = starting_list[-number:]
    return first_part, second_part


def getInputOutput(training_dataframe, dropHeader=False):
    if dropHeader:
        input_features = [i for sublist in list(training_input.values()) for i in sublist]
    else:
        input_features = [i for sublist in list(training_input.values()) for i in sublist] + ['Header'] + [
            'subject_num'] + ['subject_weight']
    x = training_dataframe[input_features]
    y = training_dataframe[output_features]
    return x, y


def getInputOutputCycles(cycles, dropHeader=False):
    cycles_in = []
    cycles_out = []

    for cycle in cycles:
        x, y = getInputOutput(cycle, dropHeader)
        cycles_in.append(x)
        cycles_out.append(y)
    return cycles_in, cycles_out


def splitTrials(subject_df, r):
    all_trials = subject_df.copy()
    all_trials['timedeltas'] = all_trials['Header'].diff().abs()
    # Trial switches when there is an abrupt change of timedeltas.
    trial_starts = [all_trials.index[0]] + list(
        all_trials[all_trials['timedeltas'] > 1].index)

    def splitByRatio(starting_list, ratio=None, number=None):
        if ratio:
            ind = math.ceil(ratio * len(starting_list)) - 1
            first_part = starting_list[:ind]
            second_part = starting_list[ind:]
        if number:
            first_part = starting_list[:-number]
            second_part = starting_list[-number:]
        return first_part, second_part

    train_trial_starts, test_trials_starts = splitByRatio(trial_starts, ratio=r)
    train_trials = all_trials.loc[:test_trials_starts[0]]
    test_trials = all_trials.loc[test_trials_starts[0]:]
    return train_trials, test_trials


def get_maximum_output_range(y_test):
    # Concatenate all cycles into one dataframe to get the min and max of every column
    df = pd.concat(y_test)
    print('Moment max and min', df['knee_angle_r_moment'].max(), df['knee_angle_r_moment'].min())
    dy_moment = df['knee_angle_r_moment'].max() - df['knee_angle_r_moment'].min()
    print('Angle max and min', df['knee_angle_r'].max(), df['knee_angle_r'].min())
    dy_angle = df['knee_angle_r'].max() - df['knee_angle_r'].min()
    return dy_moment, dy_angle


def get_normalized_cycles(cycles, scale, column_names=None):
    norm_cycles = []
    for cycle in cycles:
        norm_cycle = cycle.copy()
        if len(column_names) < 2:
            norm_cycle[column_names] = scale.transform(cycle[column_names].values.reshape(-1, 1))
        else:
            norm_cycle[column_names] = scale.transform(cycle[column_names].values)
        norm_cycles.append(norm_cycle)
    return norm_cycles


def normalize_minmax(x_train, x_test, y_train, y_test):
    print("Scaling inputs and outputs . . .", end='')
    # Create different scale objects for each signal
    sc_emg = MinMaxScaler()
    sc_angles = MinMaxScaler()
    sc_moments = MinMaxScaler()
    # Fit each scale object to the train data values
    input_data = pd.concat(x_train)
    emg_devices = training_input['emg']
    emg_data = input_data[emg_devices]
    y_data = pd.concat(y_train)
    sc_emg.fit(emg_data.values)
    sc_angles.fit(y_data['knee_angle_r'].values.reshape(-1, 1))
    sc_moments.fit(y_data['knee_angle_r_moment'].values.reshape(-1, 1))

    # Get normalized results
    x_train = get_normalized_cycles(x_train, sc_emg, column_names=emg_devices)
    x_test = get_normalized_cycles(x_test, sc_emg, column_names=emg_devices)
    # Normalize angles and then normalize again the moments with the appropriate scalers
    y_train = get_normalized_cycles(y_train, sc_angles, column_names=['knee_angle_r'])
    y_train = get_normalized_cycles(y_train, sc_moments, column_names=['knee_angle_r_moment'])
    y_test = get_normalized_cycles(y_test, sc_angles, column_names=['knee_angle_r'])
    y_test = get_normalized_cycles(y_test, sc_moments, column_names=['knee_angle_r_moment'])
    print("Done")
    return x_train, x_test, y_train, y_test, sc_angles, sc_moments


def train_model(model, criterion, optimizer, dataloader, angle_scaler, moment_scaler, num_epochs=150):
    total_losses = []
    trainingstart = time.time()
    for epoch in range(num_epochs):
        model.train()  # train mode
        epoch_total_loss = 0.0
        epoch_moment_loss = 0.0
        epoch_angle_loss = 0.0
        epoch_physics_loss = 0.0
        for i, (inputs, targets, angles, forces, subj_num, subj_weight) in enumerate(dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()  # Zero the gradients
            outputs_moment, outputs_angle = model(inputs)
            if epoch > 15:
                opensim_output = process_batch_with_opensim_ID(cwd, angles, forces, subj_num, subj_weight,
                                                               outputs_angle,
                                                               outputs_moment, targets, angle_scaler, moment_scaler)
                loss, MSE_moment, MSE_angle, MSE_physics = criterion(outputs_moment.squeeze(),
                                                                     targets[:, :, 0].squeeze(),
                                                                     outputs_angle.squeeze(),
                                                                     targets[:, :, 1].squeeze(),
                                                                     opensim_output.squeeze())
                epoch_physics_loss += MSE_physics.item()
            else:
                loss, MSE_moment, MSE_angle, MSE_physics = criterion(outputs_moment.squeeze(),
                                                                     targets[:, :, 0].squeeze(),
                                                                     outputs_angle.squeeze(),
                                                                     targets[:, :, 1].squeeze())
            epoch_total_loss += loss.item()
            epoch_moment_loss += MSE_moment.item()
            epoch_angle_loss += MSE_angle.item()

            loss.backward()  # Compute gradients
            optimizer.step()  # Update model weights
        # Divide with number of batches
        epoch_total_loss /= len(dataloader)
        epoch_moment_loss /= len(dataloader)
        epoch_angle_loss /= len(dataloader)
        epoch_physics_loss /= len(dataloader)

        total_losses.append(epoch_total_loss)
    print(f'Training completed after {time.time() - trainingstart:.1f}s')
    return total_losses


def evaluate_model(model, dataloader, criterion):
    start = time.time()
    model.eval()
    moment_RMSE = 0.0
    angle_RMSE = 0.0
    CC_angle = 0.0
    CC_moment = 0.0
    results_moment = []
    results_angle = []
    targets_moment = []
    targets_angle = []

    with torch.no_grad():
        for (inputs, targets, _, _, _, _) in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss, MSE_moment, MSE_angle, _ = criterion(outputs[0].squeeze(), targets[:, :, 0].squeeze(),
                                                       outputs[1].squeeze(), targets[:, :, 1].squeeze())
            output_moment = outputs[0].cpu().detach().numpy()
            output_angle = outputs[1].cpu().detach().numpy()
            target_np = targets.squeeze().transpose(0, 1).cpu().detach().numpy()
            target_moment = target_np[0]
            target_angle = target_np[1]
            moment_RMSE += torch.sqrt(MSE_moment).item()
            angle_RMSE += torch.sqrt(MSE_angle).item()
            CC_moment += pearsonr(target_moment.flatten(), output_moment.flatten())[0]
            CC_angle += pearsonr(target_angle.flatten(), output_angle.flatten())[0]

    moment_RMSE /= len(dataloader)
    angle_RMSE /= len(dataloader)
    CC_moment /= len(dataloader)
    CC_angle /= len(dataloader)
    print("evaluated after ", time.time() - start)
    return moment_RMSE, angle_RMSE, CC_moment, CC_angle


if __name__ == '__main__':
    cwd = os.getcwd()

    all_subjects_df = getTrainingData(cwd, subjects=subj_info, training_in=training_input,
                                      training_out=output_features,
                                      trial_types=['treadmill'],
                                      trial_feature=['Speed'], save_condition=True, dropNaN=False)
    all_subjects_df = loadTrainingData(cwd, subj_info)
    all_subjects_df.dropna(inplace=True)
    all_subjects_df.reset_index(drop=True, inplace=True)
    # group the dataframe by subjectnum

    subject_groups = all_subjects_df.groupby(['subject_num'])

    # Create a list of the subjects and their corresponding dataframes
    subject_numbers = list(subject_groups.groups.keys())
    # Get the test subjects for every iteration of the cross-validation with combinations without repetition
    # test_subjects_per_iter = combinations(subject_numbers, 2)

    RMSE_moment_per_fold = []
    NRMSE_moment_per_fold = []
    RMSE_angle_per_fold = []
    NRMSE_angle_per_fold = []
    CC_moment_per_fold = []
    CC_angle_per_fold = []

    for subject_n in subject_numbers:
        if subject_n != 'ΑΒ13':
            continue
        sub_df = subject_groups.get_group(subject_n)
        training_df, test_df = splitTrials(sub_df, 0.8)

        # Split into gait cycles, interpolate and preprocess EMGs
        _, training_cycles = getGaitCycles(training_df, preprocess_EMG=True, p=0.2)
        _, test_cycles = getGaitCycles(test_df, preprocess_EMG=True)
        training_cycles = [df for df in training_cycles if df['Header'].is_monotonic_increasing and len(df) == 200]
        test_cycles = [df for df in test_cycles if df['Header'].is_monotonic_increasing and len(df) == 200]
        # Split into input-output of NN and normalize outputs (moments already normalized by weight for every subject)
        x_train, y_train = getInputOutputCycles(training_cycles, dropHeader=False)
        x_test, y_test = getInputOutputCycles(test_cycles, dropHeader=False)

        x_train, x_test, y_train, y_test, sc_angles, sc_moments = normalize_minmax(x_train, x_test, y_train, y_test)
        # y_train, y_test, _ = normalize_angles(y_train,y_test)
        # Create tensor dataset and shuffle the gait cycles
        dy_moments, dy_angles = get_maximum_output_range(y_test)

        batchsize = 16
        train_dataset = MyDataset_PINN(x_train, y_train)
        train_dataloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=False)

        test_dataset = MyDataset_PINN(x_test, y_test)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        # Create CNN model
        length, width = 200, len(training_input['emg'])

        model = CNN2(length, width).to(device)

        criterion = CustomLoss().to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.0015)
        epochs = 150
        # Train model on train subjects
        train_losses = train_model(model, criterion, optimizer, train_dataloader, sc_angles, sc_moments,
                                   num_epochs=epochs)
        # save_losses_plot(train_losses, epochs, subject_n)
        # Evaluate on test subjects
        moment_RMSE, angle_RMSE, CC_moment, CC_angle = evaluate_model(model, test_dataloader, criterion)
        moment_NRMSE = moment_RMSE / dy_moments * 100.0
        angle_NRMSE = angle_RMSE / dy_angles * 100.0

        print(
            f'RMSE moment: {moment_RMSE} CC moment: {CC_moment} NRMSE moment = {moment_NRMSE:.3f}%\n RMSE angle: {angle_RMSE} CC angle: {CC_angle} NRMSE angle = {angle_NRMSE:.3f}%')

        RMSE_moment_per_fold.append(moment_RMSE)
        NRMSE_moment_per_fold.append(moment_NRMSE)
        RMSE_angle_per_fold.append(angle_RMSE)
        NRMSE_angle_per_fold.append(angle_NRMSE)
        CC_moment_per_fold.append(CC_moment)
        CC_angle_per_fold.append(CC_angle)


    def get_mean_and_std(array):
        return np.mean(array), np.std(array)


    RMSE_moment_avg, RMSE_moment_std = get_mean_and_std(RMSE_moment_per_fold)
    NRMSE_moment_avg, NRMSE_moment_std = get_mean_and_std(NRMSE_moment_per_fold)
    RMSE_angle_avg, RMSE_angle_std = get_mean_and_std(RMSE_angle_per_fold)
    NRMSE_angle_avg, NRMSE_angle_std = get_mean_and_std(NRMSE_angle_per_fold)

    CC_moment_avg, CC_moment_std = get_mean_and_std(CC_moment_per_fold)
    CC_angle_avg, CC_angle_std = get_mean_and_std(CC_angle_per_fold)

    # print('\n\nLeave Trials Out cross validation 80-20 results:')
    print('\n\nLeave Trials Out 80-20 cross validation results:')
    print(f'RMSE_moment = {RMSE_moment_avg} ({RMSE_moment_std})')
    print(f'CC_moment = {CC_moment_avg} ({CC_moment_std})')
    print(f'RMSE_angle = {RMSE_angle_avg} ({RMSE_angle_std})')
    print(f'CC_angle = {CC_angle_avg} ({CC_angle_std})')
    print(f'NRMSE moment = {NRMSE_moment_avg:.3f}% ({NRMSE_moment_std:.3f})')
    print(f'NRMSE angle = {NRMSE_angle_avg:.3f}% ({NRMSE_angle_std:.3f})')

    rmse_df = pd.DataFrame({'moment': RMSE_moment_per_fold, 'angle': RMSE_angle_per_fold})
    nrmse_df = pd.DataFrame({'moment': NRMSE_moment_per_fold, 'angle': NRMSE_angle_per_fold})
    cc_df = pd.DataFrame({'moment': CC_moment_per_fold, 'angle': CC_angle_per_fold})

    plt.close('all')
    f, (ax1, ax2, ax3) = plt.subplots(1, 3)
    red_square = dict(markerfacecolor='r', marker='s')
    sns.boxplot(rmse_df, ax=ax1, flierprops=red_square)
    sns.boxplot(nrmse_df, ax=ax2, flierprops=red_square)
    sns.boxplot(cc_df, ax=ax3, flierprops=red_square)
    # Customize the plot
    ax1.set_title("RMSE")
    ax2.set_title("NRMSE %")
    ax3.set_title("Pearson's CC")

    # Show the plot
    plt.tight_layout()
    plt.show()
    to_file = [RMSE_moment_per_fold, RMSE_angle_per_fold, NRMSE_moment_per_fold, NRMSE_angle_per_fold,
               CC_moment_per_fold, CC_angle_per_fold]
    with open("results_2.pkl", "wb") as f:
        pickle.dump(to_file, f)
