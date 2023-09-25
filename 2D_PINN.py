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


def splitByRatio(starting_list, ratio=None, number=None):
    if ratio:
        ind = math.ceil(ratio * len(starting_list)) - 1
        first_part = starting_list[:ind]
        second_part = starting_list[ind:]
    if number:
        first_part = starting_list[:-number]
        second_part = starting_list[-number:]
    return first_part, second_part


def getInputOutput(training_dataframe):
    input_features = [i for sublist in list(training_input.values()) for i in sublist] + ['Header', 'subject_num',
                                                                                          'subject_weight']
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
        subject_trial_data['timedeltas'] = subject_trial_data['Header'].diff()
        # print(training_dataframe['timedeltas'].tail())
        # Trial switches when there is an abrupt change of timedeltas.
        trial_starts = [subject_trial_data.index[0]] + list(
            subject_trial_data[abs(subject_trial_data['timedeltas']) > 1].index)

        # Split the beginnings (trials) into train and test
        train, test = splitByRatio(trial_starts, ratio=r)

        test_df = subject_trial_data.loc[test[0]:]
        test_trials.append(test_df)
        if val_data_ratio:
            train, val = splitByRatio(train, ratio=val_data_ratio)
            train_df = subject_trial_data.loc[:val[0]]
            train_trials.append(train_df)
            val_df = subject_trial_data.loc[val[0]:test[0]]  # From the start of the val until the start of testing
            val_trials.append(val_df)
        else:
            train_df = subject_trial_data.loc[:test[0]]
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
        train_input = scale_data.fit_transform(train_input)
        val_input = scale_data.transform(val_input)
        test_input = scale_data.transform(test_input)

    return train_input, train_output, val_input, val_output, test_input, test_output


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


def process_batch_with_opensim_ID(cwd, angles_batch, forces_batch, subj_num, subj_weight, knee_angle_output_batch,
                                  moment_outputs_batch, moments_targets_batch, angle_scaler):
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
        # Replace the IK calculated angle with the angle calculated by the Neural Network
        f, ax1 = plt.subplots(1, 1)
        ax1.plot(angles_df['time'], angles_df['knee_angle_r'], label='Target angle')

        angles_df['knee_angle_r'].iloc[:] = output_knee_angle['knee_angle_r'].iloc[:]
        angles_df['knee_angle_r'] = filter_ID(angles_df['knee_angle_r'], 12.0)

        ax1.plot(angles_df['time'], angles_df['knee_angle_r'], label='NN angle')
        ax1.legend()
        plt.show()
        # Write the .sto files for IDtool input.
        _ = write_sto_file(angles_df, 'angles.sto')
        _ = write_sto_file(forces_df, 'forces.sto')
        create_inverse_dynamics_tool_xml(cwd, subj_num[idx], angles_df)
        # Run the ID simulation and read the results into a dataframe
        tool = osim.InverseDynamicsTool('IDsetup.xml')
        # with HideOutput():
        tool.run()

        # Read the output from the .sto file, make it into a tensor and store it in a list
        opensim_output = read_sto_file('opensim_output.sto')
        opensim_output['knee_angle_r_moment'] = filter_ID(opensim_output['knee_angle_r_moment'], 10.0)
        opensim_output.iloc[:, 1:] /= subj_weight[idx]
        steps = len(opensim_output)
        # Deal with interpolated angles with consecutive equal values
        if (steps < 100):
            final_row = opensim_output.iloc[-1, :]
            missing_rows = pd.DataFrame([final_row] * (100 - steps))
            opensim_output = pd.concat([opensim_output, missing_rows])
        nn_output = moment_outputs_batch[idx, :].cpu()
        real_output = moments_targets_batch[idx, :, 0].cpu()
        # Deal with outliers and non convergence of ID Tool
        if opensim_output['knee_angle_r_moment'].abs().max() > 3 * real_output.abs().max():
            opensim_output['knee_angle_r_moment'] = real_output[:]
        plt.plot(opensim_output['time'], opensim_output['knee_angle_r_moment'], label='physics output')

        moments_per_sample.append(getTorchTensor(opensim_output['knee_angle_r_moment'].copy()))
        plt.plot(opensim_output['time'], nn_output.detach().numpy(), label='NN output')
        plt.plot(opensim_output['time'], real_output.detach().numpy(), label='Target')
        plt.legend()
        plt.show()
    # Stack the tensors into one tensor for all the batch. Right form to feed torch MSE loss function
    output_moments = torch.stack(moments_per_sample, dim=0)

    return output_moments


def normalize_angles(y_train, y_val, y_test):
    # Fit to training data

    train_data = pd.concat(y_train)
    sc = StandardScaler()
    train_angles = train_data['knee_angle_r'].values.reshape(-1, 1)
    sc.fit(train_angles)

    def get_normalized_cycles(cycles, scale):
        norm_cycles = []
        for cycle in cycles:
            angles = cycle['knee_angle_r'].values.reshape(-1, 1)
            normalized_angles = scale.transform(angles)
            cycle['knee_angle_r'].iloc[:] = normalized_angles.flatten()
            norm_cycles.append(cycle)
        return norm_cycles

    norm_train = get_normalized_cycles(y_train, sc)
    norm_val = get_normalized_cycles(y_val, sc)
    norm_test = get_normalized_cycles(y_test, sc)

    return norm_train, norm_val, norm_test, sc

def get_maximum_output_range(y_test):
    # Concatenate all cycles into one dataframe to get the min and max of every column
    df = pd.concat(y_test)
    print('Moment max and min', df['knee_angle_r_moment'].max(), df['knee_angle_r_moment'].min())
    dy_moment = df['knee_angle_r_moment'].max() - df['knee_angle_r_moment'].min()
    print('Angle max and min', df['knee_angle_r'].max(), df['knee_angle_r'].min())
    dy_angle = df['knee_angle_r'].max() - df['knee_angle_r'].min()
    return dy_moment, dy_angle


if __name__ == '__main__':
    cwd = os.getcwd()

    # training_df = getTrainingData(cwd, subjects=subj_info, training_in=training_input, training_out=output_features,
    #                               trial_types=['treadmill'],
    #                               trial_feature=['Speed'], save_condition=True, dropNaN=False)

    training_df = loadTrainingData(cwd, subj_info)
    # training_df.dropna(inplace=True)
    # training_df.reset_index(drop=True, inplace=True)
    all_cycles, all_cycles_interp = getGaitCycles(training_df, preprocess_EMG=True, p=0.1)

    # Errors of Interpolate ? ? ?
    all_cycles_interp = [df for df in all_cycles_interp if df['Header'].is_monotonic_increasing and len(df) == 100]

    x_train, y_train, x_val, y_val, x_test, y_test = getTrainTestCycles(all_cycles_interp)
    # Normalize angles. Moments already normalized by subject weight
    y_train, y_val, y_test, angle_scaler = normalize_angles(y_train, y_val, y_test)

    batchsize = 16
    train_dataset = MyDataset_PINN(x_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=False)

    val_dataset = MyDataset_PINN(x_val, y_val)
    val_dataloader = DataLoader(val_dataset, batch_size=batchsize, shuffle=False)

    test_dataset = MyDataset_PINN(x_test, y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=batchsize, shuffle=False)


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
            # b x 100 x 8
            x = x.view(x.shape[0], 1, x.shape[1], x.shape[2])
            # b x 1 x 100 x 8
            x = self.relu(self.conv1(x))
            # b x 32 x 100 x 7
            x = self.dropout1(x)
            x = self.relu(self.maxpool2(self.conv2(x)))
            # b x 32 x 20 x 2
            x = self.dropout2(x)
            x = self.relu(self.maxpool3(self.conv3(x)))
            # b x 64 x 8 x 1
            x = self.dropout3(x)
            x = self.relu(self.conv4(x))
            x = self.dropout4(x)
            # b x 128 x 4 x 1
            x = x.view(x.size(0), -1)
            # b x 128
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


    length, width = 100, len(training_input['emg'])

    model = CNN2(length, width).to(device)

    # criterion = nn.MSELoss().to(device)
    criterion = CustomLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # callback = EarlyStopCallback(model, patience=50, delta=0.001)

    # Training loop
    num_epochs = 200
    trainingstart = time.time()
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        ep_start = time.time()
        model.train()  # train mode
        epoch_total_loss = 0.0
        epoch_moment_loss = 0.0
        epoch_angle_loss = 0.0
        epoch_physics_loss = 0.0
        for i, (inputs, targets, angles, forces, subj_num, subj_weight) in enumerate(train_dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()  # Zero the gradients
            outputs_moment, outputs_angle = model(inputs)
            if epoch > 5:
                opensim_output = process_batch_with_opensim_ID(cwd, angles, forces, subj_num, subj_weight,
                                                               outputs_angle,
                                                               outputs_moment, targets, angle_scaler)
                # opensim_output /= subj_weight.unsqueeze(1)
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
        epoch_total_loss /= len(train_dataloader)
        epoch_moment_loss /= len(train_dataloader)
        epoch_angle_loss /= len(train_dataloader)
        epoch_physics_loss /= len(train_dataloader)

        train_losses.append(epoch_total_loss)

        model.eval()  # eval mode for validation
        val_total_loss = 0.0
        val_moment_loss = 0.0
        val_angle_loss = 0.0
        with torch.no_grad():
            for (inputs, targets, _, _, _, _) in val_dataloader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                predictions_moment, predictions_angle = model(inputs)
                loss, MSE_moment, MSE_angle, _ = criterion(predictions_moment.squeeze(), targets[:, :, 0].squeeze(),
                                                           predictions_angle.squeeze(), targets[:, :, 1].squeeze())

                val_total_loss += loss.item()
                val_moment_loss += MSE_moment.item()
                val_angle_loss += MSE_angle.item()

            val_total_loss /= len(val_dataloader)
            val_moment_loss /= len(val_dataloader)
            val_angle_loss /= len(val_dataloader)
            print(
                f'Epoch {epoch} : training total loss = {epoch_total_loss} / {epoch_moment_loss} / {epoch_angle_loss} / {epoch_physics_loss} | validation loss = {val_total_loss} / {val_moment_loss} / {val_angle_loss} | {time.time() - ep_start:.1f}s')
            val_losses.append(val_total_loss)
        # callback(val_loss)
        #
        # if callback.early_stop:
        #     print(f'Early stopping! at epoch {epoch + 1} and restoring best weights')
        #     break
    print(f'Training finished after {time.time() - trainingstart:.1f}')

    plt.figure()
    epochs = [i for i in range(len(train_losses))]
    plt.plot(epochs, train_losses)
    plt.plot(epochs, val_losses)
    plt.title('Total MSE per epoch')
    plt.show()

    model_file = "CNN2D_EMG.pt"
    torch.save(model.state_dict(), model_file)

    # Testing loop
    old_model_state = torch.load('CNN2D_EMG.pt')
    model.load_state_dict(old_model_state)
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    moment_loss = 0.0
    angle_loss = 0.0
    CC_angle = 0.0
    CC_moment = 0.0

    all_targets = []
    all_predictions = []
    with torch.no_grad():
        for (inputs, targets, _, _, _, _) in test_dataloader:
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
            CC_moment += pearsonr(target_moment.flatten(), output_moment.flatten())[0]
            CC_angle += pearsonr(target_angle.flatten(), output_angle.flatten())[0]
            total_loss += loss.item()
            moment_loss += MSE_moment.item()
            angle_loss += MSE_angle.item()

            all_targets.append(targets)
            all_predictions.append(outputs)
        total_loss /= len(test_dataloader)  # MSE
        moment_loss /= len(test_dataloader)
        angle_loss /= len(test_dataloader)
        CC_moment /= len(test_dataloader)
        CC_angle /= len(test_dataloader)
        print("MSE: ", total_loss, "Moment MSE:", moment_loss, "Angle MSE:", angle_loss)
        print("CC moment:",CC_moment,"CC angle:", CC_angle)

    # Store the gait cycles together, not each batch together (unwrap the batch)
    all_targets = [tensor.cpu().squeeze() for batch in all_targets for tensor in batch]
    all_predictions_moment = [tensor.cpu().squeeze() for batch, _ in all_predictions for tensor in batch]
    all_predictions_angle = [tensor.cpu().squeeze() for _, batch in all_predictions for tensor in batch]
    # Calculate the average of the test gait cycles and predictions
    avg_targets = torch.mean(torch.stack(all_targets, dim=0), dim=0)
    avg_predictions_moment = torch.mean(torch.stack(all_predictions_moment, dim=0), dim=0)
    avg_predictions_angle = torch.mean(torch.stack(all_predictions_angle, dim=0), dim=0)

    f, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(avg_targets[:, 0], color='blue', label='avg_target')
    ax1.plot(avg_predictions_moment, color='red', label='avg_prediction')
    ax1.set_title("Average of R knee moment per GC")
    ax1.legend()
    ax2.plot(avg_targets[:, 1], color='blue', label='avg_target')
    ax2.plot(avg_predictions_angle, color='red', label='avg_prediction')
    ax2.set_title("Average of R knee angle per GC")
    ax2.legend()
    plt.tight_layout()
    plt.show()
    # Plot some examples
    for targets, predictions_moment, predictions_angle in zip(all_targets, all_predictions_moment,
                                                              all_predictions_angle):
        f, (ax1, ax2) = plt.subplots(2, 1)

        ax1.plot(targets[:, 0], color='blue', label='targets')
        ax1.plot(predictions_moment.cpu(), color='red', label='predictions')
        ax1.set_title("Right Knee Moment")
        ax1.legend()
        ax2.plot(targets[:, 1], color='blue', label='targets')
        ax2.plot(predictions_angle.cpu(), color='red', label='predictions')
        ax2.set_title("Right Knee Angle")
        ax2.legend()
        plt.tight_layout()
        plt.show()
