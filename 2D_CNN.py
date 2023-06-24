import matplotlib.pyplot as plt

from gatherData import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from torch.utils.data import DataLoader, TensorDataset, Dataset

# Check if CUDA is available
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(torch.cuda.get_device_name(0))


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
    input_features = [i for sublist in list(training_input.values()) for i in sublist]
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


def getTrainTestCycles(all_cycles, scale=None):
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


def fit_scale(training_cycles, scale):
    all = pd.concat(training_cycles)
    scale.fit(all)
    return scale


def get_normalized_cycles(cycles, scale):
    norm_cycles = []
    for cycle in cycles:
        norm_cycles.append(scale.transform(cycle))
    return norm_cycles


class MyDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]

        # Perform any necessary preprocessing on the sequence and label
        # Convert the sequence and label to tensors or any other desired format

        # Example transformation: Convert sequence and label to tensors
        # All sequences are numpy arrays
        sequence_tensor = getTorchTensor(sequence)

        # All labels are dataframes
        label_tensor = getTorchTensor(label)

        return (sequence_tensor, label_tensor)


if __name__ == '__main__':
    cwd = os.getcwd()

    training_df = getTrainingData(cwd, subjects=subj_info, training_in=training_input, training_out=output_features,
                                  trial_types=['treadmill'],
                                  trial_feature=['stairHeight'], save_condition=True, dropNaN=False)
    # training_df = loadTrainingData(cwd, subj_info)
    training_df.dropna(inplace=True)
    # training_df.reset_index(drop=True, inplace=True)
    all_cycles, all_cycles_interp, _ = getGaitCycles(training_df, preprocess_EMG=True)
    # Data size
    p = 1
    all_cycles_interp = random.sample(all_cycles_interp, int(len(all_cycles_interp) * p))

    x_train, y_train, x_val, y_val, x_test, y_test = getTrainTestCycles(all_cycles_interp)

    batchsize = 16
    train_dataset = MyDataset(x_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=False)

    val_dataset = MyDataset(x_val, y_val)
    val_dataloader = DataLoader(val_dataset, batch_size=batchsize, shuffle=False)

    test_dataset = MyDataset(x_test, y_test)
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

            self.conv4 = nn.Conv2d(64, 64, kernel_size=(5, 1))
            self.dropout4 = nn.Dropout(0.3)

            self.fc1 = nn.Linear(4 * 64, 256)
            self.fc2 = nn.Linear(256, 256)
            self.fc3_1 = nn.Linear(256, sequence_length)
            self.fc3_2 = nn.Linear(256, sequence_length)
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

        def forward(self, output1, target1, output2, target2):
            mse1 = nn.MSELoss()(output1, target1)
            mse2 = nn.MSELoss()(output2, target2)
            total_loss = mse1 + mse2
            return total_loss, mse1, mse2


    length, width = x_train[0].shape

    model = CNN2(length, width).to(device)

    # criterion = nn.MSELoss().to(device)
    criterion = CustomLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # callback = EarlyStopCallback(model, patience=50, delta=0.001)

    # Training loop
    num_epochs = 300
    trainingstart = time.time()
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        ep_start = time.time()
        model.train()  # train mode
        epoch_total_loss = 0.0
        epoch_moment_loss = 0.0
        epoch_angle_loss = 0.0
        for i, (inputs, targets) in enumerate(train_dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()  # Zero the gradients
            outputs_moment, outputs_angle = model(inputs)
            loss, MSE_moment, MSE_angle = criterion(outputs_moment.squeeze(), targets[:, :, 0].squeeze(),
                                                    outputs_angle.squeeze(), targets[:, :, 1].squeeze())
            epoch_total_loss += loss.item()
            epoch_moment_loss += MSE_moment.item()
            epoch_angle_loss += MSE_angle.item()
            loss.backward()  # Compute gradients
            optimizer.step()  # Update model weights
        # Divide with number of batches
        epoch_total_loss /= len(train_dataloader)
        epoch_moment_loss /= len(train_dataloader)
        epoch_angle_loss /= len(train_dataloader)

        train_losses.append(epoch_total_loss)

        model.eval()  # eval mode for validation
        val_total_loss = 0.0
        val_moment_loss = 0.0
        val_angle_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_dataloader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                predictions_moment, predictions_angle = model(inputs)
                loss, MSE_moment, MSE_angle = criterion(predictions_moment.squeeze(), targets[:, :, 0].squeeze(),
                                                        predictions_angle.squeeze(), targets[:, :, 1].squeeze())

                val_total_loss += loss.item()
                val_moment_loss += MSE_moment.item()
                val_angle_loss += MSE_angle.item()

            val_total_loss /= len(val_dataloader)
            val_moment_loss /= len(val_dataloader)
            val_angle_loss /= len(val_dataloader)
            print(
                f'Epoch {epoch} : training total loss = {epoch_total_loss} / {epoch_moment_loss} / {epoch_angle_loss} | validation loss = {val_total_loss} / {val_moment_loss} / {val_angle_loss} | {time.time() - ep_start:.1f}s')
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

    all_targets = []
    all_predictions = []
    with torch.no_grad():
        for inputs, targets in test_dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss, MSE_moment, MSE_angle = criterion(outputs[0].squeeze(), targets[:, :, 0].squeeze(),
                                                    outputs[1].squeeze(), targets[:, :, 1].squeeze())
            total_loss += loss.item()
            moment_loss += MSE_moment.item()
            angle_loss += MSE_angle.item()

            all_targets.append(targets)
            all_predictions.append(outputs)
        total_loss /= len(test_dataloader)  # MSE
        moment_loss /= len(test_dataloader)
        angle_loss /= len(test_dataloader)
        print("MSE: ", total_loss, "Moment MSE:", moment_loss, "Angle MSE:", angle_loss)

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
    # Plot some example
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
