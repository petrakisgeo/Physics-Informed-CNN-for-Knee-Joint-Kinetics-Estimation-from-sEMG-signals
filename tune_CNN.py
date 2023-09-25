from gatherData import *

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
import optuna

optuna.logging.set_verbosity(optuna.logging.INFO)

# Check if CUDA is available
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(torch.cuda.get_device_name(0))


def getInputOutput(training_dataframe):
    input_features = [i for sublist in list(training_input.values()) for i in sublist] + ['Header']
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


def getTorchTensor(data):
    if isinstance(data, pd.DataFrame):
        return torch.tensor(data.values, dtype=torch.float32)
    else:
        return torch.tensor(data, dtype=torch.float32)


def fit_scale(training_cycles, scale):
    all = pd.concat(training_cycles)
    scale.fit(all)
    return scale


def normalize_angles(y_train, y_test, y_val=None):
    # Fit to training data
    print("Normalizing angles . . .")
    train_data = pd.concat(y_train)
    sc = MinMaxScaler()
    train_angles = train_data['knee_angle_r'].values.reshape(-1, 1)
    sc.fit(train_angles)
    print("Done")

    def get_normalized_cycles(cycles, scale):
        norm_cycles = []
        for cycle in cycles:
            cycle = cycle.copy()
            angles = cycle['knee_angle_r'].values.reshape(-1, 1)
            normalized = scale.transform(angles)
            cycle['knee_angle_r'] = normalized.flatten()
            norm_cycles.append(cycle)
        return norm_cycles

    norm_train = get_normalized_cycles(y_train, sc)
    norm_test = get_normalized_cycles(y_test, sc)
    if y_val:
        norm_val = get_normalized_cycles(y_val, sc)
        return norm_train, norm_val, norm_test, sc
    return norm_train, norm_test, sc


class tunableCNN(nn.Module):
    def __init__(self, sequence_length, n_features, units_per_ff_layer):
        super(tunableCNN, self).__init__()
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

        self.fc_layers = nn.ModuleList()
        prev_units = 896  # Flattened output of convolutional layers
        self.fc_in = nn.Linear(896, 896)
        for units in units_per_ff_layer:
            # Add a layer
            self.fc_layers.append(nn.Linear(prev_units, units))
            # For next layer
            prev_units = units
        self.fc_out_1 = nn.Linear(prev_units, sequence_length)
        self.fc_out_2 = nn.Linear(prev_units, sequence_length)
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
        # b x 64 x 4 x 1
        x = x.view(x.size(0), -1)
        # b x 64*4
        x = self.relu(self.fc_in(x))
        for layer in self.fc_layers:
            x = layer(x)
            x = self.relu(x)
        x1 = self.fc_out_1(x)
        x2 = self.fc_out_2(x)
        return x1, x2


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, output1, target1, output2, target2):
        mse1 = nn.MSELoss()(output1, target1)
        mse2 = nn.MSELoss()(output2, target2)
        total_loss = mse1 + mse2
        return total_loss, mse1, mse2


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
        emg_devices = training_input['emg']
        sequence = sequence[emg_devices]
        label = label[output_features]
        # Example transformation: Convert sequence and label to tensors
        # All sequences are numpy arrays
        sequence_tensor = getTorchTensor(sequence)

        # All labels are dataframes
        label_tensor = getTorchTensor(label)

        return (sequence_tensor, label_tensor)


def train_model(model, criterion, optimizer, dataloader, num_epochs=150):
    model.train()  # train mode
    trainingstart = time.time()
    total_losses = []
    for epoch in range(num_epochs):
        epoch_total_loss = 0.0
        # epoch_moment_loss = 0.0
        # epoch_angle_loss = 0.0
        for i, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()  # Zero the gradients
            outputs_moment, outputs_angle = model(inputs)

            loss, MSE_moment, MSE_angle = criterion(outputs_moment.squeeze(), targets[:, :, 0].squeeze(),
                                                    outputs_angle.squeeze(), targets[:, :, 1].squeeze())
            epoch_total_loss += loss.item()
            # epoch_moment_loss += MSE_moment.item()
            # epoch_angle_loss += MSE_angle.item()
            loss.backward()  # Compute gradients
            optimizer.step()  # Update model weights
        # Divide with number of batches
        epoch_total_loss /= len(dataloader)
        # epoch_moment_loss /= len(train_dataloader)
        # epoch_angle_loss /= len(train_dataloader)

        total_losses.append(epoch_total_loss)
    print(f'Training completed after {time.time() - trainingstart:.1f}s')
    return model


def evaluate_model(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss, MSE_moment, MSE_angle = criterion(outputs[0].squeeze(), targets[:, :, 0].squeeze(),
                                                    outputs[1].squeeze(), targets[:, :, 1].squeeze())
            total_loss += loss
    total_loss /= len(dataloader)
    return total_loss


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


if __name__ == '__main__':
    cwd = os.getcwd()

    all_subjects_df = getTrainingData(cwd, subjects=subj_info, training_in=training_input, training_out=output_features,
                                      trial_types=['treadmill'],
                                      trial_feature=['Speed'], save_condition=True, dropNaN=False)
    # all_subjects_df = loadTrainingData(cwd, subj_info)
    # all_subjects_df.dropna(inplace=True)
    # all_subjects_df.reset_index(drop=True, inplace=True)

    subject_groups = all_subjects_df.groupby(['subject_num'])

    # Create a list of the subjects and their corresponding dataframes
    subject_numbers = list(subject_groups.groups.keys())


    def objective(trial):
        # Number of layers for this trial
        depth = trial.suggest_int('ff layers', 1, 3)
        hidden_units = []
        # Number of neurons per layer for this trial
        for i in range(depth):
            units = 2 ** trial.suggest_int(f"units_{i}", 5, 9)
            hidden_units.append(units)
        # 1**(learning_rate_exp) for this trial
        learning_rate = trial.suggest_float('learning rate', 1e-5, 1e-1, log=True)
        criterion = CustomLoss().to(device)

        total_losses_per_subject = []
        # Leave trials out training - evaluation scheme
        scenario = 'LTO'
        for subject_n in subject_numbers:
            if scenario == 'LSO':
                train_subjects = [subject_num for subject_num in subject_numbers if subject_num != subject_n]
                training_df = pd.concat([subject_groups.get_group(num) for num in train_subjects])
                test_df = subject_groups.get_group(subject_n)
            else:
                sub_df = subject_groups.get_group(subject_n)
                training_df, test_df = splitTrials(sub_df, 0.8)

            # Split into gait cycles, interpolate and preprocess EMGs
            _, training_cycles = getGaitCycles(training_df, preprocess_EMG=True)
            _, test_cycles = getGaitCycles(test_df, preprocess_EMG=True)
            # Split into input-output of NN and normalize outputs (moments already normalized by weight for every subject)
            x_train, y_train = getInputOutputCycles(training_cycles)
            x_test, y_test = getInputOutputCycles(test_cycles)

            y_train, y_test, sc = normalize_angles(y_train, y_test)

            # Create tensor dataset and shuffle the gait cycles

            batchsize = 16
            train_dataset = MyDataset(x_train, y_train)
            train_dataloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)

            test_dataset = MyDataset(x_test, y_test)
            test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
            # Create the CNN with the trial hyperparameters. once for each subject to evaluate on leave-trials-out results
            model = tunableCNN(200, 8, hidden_units).to(device)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            model = train_model(model, criterion, optimizer, train_dataloader, num_epochs=120)
            evaluation_loss = evaluate_model(model, test_dataloader, criterion)
            total_losses_per_subject.append(evaluation_loss)
        # Take the mean of these values. The optimal model will minimize the mean sum of MSE moment + MSE angle
        total_loss = sum(total_losses_per_subject) / len(total_losses_per_subject)
        return total_loss


    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)
    # Save study
    with open("study.pkl", "wb") as f:
        pickle.dump(study, f)
    # with open("study.pkl", "rb") as f:
    #     study = pickle.load(f, encoding='utf8')

    # Print the best hyperparameters and objective value
    print("Best hyperparameters: ", study.best_params)
    print("Best validation loss: ", study.best_value)

    fig = optuna.visualization.plot_optimization_history(study)
    fig.show()
    fig = optuna.visualization.plot_param_importances(study)
    fig.show()
    fig = optuna.visualization.plot_contour(study, params=['learning rate', 'ff layers'])
    fig.show()
