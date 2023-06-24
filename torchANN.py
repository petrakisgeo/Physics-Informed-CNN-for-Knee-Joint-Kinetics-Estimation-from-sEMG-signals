from gatherData import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from torch.utils.data import DataLoader, TensorDataset
import optuna

# torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_default_dtype(torch.float64)

optuna.logging.set_verbosity(optuna.logging.INFO)

# Check if CUDA is available
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(torch.cuda.get_device_name(0))


def getInputOutput(training_dataframe):
    x = training_dataframe.drop(output_features, axis=1)
    y = training_dataframe[output_features]

    return x, y


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


def getTorchTensor(data):
    if isinstance(data, pd.DataFrame):
        return torch.tensor(data.values)
    else:
        return torch.tensor(data)


class FFNN(nn.Module):
    def __init__(self, n_features):
        super(FFNN, self).__init__()

        self.fc2 = nn.Linear(n_features, 256)
        self.batch2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(p=0.5)

        self.fc3 = nn.Linear(256, 128)
        self.batch3 = nn.BatchNorm1d(128)
        self.drop3 = nn.Dropout(p=0.5)

        self.fc4 = nn.Linear(128, 64)
        self.batch4 = nn.BatchNorm1d(64)
        self.drop4 = nn.Dropout(p=0.5)

        self.fc5 = nn.Linear(64, 1)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.batch2(self.fc2(x))
        x = self.relu(x)
        x = self.drop2(x)

        x = self.batch3(self.fc3(x))
        x = self.relu(x)
        x = self.drop3(x)

        x = self.batch4(self.fc4(x))
        x = self.relu(x)
        x = self.drop4(x)

        x = self.fc5(x)
        return x


class tunableFFNN(nn.Module):
    def __init__(self, n_features, depth, hidden_units):
        super(tunableFFNN, self).__init__()

        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        # self.dropouts = nn.ModuleList()

        prev_units = n_features
        for units in hidden_units:
            self.layers.append(nn.Linear(prev_units, units))
            self.batch_norms.append(nn.BatchNorm1d(units))
            # self.dropouts.append(nn.Dropout(p=0.5))
            prev_units = units

        self.fc_out = nn.Linear(hidden_units[-1], 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.batch_norms[i](self.layers[i](x))
            x = self.relu(x)
            # x = self.dropouts[i](x)

        x = self.fc_out(x)
        return x


if __name__ == '__main__':
    cwd = os.getcwd()

    # training_df = getTrainingData(cwd, subjects=subj_info, training_in=training_input, training_out=output_features,
    #                               trial_types=['treadmill'],
    #                               trial_feature=['Speed'], save_condition=True, dropNaN=False)
    training_df = loadTrainingData(cwd, subj_info)
    training_df.dropna(inplace=True)
    training_df, _ = dropDisplayFeatures(training_df)  # drops Header too

    sc = MinMaxScaler()
    x_train, y_train, x_val, y_val, x_test, y_test = getTrainTestData(training_df, out_features=output_features,
                                                                      scale_data=sc)

    train_features = getTorchTensor(x_train)
    train_targets = getTorchTensor(y_train)

    val_features = getTorchTensor(x_val)
    val_targets = getTorchTensor(y_val)

    test_features = getTorchTensor(x_test)
    test_targets = getTorchTensor(y_test)

    # Create DataLoader objects for batch processing
    batchsize = 64
    train_dataset = TensorDataset(train_features, train_targets)
    train_dataloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=False)

    val_dataset = TensorDataset(val_features, val_targets)
    val_dataloader = DataLoader(val_dataset, batch_size=batchsize, shuffle=False)

    test_dataset = TensorDataset(test_features, test_targets)
    test_dataloader = DataLoader(test_dataset, batch_size=batchsize, shuffle=False)


    def objective(trial):
        trial_start = time.time()
        # Set hyperparameter search space
        model = create_model(trial)
        print("Trial no:", trial.number)
        # Define loss function
        criterion = nn.MSELoss().to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        # Training loop
        num_epochs = 100

        with open("loss_logs.txt", "a") as file:
            file.write(
                f"\nModel parameters: Depth: {trial.params['depth']}, No of Units: {[2**trial.params[f'units_{i}'] for i in range(trial.params['depth'])]}\n\n")

        print(f"\nModel parameters: Depth: {trial.params['depth']}, No of Units: {[2**trial.params[f'units_{i}'] for i in range(trial.params['depth'])]}\n\n")
        for epoch in range(num_epochs):
            epoch_start = time.time()
            model.train()  # train mode
            epoch_loss = 0.0
            for i, (inputs, targets) in enumerate(train_dataloader):
                inputs = inputs.to(device)
                targets = targets.to(device)
                optimizer.zero_grad()  # Zero the gradients
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), targets.squeeze())
                epoch_loss += loss.item()
                loss.backward()  # Compute gradients
                optimizer.step()  # Update model weights

            epoch_loss /= len(train_dataloader)

            model.eval()  # eval mode for validation
            val_loss = 0.0

            with torch.no_grad():
                for inputs, targets in val_dataloader:
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    predictions = model(inputs)
                    loss = criterion(predictions.squeeze(), targets.squeeze())
                    val_loss += loss.item()
                val_loss /= len(val_dataloader)

            print(
                f"\tEpoch: {epoch + 1}, Training Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}\n | {time.time() - epoch_start:.1f}s\n")
            with open("loss_logs.txt", "a") as file:
                file.write(
                    f"\tEpoch: {epoch + 1}, Training Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}\n | {time.time() - epoch_start:.1f}s\n")
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        print(f'Finished after {time.time() - trial_start:.1f}s')
        return val_loss


    def create_model(trial):
        depth = trial.suggest_int("depth", 2, 4)
        hidden_units = []
        for i in range(depth):
            units = 2 ** trial.suggest_int(f"units_{i}", 6, 10)
            hidden_units.append(units)
        # learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
        # Create the FFNN model
        (inp, _) = next(iter(train_dataloader))
        input_dim = inp.shape[1]
        model = tunableFFNN(input_dim, depth, hidden_units).to(device)
        return model


    study = optuna.create_study(direction='minimize', pruner=optuna.pruners.HyperbandPruner())
    study.optimize(objective, n_trials=50)
    # Save study
    with open("study.pkl", "wb") as f:
        pickle.dump(study, f)

    # Print the best hyperparameters and objective value
    print("Best hyperparameters: ", study.best_params)
    print("Best validation loss: ", study.best_value)
    best_model = create_model(study.best_trial)

    best_model.eval()  # Set the model to evaluation mode
    test_loss = 0.0
    criterion = nn.MSELoss().to(device)
    with torch.no_grad():
        for inputs, targets in test_dataloader:
            inputs.to(device)
            targets.to(device)
            outputs = best_model(inputs)
            test_loss += criterion(outputs.squeeze(), targets.squeeze()).item()
    test_loss /= len(test_dataloader)
    print("Test Loss: ", test_loss)

    # Visualise stuff
    optuna.visualization.plot_optimization_history(study)
    optuna.visualization.plot_param_importances(study)
    optuna.visualization.plot_contour(study, params=['depth', 'learning_rate'])
