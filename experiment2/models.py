import torch
from torch.utils.data import DataLoader
import torch.nn as nn


# Superclass for all spectrogram prediction models
class SpectrogramPredictionModel(torch.nn.Module):
    def __init__(self):
        super(SpectrogramPredictionModel, self).__init__()
    
    def reshape_input(self, x):
        raise NotImplementedError("reshape_input method must be implemented in the derived class.")

    def reshape_output(self, y):
        raise NotImplementedError("reshape_output method must be implemented in the derived class.")

    def forward(self, x):
        raise NotImplementedError("Forward method must be implemented in the derived class.")


# Fully Connected Network Class
class FCN(SpectrogramPredictionModel):
    def __init__(self, example_in, hidden_dim1, hidden_dim2):
        super(FCN, self).__init__()
        self.input_dim = example_in.shape[2]*example_in.shape[3]
        self.output_dim = example_in.shape[2]
        self.d_model = self.output_dim
        # Define the three fully connected layers
        self.fc1 = torch.nn.Linear(self.input_dim, hidden_dim1)  # First hidden layer
        self.fc2 = torch.nn.Linear(hidden_dim1, hidden_dim2)    # Second hidden layer
        self.fc3 = torch.nn.Linear(hidden_dim2, self.output_dim)  # Output layer
        self.dropout = torch.nn.Dropout(0.1)  # Optional dropout

    def reshape_input(self, x):
        x = x.view(x.size(0), -1)
        return x

    def reshape_output(self, y):
        y = y.view(y.size(0), -1)
        return y
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        return torch.nn.functional.sigmoid(self.fc3(x))


# whenever a new model is added, this has to be updated appropriately
MODEL_LOOKUP = {
    "FCN": FCN
}

def get_model(experiment_parameters, dataset):
    # the example datapoint is used to define the dimensions of a model
    ex_datapoint = dataset.select_clip(0)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    ex_input, ex_output = next(iter(dataloader))

    model = MODEL_LOOKUP[experiment_parameters["model"]]( ex_input, **experiment_parameters["model_parameters"])

    return model