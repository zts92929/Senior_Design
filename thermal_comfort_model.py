import numpy as np
import torch
from torch import nn
from collections import OrderedDict


class ThermalComfortModel(nn.Module):
    def __init__(self, in_features, out_features, hidden_sizes, device):
        super(ThermalComfortModel, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_sizes = hidden_sizes

        layer_sizes = [in_features] + hidden_sizes
        self.features = nn.Sequential(OrderedDict([('layer{0}'.format(i + 1),
            nn.Sequential(OrderedDict([
                ('linear', nn.Linear(hidden_size, layer_sizes[i + 1], bias=True)),
                ('tanh', nn.Tanh())
            ]))) for (i, hidden_size) in enumerate(layer_sizes[:-1])]))
        self.classifier = nn.Linear(hidden_sizes[-1], out_features, bias=True)
        # The classifier is just a name so that the torch.load_state_dict works.
        self.tanh = nn.Tanh()

        self.loss_function = nn.MSELoss()
        self.device = device
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=0.002)

    def forward(self, inputs, params=None):
        features = self.features(inputs)
        output = self.classifier(features)
        output = self.tanh(output)
        return output

    def load_torch_model(self, model_path):
        """ Load pre-trained torch model
        
        Args:
            model_path (str): The path for the pre-trained torch model
        """
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        self.load_state_dict(state_dict=state_dict)

    def adapt(self, data, num_epochs=1):
        """ Adaptation of the thermal comfort model to the personalized data

        Args:
            data (list): The data used to adapt the thermal comfort model.
                         This data should be a list of ndarraies and is of length 2.
                         The 1st element is the normalized environmental and personal conditions with the shape [num_data, 6]
                         The 2nd element is the normalized target thermal sensation vote with the shape [num_data, 1]
                            and support {-1, -2/3, -1/3, 0, 1/3, 2/3, 1}
                         TODO: how to measure the conditions?
            num_epochs (int, optional): number of adaptation epochs. Defaults to 1.
        """
        self.train()
        loss_train = []
        inputs, outputs = data[0], data[1]
        inputs = torch.from_numpy(inputs).type(torch.float32).to(self.device)
        outputs = torch.from_numpy(outputs).type(torch.float32).to(self.device)
        for i in range(num_epochs):
            predictions = self(inputs)
            loss = self.loss_function(predictions, outputs)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_train.append(loss.item())
        print('The thermal comfort model has been adapted to the provided data. \nThe training loss is {}.'.format(np.mean(loss_train)))

    def predict(self, conditions):
        """ Get the predicted personalized thermal sensation vote(s) in specific value(s) of the conditions

        Args:
            conditions (ndarray): The environmental and personal conditions [num_data, 6]

        Returns:
            ndarray: The predicted personalized thermal sensation vote(s) corresponding to the input condition(s)
        """
        self.eval()
        conditions = torch.from_numpy(conditions).type(torch.float32).to(self.device)
        with torch.no_grad():
            predicted_thermal_sensations = self(conditions)
        return predicted_thermal_sensations.cpu().detach().numpy()
