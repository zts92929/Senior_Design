import numpy as np
import torch
from torch import nn
from collections import OrderedDict


class VisualComfortModel(nn.Module):
    def __init__(self, in_features, out_features, hidden_sizes, device):
        super(VisualComfortModel, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_sizes = hidden_sizes
        
        layer_sizes = [in_features] + hidden_sizes
        self.features = nn.Sequential(OrderedDict([('layer{0}'.format(i + 1),
            nn.Sequential(OrderedDict([
                ('linear', nn.Linear(hidden_size, layer_sizes[i + 1], bias=True)),
                ('tanh', nn.Tanh())
            ]))) for (i, hidden_size) in enumerate(layer_sizes[:-1])]))
        self.final_layer = nn.Linear(hidden_sizes[-1], out_features, bias=True) 
        self.tanh = nn.Tanh()
        
        self.loss_function = nn.MSELoss()
        self.device = device
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=0.002)

    def forward(self, inputs, params=None):
        features = self.features(inputs)
        output = self.final_layer(features)
        output = self.tanh(output)
        return output
    
    def load_torch_model(self, model_path):
        """ Load pre-trained torch model
        NOTE: This will not be used for the current project. 

        Args:
            model_path (str): The path for the pre-trained torch model
        """
        state_dict = torch.load(model_path)
        self.load_state_dict(state_dict=state_dict)
    
    def adapt(self, data, num_epochs=20):
        """ Adaptation of the visual comfort model to the personalized data

        Args:
            data (list): The data used to adapt the visual comfort model. 
                         This data should be a list of ndarraies and is of length 2.
                         The 1st element is the normalized brightness with the shape [num_data, 1]
                         The 2nd element is the normalized target visual sensation vote with the shape [num_data, 1] 
                            and support {-1, -2/3, -1/3, 0, 1/3, 2/3, 1}
            num_epochs (int, optional): number of adaptation epochs. Defaults to 20. 
                                        The reason why the number of adaptation epochs for the visual comfort model is
                                        larger than that for the thermal comfort model is that we do not have a pre-trained
                                        model for the visual comfort model. 
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
        print('The visual comfort model has been adapted to the provided data. \nThe training loss is {}.'.format(np.mean(loss_train)))
        
    def predict(self, brightness):
        """ Get the predicted personalized visual sensation vote(s) in specific value(s) of the brightness

        Args:
            brightness (ndarray): The brightness value(s) [num_data, 1]

        Returns:
            ndarray: The predicted personalized visual sensation vote(s) corresponding to the input condition(s)
        """
        self.eval()
        brightness = torch.from_numpy(brightness).type(torch.float32).to(self.device)
        with torch.no_grad():
            predicted_visual_sensations = self(brightness)
        return predicted_visual_sensations.cpu().detach().numpy()
    