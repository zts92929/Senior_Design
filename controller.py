import numpy as np
import torch
from torch import nn
from collections import OrderedDict


class FanController(nn.Module):
    def __init__(self, in_features=1, out_features=1, hidden_sizes=[16], device=torch.device('cpu')):
        """ Initialize the fan controller

        Args:
            in_features (int, optional): The input dimension of the controller. Defaults to 1.
                                         The default of the controller input is the thermal comfort difference 
                                         between the desired value and the predicted value of the personalized 
                                         thermal comfort model
            out_features (int, optional): The PWM voltage that is used to regulate the fan speed. Defaults to 1.
            hidden_sizes (list, optional): The hidden sizes of the fully connected neural network. 
                                           It decides the complexity of the neural network. Defaults to [16].
        """
        super(FanController, self).__init__()
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
        self.sigmoid = nn.Sigmoid()

        self.loss_function = nn.MSELoss()
        self.device = device
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=0.001)

    def forward(self, inputs, params=None):
        features = self.features(inputs)
        output = self.final_layer(features)
        output = self.sigmoid(output)
        return output
    
    def adapt(self, data, num_epochs=1):
        """ Adaptation of the fan controller to the personalized data

        Args:
            data (list): The data used to train the fan controller. 
                         This data should be a list of ndarraies and is of length 2.
                         The 1st element is the normalized thermal sensation diff with the shape [num_data, 1] and support [-1 ,1]
                         The 2nd element is the desired pwm voltage with the shape [num_data, 1] and support [0, 1]
                         TODO: how to determine the 2nd value? 
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
        print('The fan controller has been adapted to the provided data. \nThe training loss is {}.'.format(np.mean(loss_train)))
        
    def get_pwm_voltage(self, thermal_comfort_diff):
        """ Get the PWM voltage value(s) for specific value(s) of the thermal comfort discrepancy

        Args:
            thermal_comfort_diff (ndarray): The values of thermal comfort discrepancy with shape [num_data, 1]

        Returns:
            ndarray: The determined PWM voltage values for the thermal comfort discrepancy inputs
        """
        self.eval()
        thermal_comfort_diff = torch.from_numpy(thermal_comfort_diff).type(torch.float32).to(self.device)
        with torch.no_grad():
            pmv_voltage = self(thermal_comfort_diff)
        return pmv_voltage.cpu().detach().numpy()
    
    
class BulbController(nn.Module):
    def __init__(self, in_features=1, out_features=1, hidden_sizes=[16], device=torch.device('cpu')):
        """ Initialize the bulb controller to control the brightness

        Args:
            in_features (int, optional): The input dimension of the controller. Defaults to 1.
                                         The default of the controller input is the visual comfort difference 
                                         between the desired value and the predicted value of the personalized 
                                         visual comfort model
            out_features (int, optional): The PWM voltage that is used to regulate the brightness of bulb. Defaults to 1.
            hidden_sizes (list, optional): The hidden sizes of the fully connected neural network. 
                                           It decides the complexity of the neural network. Defaults to [16].
        """
        super(BulbController, self).__init__()
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
        self.sigmoid = nn.Sigmoid()

        self.loss_function = nn.MSELoss()
        self.device = device
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=0.001)

    def forward(self, inputs, params=None):
        features = self.features(inputs)
        output = self.final_layer(features)
        output = self.sigmoid(output)
        return output
    
    def adapt(self, data, num_epochs=1):
        """ Adaptation of the fan controller to the personalized data

        Args:
            data (list): The data used to train the fan controller. 
                         This data should be a list of ndarraies and is of length 2.
                         The 1st element is the normalized visual sensation diff with the shape [num_data, 1] and support [-1 ,1]
                         The 2nd element is the desired pwm voltage with the shape [num_data, 1] and support [0, 1]
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
        print('The bulb controller has been adapted to the provided data. \nThe training loss is {}.'.format(np.mean(loss_train)))
        
    def get_pwm_voltage(self, visual_comfort_diff):
        """ Get the PWM voltage value(s) for specific value(s) of the visual comfort discrepancy

        Args:
            visual_comfort_diff (ndarray): The values of visual comfort discrepancy with shape [num_data, 1]

        Returns:
            ndarray: The determined PWM voltage values for the visual comfort discrepancy inputs
        """
        self.eval()
        visual_comfort_diff = torch.from_numpy(visual_comfort_diff).type(torch.float32).to(self.device)
        with torch.no_grad():
            pmv_voltage = self(visual_comfort_diff)
        return pmv_voltage.cpu().detach().numpy()
