import torch
import numpy as np
import pandas as pd
from controller import FanController, BulbController
from thermal_comfort_model import ThermalComfortModel
from visual_comfort_model import VisualComfortModel
import os
import time
import serial
import math
from datetime import datetime
from huesdk import Hue
from flask import Flask, request, redirect
from flask_restful import Resource, Api
import ast

username=''
startMarker = '<'
endMarker = '>'
dataStarted = False
dataBuf = ""
messageComplete = False
username,hue,light,='','',''
thermal_comfort_data, thermal_feature_mean, thermal_feature_std, bulb_controller='','','',''
visual_comfort_data, visual_feature_mean, visual_feature_std, fan_controller='','','',''
visual_comfort_model,thermal_comfort_model, pid_controller_fan,pid_controller_bulb='','','',''
light_pwm,counter=75,0
data= {
    'ta':[],
    'tr':[],
    'rh':[],
    'vel':[0.15]*10,
    'met':[1]*10,
    'clo':[0.4]*10,
    'thermal_sensation':[],
}

data_light={'brightness':[],
            'visual_sensation': [],
}

app= Flask(__name__)
api= Api(app)

class Myresource(Resource):
    def post(self):
        global prev
        data=request.data
        data=data.decode()
        data = ast.literal_eval(data)
        temp_humid=read_arduino_data()

        sensation,visual_sense=data['temp:'],data['brightness:']
        write_data(temp_humid[0],temp_humid[1],sensation,visual_sense)

        return redirect('/')
    def get(self):
        global light_pwm
        temp_humid=read_arduino_data()
        print("BAMERON WANTS A GET")
        return ({
            "temp":temp_humid[0],
            "humidity":temp_humid[1],
            "light_pwm":light_pwm,

        })

class loginbutton(Resource):
    def post(self):
        global username, prev
        data=request.data
        data=data.decode()
        data = ast.literal_eval(data)
        username=data['SIGN IN E-MAIL:']

        #print(username)
        return redirect('/')

class run_loop(Resource):
    def post(self):
        return redirect('/')


api.add_resource(Myresource,'/button')
api.add_resource(loginbutton,'/login')
api.add_resource(run_loop,'/loop')


def write_data(temp,humidity,thermal_sense,visual_sense):
    global data, counter, data_light, light_pwm, username
    data['ta'].append(temp)
    data['tr'].append(temp)
    data['rh'].append(humidity)
    data['thermal_sensation'].append(int(float(thermal_sense)*6-3))
    data_light['brightness'].append(math.trunc(light_pwm * 100))
    data_light['visual_sensation'].append(int(float(visual_sense)*6-3))
    counter+=1

    #print(f"visual sensation {visual_sense}")
    #print(f"thermal_sense {thermal_sense}")
    #print(f"Thermal Sensation {int(float(thermal_sense)*6-3)}")
    #print(f"visual_sensation {float(visual_sense)*6-3}")

    if counter==10:
        counter=0
        df = pd.DataFrame(data).set_index('ta')
        df_light=pd.DataFrame(data_light).set_index('brightness')

        df_light.to_csv(f"visual_data_{username}_{datetime.now()}.csv")
        df.to_csv(f"thermal_data_{username}_{datetime.now()}.csv")

        print(f"Visual data written to visual_data_{username}_{datetime.now()}.csv")
        print(f"Thermal data written to thermal_data_{username}_{datetime.now()}.csv")
        data= {
            'ta':[],
            'tr':[],
            'rh':[],
            'vel':[0.15]*10,
            'met':[1]*10,
            'clo':[0.4]*10,
            'thermal_sensation':[],
        }
        data_light={'brightness':[],
                    'visual_sensation': [],
        }


def setup_lamp():
    global username, hue, light
    username = Hue.connect(bridge_ip = "10.0.0.113")
    hue = Hue("10.0.0.113", username)
    light = hue.get_light(name="4873_Bulb_1")

def adjust_lamp(num):
    global light, light_pwm
    light.set_brightness(light_pwm)


def read_sensation(user='zsmith47'):
    response=table.query(KeyConditionExpression=Key('username').eq(user))
    item=response[0]['Items']
    return item['sensation']


def setupSerial(baudRate, serialPortName):
    global  serialPort
    serialPort = serial.Serial(port= serialPortName, baudrate = baudRate, timeout=0, rtscts=True)
    print("Serial port " + serialPortName + " opened  Baudrate " + str(baudRate))
    waitForArduino()


def waitForArduino():
    # wait until the Arduino sends 'Arduino is ready' - allows time for Arduino reset
    # it also ensures that any bytes left over from a previous message are discarded
    print("Waiting for Arduino to reset")
    msg = ""
    while msg.find("Arduino is ready") == -1:
        msg = recvLikeArduino()
        if not (msg == 'XXX'):
            print(msg)


def recvLikeArduino():
    global startMarker, endMarker, serialPort, dataStarted, dataBuf, messageComplete

    if serialPort.inWaiting() > 0 and messageComplete == False:
        x = serialPort.read().decode("utf-8") # decode needed for Python3

        if dataStarted == True:
            if x != endMarker:
                dataBuf = dataBuf + x
            else:
                dataStarted = False
                messageComplete = True
        elif x == startMarker or x==' ':
            dataBuf = ''
            dataStarted = True

    if (messageComplete == True):
        messageComplete = False
        return dataBuf
    else:
        return "XXX"


def sendToArduino(num):
    # this adds the start- and end-markers before sending
    global startMarker, endMarker, serialPort
    fan_pwm = str(math.trunc(num * 100))

    stringWithMarkers = (startMarker)
    stringWithMarkers += stringToSend
    stringWithMarkers += (endMarker)

    print(stringWithMarkers)
    serialPort.write(stringWithMarkers.encode('utf-8')) # encode needed for Python3

class PIDController:
    """
    A discrete PID controller.
    """
    def __init__(self, proportional, integral, derivative):
        self.p = proportional
        self.i = integral
        self.d = derivative
        self.integral_value = 0
        self.previous_value = 0

    def get_pid_output(self, raw_input_):
        p_value = self.p * raw_input_
        self.integral_value += raw_input_
        i_value = self.i * self.integral_value
        d_value = self.d * (raw_input_ - self.previous_value)
        self.previous_value = raw_input_
        pid_output = p_value + i_value + d_value
        return pid_output


def synthetic_data_generator(num_data, type_data):
    if type_data == 'thermal':
        thermal_sensation_diff = np.random.uniform(low=-3.0, high=3.0, size=(num_data, 1))
        addtive_noises = np.random.normal(loc=0, scale=0.1, size=(num_data, 1))
        pwm_values_thermal = 0.5 + thermal_sensation_diff / 6.0 + addtive_noises
        pwm_values_thermal = np.clip(pwm_values_thermal, 0., 1.)
        synthetic_data = [thermal_sensation_diff, pwm_values_thermal]
    elif type_data == 'visual':
        visual_sensation_diff = np.random.uniform(low=-3.0, high=3.0, size=(num_data, 1))
        addtive_noises = np.random.normal(loc=0, scale=0.1, size=(num_data, 1))
        pwm_values_visual = 0.5 - visual_sensation_diff / 6.0 + addtive_noises
        pwm_values_visual = np.clip(pwm_values_visual, 0., 1.)
        synthetic_data = [visual_sensation_diff, pwm_values_visual]
    else:
        raise ValueError("The argument type_data should be either 'thermal' or 'visual'.")
    return synthetic_data


def load_votes_data(data_path, type_data):
    df = pd.read_csv(data_path)
    data = np.float32(df.to_numpy())
    num_data = len(data)
    feature_mean = np.mean(data[:, :-1], axis=0)
    feature_std = np.std(data[:, :-1], axis=0)
    if type_data == 'thermal':
        conditions = np.float32((data[:, :-1] - feature_mean) / feature_std)
    elif type_data == 'visual':
        conditions = np.float32(np.expand_dims((data[:, :-1] - feature_mean) / feature_std, axis=-1))
    votes = np.float32(np.expand_dims(data[:, -1] / 3.0, axis=-1))
    dataset = [conditions, votes]
    return dataset, feature_mean, feature_std


def normalize(unnormalized_data, loc, scale):
    normalized_data = (unnormalized_data - loc) / scale
    return normalized_data


def unnormalized(normalized_data, loc, scale):
    unnormalized_data = normalized_data * scale + loc
    return unnormalized_data


def read_arduino_data():
    while True:
        arduinoReply = recvLikeArduino()
        if not (arduinoReply == 'XXX'):
                        #print(arduinoReply)
            arduinoReply = arduinoReply.replace(' ','')
            arduinoReply = arduinoReply.split(',')
            temp = float(arduinoReply[0])
            humidity = float(arduinoReply[1])
                        #print(temp, humidity)
            return [temp,humidity]

def main(thermal_comfort_model_path, thermal_votes_dataset_path, visual_votes_dataset_path, device):
    global visual_comfort_data, visual_feature_mean, visual_feature_std, thermal_comfort_data, thermal_feature_mean, bulb_controller
    global thermal_feature_std, thermal_comfort_model, visual_comfort_model, pid_controller_fan, pid_controller_bulb, fan_controller
    # Build the thermal comfort model
    thermal_comfort_model = ThermalComfortModel(in_features=6, out_features=1, hidden_sizes=[128, 64], device=device)
    # Load the pre-trained thermal comfort model
    thermal_comfort_model.load_torch_model(model_path=thermal_comfort_model_path)
    thermal_comfort_model = thermal_comfort_model.to(device)

    # Build the visual comfort model
    visual_comfort_model = VisualComfortModel(in_features=1, out_features=1, hidden_sizes=[32], device=device)
    visual_comfort_model = visual_comfort_model.to(device)

    ################################################### START ############################################################
    # Load the personalized thermal sensation vote data
    """ TODO:
        In the current file, the personalized thermal sensation data are saved in the file 'small_test_dataset.csv'.
        However, in practice, these data come from the sensor measurements (e.g., the air temperatures) and DynamoDB
        (e.g., the thermal sensation votes). Thus, the task is to write some script here to load the personalized thermal
        sensation data as well as the corresponding environmental conditions from their corresponding sources.

        Since there are multiple data sources (sensors and DynamoDB), the loaded data can be saved in a csv file with
        the same format as 'small_test_dataset.csv'. Based on our discussions and my discussions with Dr. Zhang today
        (April 12), you may follow the specifictions as follows.

        1) WHEN TO SAVE DATA:
           The trigger to save data to a csv file is that the tested person provides a thermal sensation feedback (which
           is based on the 7-point thermal sensation votes). When the tested person gives us a thermal sensation vote,
           I assume this vote will be immediately saved in DynamoDB. While we get this newly saved vote in DynamoDB, we
           add a new row in the csv file, together with the measurements from sensors at the time the vote is provided.

        2) HOW TO HANDLE DIFFERENT INDIVIDUALS:
           In order to distinguish the thermal sensation votes from different individuals, we can create different csv data
           files. The individual's id can be included in the name of the csv data file so that we can find the correct data
           file when we would like to adapt the thermal sensation model.

           NOTE TO MYSELF (LIANGLIANG): One another consideration is that the adapted personalized thermal comfort model
                                        can be saved for future use. For example, the individual A works in the room on
                                        Monday, and we have already obtained a personalized thermal comfort model for A.
                                        On Tuesday, a different individual B works in the same room. In this case, we
                                        also need to learn a personalized thermal comfort model for B. In the same time,
                                        we can save the learned model for the individual A for future use. For example,
                                        when the individual A comes on Wednesday, we do not need to make the adaptation
                                        again.

        NOTE:
        In the current test dataset, 6 variables are considered to train the thermal dynamics model. I know that some
        of these variables cannot be measured. Here are some methods to handle the variables.
        Indoor air temperature (ta): I assume this can be meaured
        Mean radiant temperature (tr): Use the indoor air temperature instead
        Relative humidity (rh): I assume this can be obtained
        Air velocity (vel): This value can be estimated from the current fan speed
        Metabolic rate (met): You can use the value of 1.0
        Clothing insolation (clo): You can use the value of 0.4

        Thermal sensation votes (thermal_sensation): They need to be collected from the UI and save in the DynamoDB
                                                     so that they can be imported here when required.
    """
    thermal_comfort_data, thermal_feature_mean, thermal_feature_std = \
        load_votes_data(data_path=thermal_votes_dataset_path, type_data='thermal')
    visual_comfort_data, visual_feature_mean, visual_feature_std = \
        load_votes_data(data_path=visual_votes_dataset_path, type_data='visual')
    ##################################################### END ############################################################
    ### THERMAL COMFORT MODEL AND FAN CONTROLLER
    # Adapt the thermal comfort model with personalized thermal sensation data
    thermal_comfort_model.adapt(data=thermal_comfort_data)

    ### The following is for fan controller
    # The fan controller is defined as a general neural network.
    # Synthetic data are generated to train this neural network.
    # In practice, this dataset should from the DynamoDB. You can first write the scripts for adapt the thermal comfort model
    # After that, I will see how to import the data from DynamoDB
    thermal_data = synthetic_data_generator(num_data=500, type_data='thermal')
    fan_controller = FanController(in_features=1,
                                   out_features=1,
                                   hidden_sizes=[16],
                                   device=device)
    fan_controller = fan_controller.to(device)
    thermal_data[0] = normalize(unnormalized_data=thermal_data[0], loc=0, scale=3)
    # Adapt the fan control strategy using data
    fan_controller.adapt(data=thermal_data, num_epochs=20)

    # The PID control mechanisum is applied to the personalized thermal sensation predictions before getting the pwm voltage to control
    # the fan. Without this pid mechanisum, the thermal sensation votes will not have accumulative effects. For example, when an individual
    # provides -2 indicating he/she feels cool. In this case, the fan speed may be, for example, 0.02 m/s. In this case, if this individual
    # keeps provides -2, then the fan speed will not changed with the rule-based controller. However, for the PID controller, the integral
    # term will make sure the -2's provided by the occupant are accumulated. In addition, the derivative term may stablize the controller.
    pid_controller_fan = PIDController(proportional=1, integral=0.25, derivative=0.25)

    ### VISUAL COMFORT MODEL AND BULB CONTROLLER
    # Adapt the visual comfort model with personalized visual sensation data
    visual_comfort_model.adapt(data=visual_comfort_data)
    visual_data = synthetic_data_generator(num_data=500, type_data='visual')
    bulb_controller = BulbController(in_features=1,
                                     out_features=1,
                                     hidden_sizes=[16],
                                     device=device)
    bulb_controller = bulb_controller.to(device)
    visual_data[0] = normalize(unnormalized_data=visual_data[0], loc=0, scale=3)
    # Adapt the fan control strategy using data
    bulb_controller.adapt(data=visual_data, num_epochs=20)
    pid_controller_bulb = PIDController(proportional=1, integral=0.25, derivative=0.25)
    #THIS part below runs every 5 mintues
    setupSerial(9600, "/dev/tty.usbmodem14101")
    #setup_lamp()
    time.sleep(15)

    # dummy_indicator = 0
    # # After the adaptation of thermal comfort model, we can use it to make predictions under different conditions and the predictions
    # # will be used to decide the fan speed via the fan controller.
    # # In the current file, I used a dummy_indicator to avoid endless while loop. In practice, we can implement the scripts in the
    # # while loop EVERY 5 MINITES.

@app.route('/')
def main_loop():
    global visual_comfort_data, visual_feature_mean, visual_feature_std, thermal_comfort_data, thermal_feature_mean, thermal_feature_std, thermal_comfort_model,visual_comfort_model
    global pid_controller_fan, pid_controller_bulb, fan_controller, bulb_controller, light_pwm, prev
    #while True:
    # dummy_indicator = 1
    ################################################### START ############################################################
    # When we have new environmental and/or personal conditions, make predictions about the personalized thermal sensation votes
    # The following is some synthetic condition data
    """
        TODO: In practice, these data should be read from the sensors
        Task: Please write some scripts to import the real data from the sensors EVERY 5 MINUTES
    """
    # test_data_thermal_comfort_model = [np.asarray([32.5, 32.8600678239238, 42.1655555555556, 0.125060205635173, 1.2, 0.34]),
    #                                    np.asarray([33, 33.3599371513718, 42.8344444444444, 0.125060205635173, 1.2, 0.35]),
    #                                    np.asarray([33.5, 33.5000000000001, 42, 0.125060205635173, 1, 0.36])]
    # test_data_thermal_comfort_model = np.asarray([32.5, 32.8600678239238, 42.1655555555556, 0.125060205635173, 1.2, 0.34])

    temp_humid=read_arduino_data()
    # sensation = read_sensation()
    test_data_visual_comfort_model = np.asarray([light_pwm])
    test_data_thermal_comfort_model = np.asarray([temp_humid[0], temp_humid[0], temp_humid[1], 0.15, 1.0, 0.4])
    ##################################################### END ############################################################
    ### THERMAL COMFORT CONTROL
    test_data_thermal_comfort_model = np.asarray(test_data_thermal_comfort_model)
    normalized_test_data_thermal_comfort_model = normalize(unnormalized_data=test_data_thermal_comfort_model,
                                                           loc=thermal_feature_mean,
                                                           scale=thermal_feature_std)
    # Make thermal sensation vote predictions for the new conditions
    predicted_thermal_sensation_votes = 3 * thermal_comfort_model.predict(conditions=normalized_test_data_thermal_comfort_model)

    preferred_thermal_sensation_vote = 0
    # Calculate the difference between the predicted thermal sensation vote and the best thermal sensation vote, which is assumed
    # to be 0 temporarily
    thermal_comfort_diff = predicted_thermal_sensation_votes - preferred_thermal_sensation_vote
    thermal_comfort_diff_pid = pid_controller_fan.get_pid_output(raw_input_=thermal_comfort_diff)
    thermal_comfort_diff_normalized = normalize(unnormalized_data=thermal_comfort_diff_pid, loc=0, scale=3)
    thermal_comfort_diff_normalized = np.clip(thermal_comfort_diff_normalized, a_min=-1, a_max=1)
    # Get the PWM voltage value based on the thermal sensation vote discrepancy
    # This voltage should be implemented by the real fan DURING THE CONSIDERED 5 MINUTES
    thermal_test_pwm_voltage = fan_controller.get_pwm_voltage(thermal_comfort_diff_normalized)


    print(thermal_test_pwm_voltage[0])
    #sendToArduino(thermal_test_pwm_voltage[0])

    ### VISUAL COMFORT CONTROL
    test_data_visual_comfort_model = np.asarray(test_data_visual_comfort_model)
    normalized_test_data_visual_comfort_model = normalize(unnormalized_data=test_data_visual_comfort_model,
                                                          loc=visual_feature_mean,
                                                          scale=visual_feature_std)
    predicted_visual_sensation_votes = 3 * visual_comfort_model.predict(brightness=normalized_test_data_visual_comfort_model)

    preferred_visual_sensation_vote = 0
    visual_comfort_diff = predicted_visual_sensation_votes - preferred_visual_sensation_vote
    visual_comfort_diff_pid = pid_controller_fan.get_pid_output(raw_input_=visual_comfort_diff)
    visual_comfort_diff_normalized = normalize(unnormalized_data=visual_comfort_diff_pid, loc=0, scale=3)
    visual_comfort_diff_normalized = np.clip(visual_comfort_diff_normalized, a_min=-1, a_max=1)
    visual_test_pwm_voltage = bulb_controller.get_pwm_voltage(visual_comfort_diff_normalized)

    ### FOR BOTH FAN AND BULB CONTROLLERS
    # There are two PWM voltages that should be outputed.
    # They are controlling the fan speed and the bulb brightness, respectively.
    test_pwm_voltages = [thermal_test_pwm_voltage, visual_test_pwm_voltage]
    prev = time.time()
    print(visual_test_pwm_voltage[0])
    light_pwm=math.trunc(visual_test_pwm_voltage[0]*100)
    return("Done with function")
    #set_brightness(visual_test_pwm_voltage[0])
    #print('Test Done.')


if __name__ == '__main__':
    device = torch.device('cpu')
    thermal_comfort_model_path = 'thermal_comfort_model.th'
    thermal_votes_dataset_path = 'small_test_dataset_thermal.csv'
    visual_votes_dataset_path = 'small_test_dataset_visual.csv'
    main(thermal_comfort_model_path=thermal_comfort_model_path,
         thermal_votes_dataset_path=thermal_votes_dataset_path,
         visual_votes_dataset_path=visual_votes_dataset_path,
         device=device)
    app.run(host='0.0.0.0')
