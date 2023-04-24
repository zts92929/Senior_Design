# Thermal Comfort Model and Fan Controller

In this repository, both the thermal comfort model and fan controller are modeled with neural networks. The reason why we do this is that the neural networks have good scalability. That is, we can make them more complicated when necessary. 

The neural networks need to be trained/adapted by some kinds of collected data. Thus, the scripts to import data from the database are needed. I listed the tasks in ```main.py``` in the following form. 

```
############# START #############

    """
        TODO: Some task description
    """
    Some scripts here
############## END ##############
```

Please let me know at any time if you do not understand the task or think this task is not compatible with the current senior design project. 

By the way, can someone give the used DynamoDB account (if it exists) so that I can make some tests? 

### NOTE TO ZACH (April 12, 2023)
Based on our discussions today, I added some new task descriptions in the ```main.py``` file. The additional information is about: 

(NOTE: The following mentioned csv file is used to adapt the thermal comfort model. Please see ```small_test_dataset.csv``` for an example.)

- Using different csv files for different individuals. In order to achieve this, the user id is also required to load from DynamoDB/UI. 
<br /> For example, there are two individuals A and B, then their respective csv files can be named as ```thermal_sensation_votes_A.csv``` and ```thermal_sensation_votes_B.csv```, respectively. 

- Using a trigger to decide when to save data to the csv file. This trigger is that the tested person has a thermal sensation vote. When this happens, we save both the collected vote and the measurements of sensors at that time to the corresponding csv file. 