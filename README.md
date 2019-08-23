# 2D LiDAR/INS SLAM with extended Kalman filter

**Author:** Jan Xu

Final Year (Masters) project carried out at CalUnmanned laboratory, Department of Civil and Environmental Engineering, University of California, Berkeley.

All code used for this project can be found in this repository, written in Python 3.

# Context

As interest in autonomous robot navigation grows, Self-Localization and Mapping (SLAM) using low-cost range and inertial sensors is becoming ever-increasingly popular within the scientific community. The aim of this project was to implement SLAM algorithms by fusing odometry and pose data from an IMU with range data from a Light Detection and Ranging (LiDAR) device. A real-life experimental setup was constructed such that the sensor data is collected under conditions reflecting ground truth as close as possible. With this data, the state-space was then represented and manipulated with an extended Kalman filter, a simplified dynamical state transition model.

The system architecture for the SLAM procedure was visualized in the figure below. The blue and green blocks, “State Prediction” and “State Correction”, refer to the sensor fusion stages in the Kalman filter. Rectangular boxes with sharp corners represent physical quantities such as vectors, matrices or other types of data structures, whereas boxes with rounded corners describe various procedures.

![alt text](https://raw.githubusercontent.com/jan-xu/2d-slam/master/png/systemarchitecture.png "System architecture")

The experimental environment used for this project was a custom-built plywood maze with a 3D-printed sensor package box, seen below:

![alt text](https://raw.githubusercontent.com/jan-xu/2d-slam/master/png/IMG_3610.png "Experimental setup")

The dimensions of the model were carefully measured and then rendered numerically to establish ground truth:

![alt text](https://raw.githubusercontent.com/jan-xu/2d-slam/master/png/groundtruth.png "Ground truth")

For detailed information regarding the project, please refer to the [project report](https://raw.githubusercontent.com/jan-xu/2d-slam/master/report/FinalYearReport.pdf "Project Report").

# How to run

**Packages required:**
- Numpy
- Pandas
- Matplotlib
- Scikit-learn (if using ICP algorithm)

**Usage:**
```$ python3 main.py algorithm path-to-IMU-data path-to-LiDAR-data```

**Arguments:**
- ```algorithm```: LiDAR scan matching algorithms. Currently valid options: ```icp``` (Iterative Closest Point) and ```feature``` (feature-based line segment approach).
- ```path-to-IMU-data```: local path to .csv-file containing the IMU data
- ```path-to-LiDAR-data```: local path to .csv-file containing the LiDAR data

**Example:**
```$ python3 main.py icp sensordata/exp1_imu.csv sensordata/exp1_lidar.csv```

The folder ```sensordata``` contains all the datasets for the sensors, separated in nine different experimental runs.

# Output

The output from the main program will consist of a Matplotlib figure, plotting the estimated trajectory (dashed-dotted blue lines) and the experimental environment (purple dots/dashed lines). Two sample images, one using the ICP algorithm and one using the feature-based line segment approach for the scan matching procedure, are shown below.

![alt text](https://raw.githubusercontent.com/jan-xu/2d-slam/master/png/SLAM_ICP.png "EKF results using ICP algorithm")
![alt text](https://raw.githubusercontent.com/jan-xu/2d-slam/master/png/SLAM_feature.png "EKF results using feature-based scan matching algorithm")

Finally, the deviation error are printed out in terms of the root mean squared error (RMSE) in meters between estimate and ground truth for both the trajectory and the experimental environment. A Matplotlib figure showing the deviation error along the trajectory is also outputted before the program terminates:

![alt text](https://raw.githubusercontent.com/jan-xu/2d-slam/master/png/errorgraph.png "Graph of deviation error of estimated trajectory")

## Contact
[Jan Xu](mailto:jan.xu@berkeley.edu)

Telephone: +44 7763 524380

# License

MIT
