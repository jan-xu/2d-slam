# 2D LiDAR/INS SLAM with extended Kalman filter

**Author:** Jan Xu

Final Year (Masters) project carried out at CalUnmanned laboratory, Department of Civil and Environmental Engineering, University of California, Berkeley.

All code used for this project can be found in this repository.

## Context

As interest in autonomous robot navigation grows, Self-Localization and Mapping (SLAM) using low-cost range and inertial sensors is becoming ever-increasingly popular within the scientific community. The aim of this project was to implement SLAM algorithms by fusing odometry and pose data from an IMU with range data from a Light Detection and Ranging (LiDAR) device. A real-life experimental setup was constructed such that the sensor data is collected under conditions reflecting ground truth as close as possible. With this data, the state-space was then represented and manipulated with an extended Kalman filter, a simplified dynamical state transition model.

The experimental environment used for this project was a custom-built plywood maze with a 3D-printed sensor package box, seen below:

![alt text](https://raw.githubusercontent.com/jan-xu/2d-slam/master/png/IMG_3610.png "Experimental setup")

The dimensions of the model were carefully measured and then rendered numerically to establish ground truth:

![alt text](https://raw.githubusercontent.com/jan-xu/2d-slam/master/png/groundtruth.png "Ground truth")

For detailed information regarding the project, please refer to the [project report](https://raw.githubusercontent.com/jan-xu/2d-slam/master/report/Final Year Report.pdf "Project Report")

## Input data

- Context
- Input data
- Packages required: Numpy, Pandas, Matplotlib, Scikit-learn (for ICP)
- How to run
- Reference to report
- Images of results
- RMSE

# License

MIT
