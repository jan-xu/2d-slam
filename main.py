import sys
from slam import SLAM

if __name__ == "__main__":
    # Verify user input arguments and initialize input data
    try:
        assert len(sys.argv) == 4
        algorithm = sys.argv[1].lower()
        imu_path = sys.argv[2]
        lid_path = sys.argv[3]
    except AssertionError:
        n_args = len(sys.argv) - 1
        print("\nExpected number of arguments: 3. Received: {0}. Correct usage of function:".format(n_args))
        print("´´´$ python3 main.py [algorithm] [path-to-IMU-data] [path-to-LiDAR-data]´´´\n")
        sys.exit()

    try:
        assert algorithm in ["feature", "icp"]
    except AssertionError:
        print("\nInvalid algorithm choice, choose between 'feature' or 'icp'. Correct usage of function:")
        print("´´´$ python3 main.py [algorithm] [path-to-IMU-data] [path-to-LiDAR-data]´´´\n")
        sys.exit()

    try:
        slam = SLAM(algorithm, imu_path, lid_path)
        slam.get_imu_data()
        slam.get_lidar_data()
    except:
        print("\nInvalid filename(s). Correct usage of function:")
        print("´´´$ python3 main.py [algorithm] [path-to-IMU-data] [path-to-LiDAR-data]´´´\n")
        sys.exit()

    # Visualize ground truth
    print("Plotting ground truth\n" + 50*"=")
    slam.plot_ground_truth()

    # Processing data
    slam.set_params()
    slam.initialize_arrays()
    slam.ekf()
    slam.postprocess()

    # Displaying results
    print("Plotting results\n" + 50*"=")
    slam.plot_results()
    print("Deviation errors\n" + 50*"=")
    print("RMSE for trajectory estimate:", slam.RMSE_traj, "m")
    if algorithm == "icp":
        print("RMSE for mapping estimate:", slam.RMSE_wall_icp, "m")
    if algorithm == "feature":
        print("RMSE for mapping estimate:", slam.RMSE_wall_feature, "m")
    print("Plotting trajectory error\n" + 50*"=")
    slam.plot_traj_error()
