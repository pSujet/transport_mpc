# Transport MPC Package
Predictive controller for load transportation in microgravity environment. We present centralized and decentralized MPC controllers based on the work of **Collaborative Load Transportation in Microgravity Environments: Centralized and Decentralized Predictive Controllers**.

<p align="center">
  <img src="https://github.com/pSujet/transport-mpc/blob/main/media/tethered.png" width="500"/>
</p>

## Requirements
This package was tested on a Ubuntu 20.04 LTS and Ubuntu 22.04 LTS with the following packages:
- Python 3.10
- CasADi 3.6.5
- matplotlib 3.5.1

To install the required python dependencies system-wide, please run:
```bash
pip install casadi==3.6.5
pip install -U matplotlib==3.5.1
```

**NOTE**: Please ensure that your system can run the proposed controllers in real-time. If the sampling frequency of 0.1s is not achievable, the controller performance will degrade. 

## Running Simulation in Python
1. Clone the repository:
```bash
git clone git@github.com:pSujet/transport_mpc.git
```
2. Run the demo of three agents with a centralized MPC controller:
```bash
python run_tracking_cen.py
```
3. Run the demo of three agents with a decentralized MPC controller:
```bash
python run_tracking_decen.py
```
4. Run the demo of three agents to compare each controller:
```bash
python run_tracking_compare.py
```
Press 'P' to pause the simulation and 'R' to resume.

<p align="middle">
  <img src="https://github.com/pSujet/transport-mpc/blob/main/media/3A.gif" width="300"/>
  <em>Centralized</em>
  <img src="https://github.com/pSujet/transport-mpc/blob/main/media/3A_red.gif" width="300"/>
  <em>Reduced</em>
  <img src="https://github.com/pSujet/transport-mpc/blob/main/media/3A_decen.gif" width="300"/>
  <em>Decentralized</em>
</p>


## Running Simulation in Gazebo (2D)
To run the simulation in Gazebo, you need to also install the simulation package: [discower_transportation](https://github.com/DISCOWER/discower_transportation).

To run a centralized controller
```
python start_controller.py 
```
To run a decentralized controller
```
ros2 launch transport-mpc start_controller_decen.launch.py
```
