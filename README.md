
## About

This repo is design for the final project of ROAS 6000H Human Centric Machine Perception and the code is designed for sim2sim and sim2real deploy of KungfuBot. The code is modify from [unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym). The training pipeline follows [KungfuBot](https://github.com/TeleHuman/PBHC).

<!-- ## Sim2sim deploy video

## Sim2real deploy video -->


## Installation

- Creat a virtual env and install KungfuBot. Refer to `INSTALL.md` in [KungfuBot](https://github.com/TeleHuman/PBHC) for environment setup and installation instructions. Since deploy code would import torch and other dependency embedded in isaacgym, so please **do not** skip installing isaacgym.

- Following the installation guidance to install python simulation [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco) in the same env. 

- Since the original simulator will only update the simulation step after policy inference, which do not align with the real world deployment. Please change the following files for better sim2real transfer.
    - Adjust `unitree_mujoco/simulate_python/config.py`, change `ROBOT_TYPE="g1"` and`SIMULATE_DT=0.002` to simulate real world state update.
    - Use `modified_mujoco/unitree_mujoco.py` in our repo to override `unitree_mujoco/simulate_python/unitree_mujoco.py`, which will add thread lock for simulating low level PD control.
    - Use `modified_mujoco/unitree_sdk2py_bridge.py` in our repo to override `unitree_mujoco/simulate_python/unitree_sdk2py_bridge.py`, which will simulate 500 Hz low level PD control.
    - Modified `unitree_mujoco/unitree_robots/g1/scence.xml` to remove obstacles. Specifically,comment out the lines in `<world body>` tag except
    ```
    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane" />
    ```
## Usage
For sim2sim deployment:
- Plugin a gamepad to your computer to further control the robot befor starting the simulator. Otherwise the simulator will report an error.
- Activate the virtual env and start the simulator using
```
python unitree_mujoco/simulate_python/unitree_mujoco.py
```
- Start another terminal,activate the virtual env and start the deploy code using
```
python deploy_code/deploy_real.py
```
- After starting the controller, press `start` to get into `position control mode`,the robot will not keep balance in this mode. After that press `A` to get into `balance standing mode`. Press key `9` on the keyboard to enforce/realse the robot from the virtual elatic band. Then you can press `B` to get into `motion mimic mode`. **Please noticing that key `9` is on the keyboard, other keys are on gamepad.**

you can change corresponding `policy_path`,`motion_file` and `init_pos` in `configs/g1_29.yaml` to mimic different trained motion. Now `shakehand` and `walk_AMASS` motions are avaliable. You can also try to play `walk_1127` and `walk_1201_edit_cont` which are some failure cases.

For sim2real deploy:
- Please follow [unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym) to setup the hardware.
- Modify the net interface and domain id in `deploy_real.py` to adapt to your robot. For simulation the net_interface is "lo" and domain id is "1",while in the real robot the net_interface should adjust based on the hardware and the domain id should set to default value "0".
```
net = cfg.get("net_interface", "lo")
ChannelFactoryInitialize(1, net)
```
- After connecting to the robot, start the deploy code as above.
- Use the unitree gamepad to control the robot with the same way in the simulator.