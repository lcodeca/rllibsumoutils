# RLLIB+SUMO Utils

Python3 library able to connect the RLLIB framework with the SUMO simulator.

Contact: Lara CODECA [lara.codeca@gmail.com]

This program and the accompanying materials are made available under the terms of
the Eclipse Public License 2.0 which is available at <http://www.eclipse.org/legal/epl-2.0>.

## Minimum Requirements

* [SUMO 1.5](https://github.com/eclipse/sumo/tree/v1_5_0).
* RLLIB [RAY 0.8.2](https://github.com/ray-project/ray/tree/ray-0.8.2)

### Tested with

* Eclipse SUMO Version 1.8.0
* RLlib Version 1.3.0

## Installation

* Install: `pip3 install .` from the root directory, or `python3 setup.py install`
* Development install: `pip3 install -e .` or `python3 setup.py develop`

### Note: rtree library

To use `rtree`, `libspatialindex-dev` is required to be installed.

[In Ubuntu/Debian] `sudo apt install libspatialindex-dev python-rtree` and then `pip install rtree`

## Example

* Given the ~under development~ status of the project, some examples are provided.
  * `example/scenario`
    * Simple SUMO scenario.
  * `example/marlenvironment.py`
    * Example of MARL environment implemented using RLLIB (SUMOTestMultiAgentEnv)
  * `example/train.py`
    * Example of PPO trainer using SUMOTestMultiAgentEnv
    * How to: `python3 train.py`

## Docker Environment

See [RLLIB SUMO Docker](https://github.com/lcodeca/rllibsumodocker) for details on my development and learning environment.
