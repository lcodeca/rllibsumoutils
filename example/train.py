#!/usr/bin/env python3

""" Example Trainer for RLLIB + SUMO Utlis

    Author: Lara CODECA

    This program and the accompanying materials are made available under the
    terms of the Eclipse Public License 2.0 which is available at
    http://www.eclipse.org/legal/epl-2.0.
"""

from copy import deepcopy
import pathlib
import random
import sys
import traceback

import ray

from ray.rllib.agents.ppo import ppo
from ray.tune.logger import pretty_print

import marlenvironment

def _main():
    """ Training example """

    # Algorithm.
    policy_class = ppo.PPOTFPolicy
    # https://github.com/ray-project/ray/blob/releases/0.8.3/rllib/agents/trainer.py#L41
    # https://github.com/ray-project/ray/blob/releases/0.8.3/rllib/agents/ppo/ppo.py#L15
    policy_conf = ppo.DEFAULT_CONFIG
    policy_conf['log_level'] = 'INFO'
    policy_conf['num_workers'] = 2
    policy_conf['train_batch_size'] = 1
    policy_conf['sgd_minibatch_size'] = 1
    policy_conf['rollout_fragment_length'] = 1
    policy_conf['simple_optimizer'] = True
    policy_conf['batch_mode'] = 'complete_episodes'
    policy_params = {}

    # Load default Scenario configuration for the LEARNING ENVIRONMENT
    scenario_config = {
        "agent-rnd-order": True,
        "sumo-cfg-file": "{}/scenario/sumo.cfg.xml".format(
            pathlib.Path(__file__).parent.absolute()),
        "sumo-params": ['--collision.action', 'warn'],
        "seed": 42,
        "misc": {
            "algo-update-freq": 10,     # number of traci.simulationStep() each learning step.
            "end-of-sim": 3600,         # [s]
            "max-distance": 5000,       # [m]
        }
    }

    # Initialize RAY.
    ray.tune.registry.register_env('test_env', marlenvironment.env_creator)
    ray.init(memory=52428800, object_store_memory=78643200) ## minimum values

    # Associate the agents with their configuration.
    agent_init = {
        "agent_0": {
            "origin": "road",
            "destination": "road",
            "start": 0,
            "actions": {          # increase/decrease the speed of:
                "acc": 1.0,     # [m/s]
                "none": 0.0,    # [m/s]
                "dec": -1.0,    # [m/s]
            },
            "max-speed": 130,   # km/h
            "seed": 75834444,
            "init": [0, 0],
        },
        "agent_1": {
            "origin": "road",
            "destination": "road",
            "start": 0,
            "actions": {
                "acc": 1.0,
                "none": 0.0,
                "dec": -1.0,
            },
            "max-speed": 130,
            "seed": 44447583,
            "init": [0, 0],
        }
    }

    ## MARL Environment Init
    env_config = {
        'agent_init': agent_init,
        'scenario_config': scenario_config,
    }
    marl_env = marlenvironment.SUMOTestMultiAgentEnv(env_config)

    # Config for the PPO trainer from the MARLEnv
    policies = {}
    for agent in marl_env.get_agents():
        agent_policy_params = deepcopy(policy_params)
        from_val, to_val = agent_init[agent]['init']
        agent_policy_params['init'] = lambda: random.randint(from_val, to_val)
        agent_policy_params['actions'] = marl_env.get_set_of_actions(agent)
        agent_policy_params['seed'] = agent_init[agent]['seed']
        policies[agent] = (policy_class,
                           marl_env.get_obs_space(agent),
                           marl_env.get_action_space(agent),
                           agent_policy_params)
    policy_conf['multiagent'] = {
        'policies': policies,
        'policy_mapping_fn': lambda agent_id: agent_id,
        'policies_to_train': ['ppo_policy'],
    }
    policy_conf['env_config'] = env_config
    trainer = ppo.PPOTrainer(env='test_env',
                             config=policy_conf)

    # Single training iteration, just for testing.
    result = trainer.train()
    print(pretty_print(result))

if __name__ == '__main__':
    try:
        _main()
    except Exception:
        EXC_TYPE, EXC_VALUE, EXC_TRACEBACK = sys.exc_info()
        traceback.print_exception(EXC_TYPE, EXC_VALUE, EXC_TRACEBACK, file=sys.stdout)
    finally:
        ray.shutdown()
