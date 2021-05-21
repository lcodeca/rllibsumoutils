#!/usr/bin/env python3

""" Example Trainer for RLLIB + SUMO Utlis

    Author: Lara CODECA

    This program and the accompanying materials are made available under the
    terms of the Eclipse Public License 2.0 which is available at
    http://www.eclipse.org/legal/epl-2.0.
"""

from copy import deepcopy
import logging
import pathlib
from pprint import pformat
import random
import sys
import traceback

import ray

from ray.rllib.agents.ppo import ppo
from ray.tune.logger import pretty_print

import marlenvironment

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger('ppotrain')

def _main():
    """ Training example """

    # Initialize RAY.
    ray.tune.registry.register_env('test_env', marlenvironment.env_creator)
    ray.init()

    # Algorithm.
    policy_class = ppo.PPOTFPolicy
    # https://github.com/ray-project/ray/blob/releases/0.8.3/rllib/agents/trainer.py#L41
    # https://github.com/ray-project/ray/blob/releases/0.8.3/rllib/agents/ppo/ppo.py#L15
    policy_conf = ppo.DEFAULT_CONFIG
    policy_conf['batch_mode'] = 'complete_episodes'
    policy_conf['log_level'] = 'WARN'
    policy_conf['min_iter_time_s'] = 5
    policy_conf['num_workers'] = 2
    policy_conf['rollout_fragment_length'] = 1
    policy_conf['seed'] = 42
    policy_conf['sgd_minibatch_size'] = 1
    policy_conf['simple_optimizer'] = True
    policy_conf['train_batch_size'] = 1

    # Load default Scenario configuration for the LEARNING ENVIRONMENT
    scenario_config = deepcopy(marlenvironment.DEFAULT_SCENARIO_CONFING)
    scenario_config['seed'] = 42
    scenario_config['log_level'] = 'INFO'
    scenario_config['sumo_config']['sumo_connector'] = 'libsumo'
    scenario_config['sumo_config']['sumo_cfg'] = '{}/scenario/sumo.cfg.xml'.format(
        pathlib.Path(__file__).parent.absolute())
    scenario_config['sumo_config']['sumo_params'] = ['--collision.action', 'warn']
    scenario_config['sumo_config']['trace_file'] = False
    scenario_config['sumo_config']['end_of_sim'] = 3600 # [s]
    scenario_config['sumo_config']['update_freq'] = 10 # number of traci.simulationStep()
                                                       # for each learning step.
    scenario_config['sumo_config']['log_level'] = 'INFO'
    logger.info('Scenario Configuration: \n %s', pformat(scenario_config))

    # Associate the agents with their configuration.
    agent_init = {
        'agent_0': deepcopy(marlenvironment.DEFAULT_AGENT_CONFING),
        'agent_1': deepcopy(marlenvironment.DEFAULT_AGENT_CONFING),
    }
    logger.info('Agents Configuration: \n %s', pformat(agent_init))

    ## MARL Environment Init
    env_config = {
        'agent_init': agent_init,
        'scenario_config': scenario_config,
    }
    marl_env = marlenvironment.SUMOTestMultiAgentEnv(env_config)

    # Config for the PPO trainer from the MARLEnv
    policies = {}
    for agent in marl_env.get_agents():
        agent_policy_params = {}
        policies[agent] = (policy_class,
                           marl_env.get_obs_space(agent),
                           marl_env.get_action_space(agent),
                           agent_policy_params)
    policy_conf['multiagent']['policies'] = policies
    policy_conf['multiagent']['policy_mapping_fn'] = lambda agent_id: agent_id
    policy_conf['multiagent']['policies_to_train'] = ['ppo_policy']
    policy_conf['env_config'] = env_config

    logger.info('PPO Configuration: \n %s', pformat(policy_conf))
    trainer = ppo.PPOTrainer(env='test_env',
                             config=policy_conf)

    # Single training iteration, just for testing.
    try:
        result = trainer.train()
        print('Results: \n {}'.format(pretty_print(result)))
    except Exception:
        EXC_TYPE, EXC_VALUE, EXC_TRACEBACK = sys.exc_info()
        traceback.print_exception(EXC_TYPE, EXC_VALUE, EXC_TRACEBACK, file=sys.stdout)
    finally:
        ray.shutdown()

if __name__ == '__main__':
    _main()
