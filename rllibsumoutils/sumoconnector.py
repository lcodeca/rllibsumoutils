""" RLLIB SUMO Utils - SUMO Connector

    Author: Lara CODECA

    This program and the accompanying materials are made available under the
    terms of the Eclipse Public License 2.0 which is available at
    http://www.eclipse.org/legal/epl-2.0.
"""

import logging
import os
from random import random
import sys
from datetime import datetime

# """ Import SUMO library """
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
    import sumolib
    # import traci
    import libsumo as traci
    import traci.constants as tc
else:
    raise Exception("Please declare environment variable 'SUMO_HOME'")

####################################################################################################

logging.basicConfig()
logger = logging.getLogger(__name__)

####################################################################################################

DEFAULT_CONFIG = {
    # SUMO configuration file. Required. String.
    'sumo_cfg': None,
    # Overides <output-prefix value='..'/>.
    # Required when using multiple environments at the same time. String.
    'sumo_output': '',
    # Additional parameter for the SUMO command line.
    # It cannot contain --output-prefix. List of strings.
    'sumo_params': None,
    # Enable TraCI trace file. Boolean.
    'trace_file': False,
    # SUMO Simulation ending time, in seconds. Float.
    'end_of_sim': None,
    # SUMO update frequency in number of traci.simulationStep() calls. Integer.
    'update_freq': 1,
    # SUMO tripinfo file as defined in the sumo configuration file in <tripinfo-output value='..'/>.
    # Required for gathering metrics only. String.
    'tripinfo_keyword': None,
    # SUMO tripinfo XML Schema file. Required for gathering metrics only. String.
    'tripinfo_xml_schema': None,
    # Logging legel. Should be one of DEBUG, INFO, WARN, or ERROR.
    'log_level': 'WARN',
    # Anything. User defined.
    'misc': None,
}

####################################################################################################

class SUMOConnector(object):
    """ Handler of a SUMO simulation. """

    def __init__(self, config):
        """
        Initialize SUMO and sets the beginning of the simulation.

        Param:
            config: Dict. See DEFAULT_CONFIG.
        """
        self._config = config

        # logging
        level = logging.getLevelName(config['log_level'])
        logger.setLevel(level)

        # TraCI Handler and SUMO simulation
        self._sumo_label = '{}'.format(self._get_unique_id())
        self._sumo_output_prefix = '{}{}'.format(config['sumo_output'], self._sumo_label)
        self._sumo_parameters = ['sumo', '-c', config['sumo_cfg']]
        if config['sumo_params'] is not None:
            self._sumo_parameters.extend(config['sumo_params'])
        self._sumo_parameters.extend(['--output-prefix', self._sumo_output_prefix])
        logger.debug('SUMO command line: %s', str(self._sumo_parameters))
        if config['trace_file']:
            traci.start(self._sumo_parameters,
                        traceFile='{}.tracefile.log'.format(self._sumo_output_prefix))
        else:
            traci.start(self._sumo_parameters)
        self.traci_handler = traci
        ## From now on, the call must always be to self.traci_handler

        self._is_ongoing = True
        self._start_time = self.traci_handler.simulation.getTime()
        self._sumo_steps = 0
        self._manually_stopped = False

        # Initialize simulation
        self._initialize_simulation()

        # Initialize metrics
        self._initialize_metrics()

    def __del__(self):
        try:
            self.end_simulation()
        except KeyError:
            logger.warning('Simulation %s already closed.', self._sumo_label)
        #TODO: delete files

    @staticmethod
    def _get_unique_id():
        now = datetime.utcnow()
        now_in_sec = (now - datetime(1970, 1, 1)).total_seconds()
        return round(random() * now_in_sec)

    ################################################################################################

    def _initialize_simulation(self):
        """ Specific simulation initialization. """
        raise NotImplementedError

    def _initialize_metrics(self):
        """ Specific metrics initialization """
        raise NotImplementedError

    def _default_step_action(self, agents):
        """ Specific code to be executed in every simulation step """
        raise NotImplementedError

    ################################################################################################

    def _stopping_condition(self, current_step_counter, until_end):
        """ Computes the stopping condition. """
        if self._manually_stopped:
            return True
        if self.traci_handler.simulation.getMinExpectedNumber() <= 0:
            # No entities left in the simulation.
            return True
        if self._config['end_of_sim'] is not None:
            if self.traci_handler.simulation.getTime() > self._config['end_of_sim']:
                # the simulatio reach the predefined (from parameters) end
                return True
        if current_step_counter == self._config['update_freq'] and not until_end:
            return True
        return False

    def step(self, until_end=False, agents=set()):
        """
        Runs a "learning" step and returns if the simulation has finished.
        This function in meant to be called by the RLLIB Environment.

        Params:
            until_end: Bool. If True, run the sumo simulation until the end.
            agents: Set(String). It passes the agents to the _default_step_action function.

        Return:
            Bool. True iff the simulation is still ongoing.
        """
        ## Execute SUMO steps until the learning needs to happen
        current_step_counter = 0
        logger.debug('============================================================================')
        while not self._stopping_condition(current_step_counter, until_end):
            logger.debug('[%s] Current step counter: %d, Update frequency: %d',
                         str(until_end), current_step_counter, self._config['update_freq'])
            self.traci_handler.simulationStep()
            self._sumo_steps += 1
            current_step_counter += 1
            self._default_step_action(agents)
        logger.debug('============================================================================')

        # If the simulation has finished
        if self.is_ongoing_sim():
            return True
        logger.debug('The SUMO simulation is done.')
        return False

    def fast_forward(self, time):
        """
        Move the simulation forward (without doing anything else) until the given time.
        Param:
            time: Float, simulation time in seconds.
        """
        logger.debug('Fast-forward from time %.2f', self.traci_handler.simulation.getTime())
        self.traci_handler.simulationStep(float(time))
        logger.debug('Fast-forward to time %.2f', self.traci_handler.simulation.getTime())

    ################################################################################################

    def get_sumo_steps(self):
        """ Returns the total number of traci.simulationStep() calls."""
        return self._sumo_steps

    def is_ongoing_sim(self):
        """ Return True iff the SUMO simulation is still ongoing. """
        if self._manually_stopped:
            return False
        if self.traci_handler.simulation.getMinExpectedNumber() <= 0:
            # No entities left in the simulation.
            return False
        if self._config['end_of_sim'] is not None:
            if self.traci_handler.simulation.getTime() > self._config['end_of_sim']:
                # the simulatio reach the predefined (from parameters) end
                return False
        return True

    def get_current_time(self):
        """ Returns the current simulation time, or None if the simulation is not ongoing. """
        if self.is_ongoing_sim():
            return self.traci_handler.simulation.getTime()
        return None

    def end_simulation(self):
        """ Forces the simulation to stop. """
        if self.is_ongoing_sim():
            logger.info('Closing TraCI %s', self._sumo_label)
            self._manually_stopped = True
            self.traci_handler.close()
        else:
            logger.warning('TraCI %s is already closed.', self._sumo_label)

    ################################################################################################
