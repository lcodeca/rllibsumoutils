""" RLLIB SUMO Utils - SUMO Connector

    Author: Lara CODECA

    This program and the accompanying materials are made available under the
    terms of the Eclipse Public License 2.0 which is available at
    http://www.eclipse.org/legal/epl-2.0.
"""

import collections
import logging
import os
from pprint import pformat
import sys

# """ Import SUMO library """
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
    import traci
    import traci.constants as tc
else:
    raise Exception("Please declare environment variable 'SUMO_HOME'")

####################################################################################################

LOGGER = logging.getLogger(__name__)

####################################################################################################

class SUMOConnector(object):
    """ Handler of a SUMO simulation. """
    def __init__(self, config):
        """
        Initialize SUMO and sets the beginning of the simulation.

        Param:
            config: Dict. 
                {
                    'sumo_cfg': SUMO configuration file. String.
                    'sumo_params': Additional parameter for the SUMO command line. 
                                   List of strings.
                    'end_of_sim': Simulation ending time, in seconds. Float.
                    'update_freq': SUMO update frequency in number of traci.simulationStep() calls. 
                                   Integer. 
                    'tripinfo_xml_file': SUMO tripinfo file. Required for gathering metrics only. 
                                         String.
                    'tripinfo_xml_schema': SUMO tripinfo XML Schema file. 
                                           Required for gathering metrics only. String.
                    'misc': Anything. User defined.
                }
        """
        self._config = config
        sumo_parameters = ['sumo', '-c', config['sumo_cfg']]
        if config['sumo_params']:
            sumo_parameters.extend(config['sumo_params'])
        traci.start(sumo_parameters)

        self.traci_handler = traci
        self._is_ongoing = True
        self._start_time = traci.simulation.getTime()
        self._sumo_steps = 0
        self._manually_stopped = False

        # Initialize simulation
        self._initialize_simulation()

        # Initialize metrics
        self._initialize_metrics()

    def __del__(self):
        self.end_simulation()

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
        LOGGER.debug('=============================================================================')
        while not self._stopping_condition(current_step_counter, until_end):
            LOGGER.debug('[%s] Current step counter: %d, Update frequency: %d', 
                         str(until_end), current_step_counter, self._config['update_freq'])
            self.traci_handler.simulationStep()
            self._sumo_steps += 1
            current_step_counter += 1
            self._default_step_action(agents)
        LOGGER.debug('=============================================================================')

        ## Set if simulation has ended
        self._is_ongoing = (not self._manually_stopped and
                            self.traci_handler.simulation.getMinExpectedNumber() > 0 and
                            self.traci_handler.simulation.getTime() <= self._config['end_of_sim'])
        

        if not self.is_ongoing_sim():
            LOGGER.debug('The SUMO simulation is done.')

        # If the simulation has finished
        return self.is_ongoing_sim()

    def fast_forward(self, time):
        """ 
        Move the simulation forward (without doing anything else) until the given time. 
        Param:
            time: Float, simulation time in seconds.
        """
        LOGGER.debug('Fast-forward from time %.2f', self.traci_handler.simulation.getTime())
        self.traci_handler.simulationStep(float(time))
        LOGGER.debug('Fast-forward to time %.2f', self.traci_handler.simulation.getTime())

    ################################################################################################

    def get_sumo_steps(self):
        """ Returns the total number of traci.simulationStep() calls."""
        return self._sumo_steps

    def is_ongoing_sim(self):
        """ Return True iff the SUMO simulation is still ongoing. """
        return self._is_ongoing and not self._manually_stopped

    def get_current_time(self):
        """ Returns the current simulation time, or None if the simulation is not ongoing. """
        if self.is_ongoing_sim():
            return self.traci_handler.simulation.getTime()
        return None

    def end_simulation(self):
        """ Forces the simulation to stop. """
        self._manually_stopped = True
        try:
            self.traci_handler.close()
        except KeyError:
            # The 'default' connection was already closed.
            pass

    ################################################################################################