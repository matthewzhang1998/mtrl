# -----------------------------------------------------------------------------
#   @author:
#       Tingwu Wang
# -----------------------------------------------------------------------------
import numpy as np
import tensorflow as tf
import os

from util import whitening_util, init_path, tf_util


class BasePolicy(object):
    '''
        @brief:
            In this object class, we define the network structure, the restore
            function and save function.
            It will only be called in the agent/agent.py
    '''

    def __init__(self, params, session, name_scope,
                 observation_size, action_size, path=None):
        self.params = params

        self._session = session
        self._name_scope = name_scope

        self.path = path if path is not None else self.params.output_dir

        self._observation_size = observation_size
        self._action_size = action_size

        self._task_name = params.task
        self._network_shape = params.policy_network_shape

        self._iters_so_far = 0

        self._npr = np.random.RandomState(params.seed)

        self._whitening_operator = {}
        self._whitening_variable = []
        self._base_dir = init_path.get_abs_base_dir()

    def build_network(self):
        raise NotImplementedError

    def build_loss(self):
        raise NotImplementedError

    def _build_ph(self):

        # initialize the running mean and std (whitening)
        whitening_util.add_whitening_operator(
            self._whitening_operator, self._whitening_variable,
            'state', self._observation_size
        )

        # initialize the input placeholder
        self._input_ph = {
            'start_state': tf.placeholder(
                tf.float32, [None, self._observation_size], name='start_state'
            )
        }

    def get_input_placeholder(self):
        return self._input_ph

    def get_weights(self):
        return None

    def set_weights(self, weights_dict):
        pass

    def _set_var_list(self):
        # collect the tf variable and the trainable tf variable
        self._trainable_var_list = [var for var in tf.trainable_variables()
                                    if self._name_scope in var.name]

        self._all_var_list = [var for var in tf.global_variables()
                              if self._name_scope in var.name]

        # the weights that actually matter
        self._network_var_list = \
            self._trainable_var_list + self._whitening_variable

        self._set_network_weights = tf_util.set_network_weights(
            self._session, self._network_var_list, self._name_scope
        )

        self._get_network_weights = tf_util.get_network_weights(
            self._session, self._network_var_list, self._name_scope
        )

    def load_checkpoint(self, ckpt_path):
        pass

    def build_writer(self):
        pass

    def save_checkpoint(self, ckpt_path):
        pass

    def get_whitening_operator(self):
        return self._whitening_operator

    def _set_whitening_var(self, whitening_stats):
        whitening_util.set_whitening_var(
            self._session, self._whitening_operator, whitening_stats, ['state']
        )

    def train(self, data_dict, replay_buffer, training_info={}):
        raise NotImplementedError

    def eval(self, data_dict):
        raise NotImplementedError

    def act(self, data_dict):
        raise NotImplementedError