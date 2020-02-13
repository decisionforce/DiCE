import numpy as np
from ray.rllib.models.tf.misc import normc_initializer, get_activation_fn
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils import try_import_tf

tf = try_import_tf()


class ActorDoubleCriticNetwork(TFModelV2):
    """Generic fully connected network implemented in ModelV2 API."""

    def __init__(
            self, obs_space, action_space, num_outputs, model_config, name
    ):
        super(ActorDoubleCriticNetwork, self).__init__(
            obs_space, action_space, num_outputs, model_config, name
        )

        activation = get_activation_fn(model_config.get("fcnet_activation"))
        hiddens = model_config.get("fcnet_hiddens")
        no_final_linear = model_config.get("no_final_linear")
        vf_share_layers = model_config.get("vf_share_layers")
        self.use_diversity_value_network = model_config['custom_options'][
            "use_diversity_value_network"]

        # we are using obs_flat, so take the flattened shape as input
        inputs = tf.keras.layers.Input(
            shape=(np.product(obs_space.shape),), name="observations"
        )
        last_layer = inputs
        i = 1

        if no_final_linear:
            # the last layer is adjusted to be of size num_outputs
            for size in hiddens[:-1]:
                last_layer = tf.keras.layers.Dense(
                    size,
                    name="fc_{}".format(i),
                    activation=activation,
                    kernel_initializer=normc_initializer(1.0)
                )(last_layer)
                i += 1
            layer_out = tf.keras.layers.Dense(
                num_outputs,
                name="fc_out",
                activation=activation,
                kernel_initializer=normc_initializer(1.0)
            )(last_layer)
        else:
            # the last layer is a linear to size num_outputs
            for size in hiddens:
                last_layer = tf.keras.layers.Dense(
                    size,
                    name="fc_{}".format(i),
                    activation=activation,
                    kernel_initializer=normc_initializer(1.0)
                )(last_layer)
                i += 1
            layer_out = tf.keras.layers.Dense(
                num_outputs,
                name="fc_out",
                activation=None,
                kernel_initializer=normc_initializer(0.01)
            )(last_layer)

        assert not vf_share_layers
        last_layer = inputs
        i = 1
        for size in hiddens:
            last_layer = tf.keras.layers.Dense(
                size,
                name="fc_value_{}".format(i),
                activation=activation,
                kernel_initializer=normc_initializer(1.0)
            )(last_layer)
            i += 1
        value_out = tf.keras.layers.Dense(
            1,
            name="value_out",
            activation=None,
            kernel_initializer=normc_initializer(0.01)
        )(last_layer)

        if self.use_diversity_value_network:
            # build the value network for novel
            last_layer = inputs
            i = 1
            for size in hiddens:
                last_layer = tf.keras.layers.Dense(
                    size,
                    name="fc_value_novel_{}".format(i),
                    activation=activation,
                    kernel_initializer=normc_initializer(1.0)
                )(last_layer)
                i += 1

            value_out_novel = tf.keras.layers.Dense(
                1,
                name="value_out_novel",
                activation=None,
                kernel_initializer=normc_initializer(0.01)
            )(last_layer)

            self.base_model = tf.keras.Model(
                inputs, [layer_out, value_out, value_out_novel]
            )
        else:
            self.base_model = tf.keras.Model(inputs, [layer_out, value_out])
        self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):
        if self.use_diversity_value_network:
            model_out, self._value_out, self._value_out_novel = \
                self.base_model(input_dict["obs_flat"])
        else:
            model_out, self._value_out = \
                self.base_model(input_dict["obs_flat"])
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])

    def diversity_value_function(self):
        assert self.use_diversity_value_network
        return tf.reshape(self._value_out_novel, [-1])
