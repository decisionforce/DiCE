from gym.spaces import Discrete
from ray.rllib.models.tf.tf_action_dist import SquashedGaussian
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.tf_ops import minimize_and_clip

tf = try_import_tf()


def _flatten(tensor):
    flat = tf.reshape(tensor, shape=[-1])
    return flat, tensor.shape, flat.shape


def get_dist_class(config, action_space):
    assert config["_use_beta_distribution"] is False, \
        "Beta-distr. not supported for tf!"
    if isinstance(action_space, Discrete):
        # action_dist_class = Categorical
        raise ValueError()
    else:
        action_dist_class = SquashedGaussian
        # (SquashedGaussian if config["normalize_actions"] else DiagGaussian)
    return action_dist_class


def dice_sac_loss(policy, model, _, train_batch):
    # I think diversity should be compute here.
    model_out_t, _ = model(
        {
            "obs": train_batch[SampleBatch.CUR_OBS],
            "is_training": policy._get_is_training_placeholder(),
        }, [], None
    )

    diversity_reward = train_batch["diversity_rewards"]

    model_out_tp1, _ = model(
        {
            "obs": train_batch[SampleBatch.NEXT_OBS],
            "is_training": policy._get_is_training_placeholder(),
        }, [], None
    )

    target_model_out_tp1, _ = policy.target_model(
        {
            "obs": train_batch[SampleBatch.NEXT_OBS],
            "is_training": policy._get_is_training_placeholder(),
        }, [], None
    )

    # Discrete case.
    if model.discrete:
        raise NotImplementedError()
    # Continuous actions case.
    else:
        # Sample simgle actions from distribution.
        action_dist_class = get_dist_class(policy.config, policy.action_space)
        action_dist_t = action_dist_class(
            model.get_policy_output(model_out_t), policy.model
        )
        policy_t = action_dist_t.sample()
        log_pis_t = tf.expand_dims(action_dist_t.logp(policy_t), -1)
        action_dist_tp1 = action_dist_class(
            model.get_policy_output(model_out_tp1), policy.model
        )
        policy_tp1 = action_dist_tp1.sample()
        log_pis_tp1 = tf.expand_dims(action_dist_tp1.logp(policy_tp1), -1)

        # Q-values for the actually selected actions.
        q_t = model.get_q_values(model_out_t, train_batch[SampleBatch.ACTIONS])
        diversity_q_t = model.get_diversity_q_values(
            model_out_t, train_batch[SampleBatch.ACTIONS]
        )
        if policy.config["twin_q"]:
            twin_q_t = model.get_twin_q_values(
                model_out_t, train_batch[SampleBatch.ACTIONS]
            )
        if policy.config["diversity_twin_q"]:
            diversity_twin_q_t = model.get_diversity_twin_q_values(
                model_out_t, train_batch[SampleBatch.ACTIONS]
            )

        # Q-values for current policy in given current state.
        q_t_det_policy = model.get_q_values(model_out_t, policy_t)
        diversity_q_t_det_policy = model.get_diversity_q_values(
            model_out_t, policy_t
        )
        if policy.config["twin_q"]:
            twin_q_t_det_policy = model.get_twin_q_values(
                model_out_t, policy_t
            )
            q_t_det_policy = tf.reduce_min(
                (q_t_det_policy, twin_q_t_det_policy), axis=0
            )
        if policy.config["diversity_twin_q"]:
            diversity_twin_q_t_det_policy = model.get_diversity_twin_q_values(
                model_out_t, policy_t
            )
            diversity_q_t_det_policy = tf.reduce_min(
                (diversity_q_t_det_policy, diversity_twin_q_t_det_policy),
                axis=0
            )

        # target q network evaluation
        q_tp1 = policy.target_model.get_q_values(
            target_model_out_tp1, policy_tp1
        )
        diversity_q_tp1 = policy.target_model.get_diversity_q_values(
            target_model_out_tp1, policy_tp1
        )
        if policy.config["twin_q"]:
            twin_q_tp1 = policy.target_model.get_twin_q_values(
                target_model_out_tp1, policy_tp1
            )
            # Take min over both twin-NNs.
            q_tp1 = tf.reduce_min((q_tp1, twin_q_tp1), axis=0)
        if policy.config["diversity_twin_q"]:
            diversity_twin_q_tp1 = \
                policy.target_model.get_diversity_twin_q_values(
                    target_model_out_tp1, policy_tp1)
            # Take min over both twin-NNs.
            diversity_q_tp1 = tf.reduce_min(
                (diversity_q_tp1, diversity_twin_q_tp1), axis=0
            )

        q_t_selected = tf.squeeze(q_t, axis=len(q_t.shape) - 1)
        diversity_q_t_selected = tf.squeeze(
            diversity_q_t, axis=len(diversity_q_t.shape) - 1
        )
        if policy.config["twin_q"]:
            twin_q_t_selected = tf.squeeze(twin_q_t, axis=len(q_t.shape) - 1)
        if policy.config["diversity_twin_q"]:
            diversity_twin_q_t_selected = tf.squeeze(
                diversity_twin_q_t, axis=len(diversity_q_t.shape) - 1
            )
        q_tp1 -= model.alpha * log_pis_tp1
        diversity_q_tp1 -= model.alpha * log_pis_tp1  # ???

        q_tp1_best = tf.squeeze(input=q_tp1, axis=len(q_tp1.shape) - 1)
        q_tp1_best_masked = (
                                    1.0 - tf.cast(
                                train_batch[SampleBatch.DONES], tf.float32)
                            ) * q_tp1_best

        diversity_q_tp1_best = tf.squeeze(
            input=diversity_q_tp1, axis=len(diversity_q_tp1.shape) - 1
        )
        diversity_q_tp1_best_masked = (
                                              1.0 - tf.cast(
                                          train_batch[SampleBatch.DONES],
                                          tf.float32)
                                      ) * diversity_q_tp1_best

    assert policy.config["n_step"] == 1, "TODO(hartikainen) n_step > 1"

    # compute RHS of bellman equation
    q_t_selected_target = tf.stop_gradient(
        train_batch[SampleBatch.REWARDS] +
        policy.config["gamma"] ** policy.config["n_step"] * q_tp1_best_masked
    )

    diversity_q_t_selected_target = tf.stop_gradient(
        # train_batch["diversity_rewards"] +
        diversity_reward + policy.config["gamma"] ** policy.config["n_step"] *
        diversity_q_tp1_best_masked
    )

    # Compute the TD-error (potentially clipped).
    base_td_error = tf.abs(q_t_selected - q_t_selected_target)
    diversity_base_td_error = tf.abs(
        diversity_q_t_selected - diversity_q_t_selected_target
    )
    if policy.config["twin_q"]:
        twin_td_error = tf.abs(twin_q_t_selected - q_t_selected_target)
        td_error = 0.5 * (base_td_error + twin_td_error)
    if policy.config["diversity_twin_q"]:
        diversity_twin_td_error = tf.abs(
            diversity_twin_q_t_selected - diversity_q_t_selected_target
        )
        diversity_td_error = 0.5 * (
                diversity_base_td_error + diversity_twin_td_error
        )
    else:
        td_error = base_td_error
        diversity_td_error = diversity_base_td_error

    diversity_critic_loss = [
        tf.losses.mean_squared_error(
            labels=diversity_q_t_selected_target,
            predictions=diversity_q_t_selected,
            weights=0.5
        )
    ]

    critic_loss = [
        tf.losses.mean_squared_error(
            labels=q_t_selected_target, predictions=q_t_selected, weights=0.5
        )
    ]
    if policy.config["twin_q"]:
        critic_loss.append(
            tf.losses.mean_squared_error(
                labels=q_t_selected_target,
                predictions=twin_q_t_selected,
                weights=0.5
            )
        )
    if policy.config["diversity_twin_q"]:
        diversity_critic_loss.append(
            tf.losses.mean_squared_error(
                labels=diversity_q_t_selected_target,
                predictions=diversity_twin_q_t_selected,
                weights=0.5
            )
        )

    # Alpha- and actor losses.
    # Note: In the papers, alpha is used directly, here we take the log.
    # Discrete case: Multiply the action probs as weights with the original
    # loss terms (no expectations needed).
    if model.discrete:
        raise NotImplementedError()
    else:
        alpha_loss = -tf.reduce_mean(
            model.log_alpha *
            tf.stop_gradient(log_pis_t + model.target_entropy)
        )
        actor_loss = tf.reduce_mean(model.alpha * log_pis_t - q_t_det_policy)
        diversity_actor_loss = tf.reduce_mean(
            model.alpha * log_pis_t - diversity_q_t_det_policy
        )

    # save for stats function
    policy.policy_t = policy_t
    policy.q_t = q_t
    policy.td_error = td_error
    policy.actor_loss = actor_loss
    policy.critic_loss = critic_loss
    policy.alpha_loss = alpha_loss
    policy.alpha_value = model.alpha
    policy.target_entropy = model.target_entropy

    policy.entropy = tf.zeros_like(actor_loss) \
        if isinstance(action_dist_t, SquashedGaussian) else \
        action_dist_t.entropy()

    policy.diversity_critic_loss = diversity_critic_loss
    policy.diversity_actor_loss = diversity_actor_loss
    policy.diversity_td_error = diversity_td_error
    policy.log_pis_t = log_pis_t

    # add what we need here
    policy.diversity_reward_mean = tf.reduce_mean(
        # train_batch["diversity_rewards"]
        diversity_reward
    )

    # in a custom apply op we handle the losses separately, but return them
    # combined in one loss for now
    return actor_loss + tf.add_n(critic_loss) + alpha_loss + \
           tf.add_n(diversity_critic_loss) + diversity_actor_loss


def dice_sac_gradient(policy, optimizer, loss):
    if policy.config["grad_clip"]:

        actor_grads_and_vars = minimize_and_clip(
            policy._actor_optimizer,
            policy.actor_loss,
            var_list=policy.model.policy_variables(),
            clip_val=policy.config["grad_clip"]
        )

        diversity_actor_grads_and_vars = minimize_and_clip(
            policy._actor_optimizer,
            policy.diversity_actor_loss,
            var_list=policy.model.policy_variables(),
            clip_val=policy.config["grad_clip"]
        )

        if policy.config["twin_q"]:
            q_variables = policy.model.q_variables()
            half_cutoff = len(q_variables) // 2
            critic_grads_and_vars = []
            critic_grads_and_vars += minimize_and_clip(
                policy._critic_optimizer[0],
                policy.critic_loss[0],
                var_list=q_variables[:half_cutoff],
                clip_val=policy.config["grad_clip"]
            )
            critic_grads_and_vars += minimize_and_clip(
                policy._critic_optimizer[1],
                policy.critic_loss[1],
                var_list=q_variables[half_cutoff:],
                clip_val=policy.config["grad_clip"]
            )
        else:
            critic_grads_and_vars = minimize_and_clip(
                policy._critic_optimizer[0],
                policy.critic_loss[0],
                var_list=policy.model.q_variables(),
                clip_val=policy.config["grad_clip"]
            )

        if policy.config["diversity_twin_q"]:
            # Diversity
            diversity_q_variables = policy.model.diversity_q_variables()
            diversity_half_cutoff = len(diversity_q_variables) // 2
            diversity_critic_grads_and_vars = []
            diversity_critic_grads_and_vars += minimize_and_clip(
                policy._diversity_critic_optimizer[0],
                policy.diversity_critic_loss[0],
                var_list=diversity_q_variables[:diversity_half_cutoff],
                clip_val=policy.config["grad_clip"]
            )
            diversity_critic_grads_and_vars += minimize_and_clip(
                policy._diversity_critic_optimizer[1],
                policy.diversity_critic_loss[1],
                var_list=diversity_q_variables[diversity_half_cutoff:],
                clip_val=policy.config["grad_clip"]
            )
        else:
            diversity_critic_grads_and_vars = minimize_and_clip(
                policy._diversity_critic_optimizer[0],
                policy.diversity_critic_loss[0],
                var_list=policy.model.diversity_q_variables(),
                clip_val=policy.config["grad_clip"]
            )

        alpha_grads_and_vars = minimize_and_clip(
            policy._alpha_optimizer,
            policy.alpha_loss,
            var_list=[policy.model.log_alpha],
            clip_val=policy.config["grad_clip"]
        )
    else:
        actor_grads_and_vars = policy._actor_optimizer.compute_gradients(
            policy.actor_loss, var_list=policy.model.policy_variables()
        )

        diversity_actor_grads_and_vars = \
            policy._actor_optimizer.compute_gradients(
                policy.diversity_actor_loss,
                var_list=policy.model.policy_variables())

        if policy.config["twin_q"]:
            q_variables = policy.model.q_variables()
            half_cutoff = len(q_variables) // 2
            base_q_optimizer, twin_q_optimizer = policy._critic_optimizer
            critic_grads_and_vars = base_q_optimizer.compute_gradients(
                policy.critic_loss[0], var_list=q_variables[:half_cutoff]
            ) + twin_q_optimizer.compute_gradients(
                policy.critic_loss[1], var_list=q_variables[half_cutoff:]
            )
        else:
            critic_grads_and_vars = policy._critic_optimizer[
                0].compute_gradients(
                policy.critic_loss[0], var_list=policy.model.q_variables()
            )

        if policy.config["diversity_twin_q"]:
            # Diversity
            diversity_q_variables = policy.model.diversity_q_variables()
            diversity_half_cutoff = len(diversity_q_variables) // 2
            diversity_base_q_optimizer, diversity_twin_q_optimizer = \
                policy._diversity_critic_optimizer
            diversity_critic_grads_and_vars = \
                diversity_base_q_optimizer.compute_gradients(
                    policy.diversity_critic_loss[0],
                    var_list=diversity_q_variables[:diversity_half_cutoff]
                ) + diversity_twin_q_optimizer.compute_gradients(
                    policy.diversity_critic_loss[1],
                    var_list=diversity_q_variables[diversity_half_cutoff:]
                )
        else:
            diversity_critic_grads_and_vars = policy._critic_optimizer[
                0].compute_gradients(
                policy.diversity_critic_loss[0],
                var_list=policy.model.diversity_q_variables()
            )

        alpha_grads_and_vars = policy._alpha_optimizer.compute_gradients(
            policy.alpha_loss, var_list=[policy.model.log_alpha]
        )

    policy_grad = actor_grads_and_vars
    diversity_grad = diversity_actor_grads_and_vars

    return_gradients = {}
    policy_grad_flatten = []
    policy_grad_info = []
    diversity_grad_flatten = []
    diversity_grad_info = []

    # First, flatten task gradient and diversity gradient into two vector.
    for (pg, var), (ng, var2) in zip(policy_grad, diversity_grad):
        assert var == var2
        if pg is None:
            return_gradients[var] = (ng, var2)
            continue
        if ng is None:
            return_gradients[var] = (pg, var)
            continue

        pg_flat, pg_shape, pg_flat_shape = _flatten(pg)
        policy_grad_flatten.append(pg_flat)
        policy_grad_info.append((pg_flat_shape, pg_shape, var))

        ng_flat, ng_shape, ng_flat_shape = _flatten(ng)
        diversity_grad_flatten.append(ng_flat)
        diversity_grad_info.append((ng_flat_shape, ng_shape))

    policy_grad_flatten = tf.concat(policy_grad_flatten, 0)
    diversity_grad_flatten = tf.concat(diversity_grad_flatten, 0)

    # Second, compute the norm of two gradient.
    policy_grad_norm = tf.linalg.l2_normalize(policy_grad_flatten)
    diversity_grad_norm = tf.linalg.l2_normalize(diversity_grad_flatten)

    # Third, compute the bisector.
    final_grad = tf.linalg.l2_normalize(policy_grad_norm + diversity_grad_norm)

    # Fourth, compute the length of the final gradient.
    pg_length = tf.norm(tf.multiply(policy_grad_flatten, final_grad))
    ng_length = tf.norm(tf.multiply(diversity_grad_flatten, final_grad))
    ng_length = tf.minimum(pg_length, ng_length)
    tg_lenth = (pg_length + ng_length) / 2

    final_grad = final_grad * tg_lenth

    # add some stats.
    policy.gradient_cosine_similarity = tf.reduce_sum(
        tf.multiply(policy_grad_norm, diversity_grad_norm)
    )
    policy.policy_grad_norm = tf.norm(policy_grad_flatten)
    policy.diversity_grad_norm = tf.norm(diversity_grad_flatten)

    # Fifth, split the flatten vector into the original form as the final
    # gradients.
    count = 0
    for idx, (flat_shape, org_shape, var) in enumerate(policy_grad_info):
        assert flat_shape is not None
        size = flat_shape.as_list()[0]
        grad = final_grad[count:count + size]
        return_gradients[var] = (tf.reshape(grad, org_shape), var)
        count += size

    if policy.config["grad_clip"] is not None:
        ret_grads = [return_gradients[var][0] for _, var in policy_grad]
        clipped_grads, _ = tf.clip_by_global_norm(
            ret_grads, policy.config["grad_clip"]
        )
        actor_grads_and_vars_fused = [
            (g, return_gradients[var][1])
            for g, (_, var) in zip(clipped_grads, policy_grad)
        ]
    else:
        actor_grads_and_vars_fused = [
            return_gradients[var] for _, var in policy_grad
        ]

    # save these for later use in build_apply_op
    policy._actor_grads_and_vars = [
        (g, v) for (g, v) in actor_grads_and_vars_fused if g is not None
    ]
    policy._critic_grads_and_vars = [
        (g, v) for (g, v) in critic_grads_and_vars if g is not None
    ]
    policy._diversity_critic_grads_and_vars = [
        (g, v) for (g, v) in diversity_critic_grads_and_vars if g is not None
    ]
    policy._alpha_grads_and_vars = [
        (g, v) for (g, v) in alpha_grads_and_vars if g is not None
    ]

    assert policy._actor_grads_and_vars
    assert policy._critic_grads_and_vars
    assert policy._diversity_critic_grads_and_vars
    assert policy._alpha_grads_and_vars

    grads_and_vars = (
            policy._actor_grads_and_vars + policy._critic_grads_and_vars +
            policy._diversity_critic_grads_and_vars +
            policy._alpha_grads_and_vars
    )

    return grads_and_vars


def apply_gradients(policy, optimizer, grads_and_vars):
    actor_apply_ops = policy._actor_optimizer.apply_gradients(
        policy._actor_grads_and_vars
    )

    cgrads = policy._critic_grads_and_vars
    half_cutoff = len(cgrads) // 2
    if policy.config["twin_q"]:
        critic_apply_ops = [
            policy._critic_optimizer[0].apply_gradients(cgrads[:half_cutoff]),
            policy._critic_optimizer[1].apply_gradients(cgrads[half_cutoff:])
        ]
    else:
        critic_apply_ops = [
            policy._critic_optimizer[0].apply_gradients(cgrads)
        ]

    diverity_cgrads = policy._diversity_critic_grads_and_vars
    diversity_half_cutoff = len(diverity_cgrads) // 2
    if policy.config["diversity_twin_q"]:
        diversity_critic_apply_ops = [
            policy._diversity_critic_optimizer[0].apply_gradients(
                diverity_cgrads[:diversity_half_cutoff]
            ), policy._diversity_critic_optimizer[1].apply_gradients(
                diverity_cgrads[diversity_half_cutoff:]
            )
        ]
    else:
        diversity_critic_apply_ops = [
            policy._diversity_critic_optimizer[0].apply_gradients(
                policy._diversity_critic_grads_and_vars
            )
        ]

    alpha_apply_ops = policy._alpha_optimizer.apply_gradients(
        policy._alpha_grads_and_vars,
        global_step=tf.train.get_or_create_global_step()
    )
    return tf.group(
        [actor_apply_ops, alpha_apply_ops] + critic_apply_ops +
        diversity_critic_apply_ops
    )
