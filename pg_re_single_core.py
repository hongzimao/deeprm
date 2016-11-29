import numpy as np
import time
import theano
import cPickle
import matplotlib.pyplot as plt

import environment
import pg_network
import slow_down_cdf


def discount(x, gamma):
    """
    Given vector x, computes a vector y such that
    y[i] = x[i] + gamma * x[i+1] + gamma^2 x[i+2] + ...
    """
    out = np.zeros(len(x))
    out[-1] = x[-1]
    for i in reversed(xrange(len(x)-1)):
        out[i] = x[i] + gamma*out[i+1]
    assert x.ndim >= 1
    # More efficient version:
    # scipy.signal.lfilter([1],[1,-gamma],x[::-1], axis=0)[::-1]
    return out


def get_entropy(vec):
    entropy = - np.sum(vec * np.log(vec))
    if np.isnan(entropy):
        entropy = 0
    return entropy


def get_traj(agent, env, episode_max_length, render=False):
    """
    Run agent-environment loop for one whole episode (trajectory)
    Return dictionary of results
    """
    env.reset()
    obs = []
    acts = []
    rews = []
    entropy = []
    info = []

    ob = env.observe()

    for _ in xrange(episode_max_length):
        act_prob = agent.get_one_act_prob(ob)
        csprob_n = np.cumsum(act_prob)
        a = (csprob_n > np.random.rand()).argmax()

        obs.append(ob)  # store the ob at current decision making step
        acts.append(a)

        ob, rew, done, info = env.step(a, repeat=True)

        rews.append(rew)
        entropy.append(get_entropy(act_prob))

        if done: break
        if render: env.render()

    return {'reward': np.array(rews),
            'ob': np.array(obs),
            'action': np.array(acts),
            'entropy': entropy,
            'info': info
            }


def concatenate_all_ob(trajs, pa):

    timesteps_total = 0
    for i in xrange(len(trajs)):
        timesteps_total += len(trajs[i]['reward'])

    all_ob = np.zeros(
        (timesteps_total, 1, pa.network_input_height, pa.network_input_width),
        dtype=theano.config.floatX)

    timesteps = 0
    for i in xrange(len(trajs)):
        for j in xrange(len(trajs[i]['reward'])):
            all_ob[timesteps, 0, :, :] = trajs[i]['ob'][j]
            timesteps += 1

    return all_ob


def concatenate_all_ob_across_examples(all_ob, pa):
    num_ex = len(all_ob)
    total_samp = 0
    for i in xrange(num_ex):
        total_samp += all_ob[i].shape[0]

    all_ob_contact = np.zeros(
        (total_samp, 1, pa.network_input_height, pa.network_input_width),
        dtype=theano.config.floatX)

    total_samp = 0

    for i in xrange(num_ex):
        prev_samp = total_samp
        total_samp += all_ob[i].shape[0]
        all_ob_contact[prev_samp : total_samp, :, :, :] = all_ob[i]

    return all_ob_contact


def process_all_info(trajs):
    enter_time = []
    finish_time = []
    job_len = []

    for traj in trajs:
        enter_time.append(np.array([traj['info'].record[i].enter_time for i in xrange(len(traj['info'].record))]))
        finish_time.append(np.array([traj['info'].record[i].finish_time for i in xrange(len(traj['info'].record))]))
        job_len.append(np.array([traj['info'].record[i].len for i in xrange(len(traj['info'].record))]))

    enter_time = np.concatenate(enter_time)
    finish_time = np.concatenate(finish_time)
    job_len = np.concatenate(job_len)

    return enter_time, finish_time, job_len


def plot_lr_curve(output_file_prefix, max_rew_lr_curve, mean_rew_lr_curve, slow_down_lr_curve,
                  ref_discount_rews, ref_slow_down):
    num_colors = len(ref_discount_rews) + 2
    cm = plt.get_cmap('gist_rainbow')

    fig = plt.figure(figsize=(12, 5))

    ax = fig.add_subplot(121)
    ax.set_color_cycle([cm(1. * i / num_colors) for i in range(num_colors)])

    ax.plot(mean_rew_lr_curve, linewidth=2, label='PG mean')
    for k in ref_discount_rews:
        ax.plot(np.tile(np.average(ref_discount_rews[k]), len(mean_rew_lr_curve)), linewidth=2, label=k)
    ax.plot(max_rew_lr_curve, linewidth=2, label='PG max')

    plt.legend(loc=4)
    plt.xlabel("Iteration", fontsize=20)
    plt.ylabel("Discounted Total Reward", fontsize=20)

    ax = fig.add_subplot(122)
    ax.set_color_cycle([cm(1. * i / num_colors) for i in range(num_colors)])

    ax.plot(slow_down_lr_curve, linewidth=2, label='PG mean')
    for k in ref_discount_rews:
        ax.plot(np.tile(np.average(np.concatenate(ref_slow_down[k])), len(slow_down_lr_curve)), linewidth=2, label=k)

    plt.legend(loc=1)
    plt.xlabel("Iteration", fontsize=20)
    plt.ylabel("Slowdown", fontsize=20)

    plt.savefig(output_file_prefix + "_lr_curve" + ".pdf")


def launch(pa, pg_resume=None, render=False, repre='image', end='no_new_job'):

    env = environment.Env(pa, render=render, repre=repre, end=end)

    pg_learner = pg_network.PGLearner(pa)

    if pg_resume is not None:
        net_handle = open(pg_resume, 'rb')
        net_params = cPickle.load(net_handle)
        pg_learner.set_net_params(net_params)

    # ----------------------------
    print("Preparing for data...")
    # ----------------------------

    ref_discount_rews, ref_slow_down = slow_down_cdf.launch(pa, pg_resume=None, render=False, plot=False, repre=repre, end=end)

    mean_rew_lr_curve = []
    max_rew_lr_curve = []
    slow_down_lr_curve = []

    timer_start = time.time()

    for iteration in xrange(pa.num_epochs):

        all_ob = []
        all_action = []
        all_adv = []
        all_eprews = []
        all_eplens = []
        all_slowdown = []
        all_entropy = []

        # go through all examples
        for ex in xrange(pa.num_ex):

            # Collect trajectories until we get timesteps_per_batch total timesteps
            trajs = []

            for i in xrange(pa.num_seq_per_batch):
                traj = get_traj(pg_learner, env, pa.episode_max_length)
                trajs.append(traj)

            # roll to next example
            env.seq_no = (env.seq_no + 1) % env.pa.num_ex

            all_ob.append(concatenate_all_ob(trajs, pa))

            # Compute discounted sums of rewards
            rets = [discount(traj["reward"], pa.discount) for traj in trajs]
            maxlen = max(len(ret) for ret in rets)
            padded_rets = [np.concatenate([ret, np.zeros(maxlen - len(ret))]) for ret in rets]

            # Compute time-dependent baseline
            baseline = np.mean(padded_rets, axis=0)

            # Compute advantage function
            advs = [ret - baseline[:len(ret)] for ret in rets]
            all_action.append(np.concatenate([traj["action"] for traj in trajs]))
            all_adv.append(np.concatenate(advs))

            all_eprews.append(np.array([discount(traj["reward"], pa.discount)[0] for traj in trajs]))  # episode total rewards
            all_eplens.append(np.array([len(traj["reward"]) for traj in trajs]))  # episode lengths

            # All Job Stat
            enter_time, finish_time, job_len = process_all_info(trajs)
            finished_idx = (finish_time >= 0)
            all_slowdown.append(
                (finish_time[finished_idx] - enter_time[finished_idx]) / job_len[finished_idx]
            )

            # Action prob entropy
            all_entropy.append(np.concatenate([traj["entropy"]]))

        all_ob = concatenate_all_ob_across_examples(all_ob, pa)
        all_action = np.concatenate(all_action)
        all_adv = np.concatenate(all_adv)

        # Do policy gradient update step
        loss = pg_learner.train(all_ob, all_action, all_adv)
        eprews = np.concatenate(all_eprews)  # episode total rewards
        eplens = np.concatenate(all_eplens)  # episode lengths

        all_slowdown = np.concatenate(all_slowdown)

        all_entropy = np.concatenate(all_entropy)

        timer_end = time.time()

        print "-----------------"
        print "Iteration: \t %i" % iteration
        print "NumTrajs: \t %i" % len(eprews)
        print "NumTimesteps: \t %i" % np.sum(eplens)
        print "Loss:     \t %s" % loss
        print "MaxRew: \t %s" % np.average([np.max(rew) for rew in all_eprews])
        print "MeanRew: \t %s +- %s" % (eprews.mean(), eprews.std())
        print "MeanSlowdown: \t %s" % np.mean(all_slowdown)
        print "MeanLen: \t %s +- %s" % (eplens.mean(), eplens.std())
        print "MeanEntropy \t %s" % (np.mean(all_entropy))
        print "Elapsed time\t %s" % (timer_end - timer_start), "seconds"
        print "-----------------"

        timer_start = time.time()

        max_rew_lr_curve.append(np.average([np.max(rew) for rew in all_eprews]))
        mean_rew_lr_curve.append(eprews.mean())
        slow_down_lr_curve.append(np.mean(all_slowdown))

        if iteration % pa.output_freq == 0:
            param_file = open(pa.output_filename + '_' + str(iteration) + '.pkl', 'wb')
            cPickle.dump(pg_learner.get_params(), param_file, -1)
            param_file.close()

            slow_down_cdf.launch(pa, pa.output_filename + '_' + str(iteration) + '.pkl',
                                 render=False, plot=True, repre=repre, end=end)

            plot_lr_curve(pa.output_filename,
                          max_rew_lr_curve, mean_rew_lr_curve, slow_down_lr_curve,
                          ref_discount_rews, ref_slow_down)


def main():

    import parameters

    pa = parameters.Parameters()

    pa.simu_len = 200  # 1000
    pa.num_ex = 10  # 100
    pa.num_nw = 10
    pa.num_seq_per_batch = 20
    pa.output_freq = 50

    # pa.max_nw_size = 5
    # pa.job_len = 5
    pa.new_job_rate = 0.3

    pa.episode_max_length = 2000  # 2000

    pa.compute_dependent_parameters()

    pg_resume = None
    # pg_resume = 'data/tmp_0.pkl'

    render = False

    launch(pa, pg_resume, render, repre='image', end='all_done')


if __name__ == '__main__':
    main()
