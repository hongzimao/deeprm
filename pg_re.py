import time
import threading
import numpy as np
import theano
import cPickle
import matplotlib.pyplot as plt

from multiprocessing import Process
from multiprocessing import Manager

import environment
import job_distribution
import pg_network
import slow_down_cdf


def init_accums(pg_learner):  # in rmsprop
    accums = []
    params = pg_learner.get_params()
    for param in params:
        accum = np.zeros(param.shape, dtype=param.dtype)
        accums.append(accum)
    return accums


def rmsprop_updates_outside(grads, params, accums, stepsize, rho=0.9, epsilon=1e-9):

    assert len(grads) == len(params)
    assert len(grads) == len(accums)
    for dim in xrange(len(grads)):
        accums[dim] = rho * accums[dim] + (1 - rho) * grads[dim] ** 2
        params[dim] += (stepsize * grads[dim] / np.sqrt(accums[dim] + epsilon))


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


def get_traj(agent, env, episode_max_length):
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


def get_traj_worker(pg_learner, env, pa, result):

    trajs = []

    for i in xrange(pa.num_seq_per_batch):
        traj = get_traj(pg_learner, env, pa.episode_max_length)
        trajs.append(traj)

    all_ob = concatenate_all_ob(trajs, pa)

    # Compute discounted sums of rewards
    rets = [discount(traj["reward"], pa.discount) for traj in trajs]
    maxlen = max(len(ret) for ret in rets)
    padded_rets = [np.concatenate([ret, np.zeros(maxlen - len(ret))]) for ret in rets]

    # Compute time-dependent baseline
    baseline = np.mean(padded_rets, axis=0)

    # Compute advantage function
    advs = [ret - baseline[:len(ret)] for ret in rets]
    all_action = np.concatenate([traj["action"] for traj in trajs])
    all_adv = np.concatenate(advs)

    all_eprews = np.array([discount(traj["reward"], pa.discount)[0] for traj in trajs])  # episode total rewards
    all_eplens = np.array([len(traj["reward"]) for traj in trajs])  # episode lengths

    # All Job Stat
    enter_time, finish_time, job_len = process_all_info(trajs)
    finished_idx = (finish_time >= 0)
    all_slowdown = (finish_time[finished_idx] - enter_time[finished_idx]) / job_len[finished_idx]

    all_entropy = np.concatenate([traj["entropy"] for traj in trajs])

    result.append({"all_ob": all_ob,
                   "all_action": all_action,
                   "all_adv": all_adv,
                   "all_eprews": all_eprews,
                   "all_eplens": all_eplens,
                   "all_slowdown": all_slowdown,
                   "all_entropy": all_entropy})


def launch(pa, pg_resume=None, render=False, repre='image', end='no_new_job'):

    # ----------------------------
    print("Preparing for workers...")
    # ----------------------------

    pg_learners = []
    envs = []

    nw_len_seqs, nw_size_seqs = job_distribution.generate_sequence_work(pa, seed=42)

    for ex in xrange(pa.num_ex):

        print "-prepare for env-", ex

        env = environment.Env(pa, nw_len_seqs=nw_len_seqs, nw_size_seqs=nw_size_seqs,
                              render=False, repre=repre, end=end)
        env.seq_no = ex
        envs.append(env)

    for ex in xrange(pa.batch_size + 1):  # last worker for updating the parameters

        print "-prepare for worker-", ex

        pg_learner = pg_network.PGLearner(pa)

        if pg_resume is not None:
            net_handle = open(pg_resume, 'rb')
            net_params = cPickle.load(net_handle)
            pg_learner.set_net_params(net_params)

        pg_learners.append(pg_learner)

    accums = init_accums(pg_learners[pa.batch_size])

    # --------------------------------------
    print("Preparing for reference data...")
    # --------------------------------------

    ref_discount_rews, ref_slow_down = slow_down_cdf.launch(pa, pg_resume=None, render=False, plot=False, repre=repre, end=end)
    mean_rew_lr_curve = []
    max_rew_lr_curve = []
    slow_down_lr_curve = []

    # --------------------------------------
    print("Start training...")
    # --------------------------------------

    timer_start = time.time()

    for iteration in xrange(1, pa.num_epochs):

        ps = []  # threads
        manager = Manager()  # managing return results
        manager_result = manager.list([])

        ex_indices = range(pa.num_ex)
        np.random.shuffle(ex_indices)

        all_eprews = []
        grads_all = []
        loss_all = []
        eprews = []
        eplens = []
        all_slowdown = []
        all_entropy = []

        ex_counter = 0
        for ex in xrange(pa.num_ex):

            ex_idx = ex_indices[ex]
            p = Process(target=get_traj_worker,
                        args=(pg_learners[ex_counter], envs[ex_idx], pa, manager_result, ))
            ps.append(p)

            ex_counter += 1

            if ex_counter >= pa.batch_size or ex == pa.num_ex - 1:

                print ex, "out of", pa.num_ex

                ex_counter = 0

                for p in ps:
                    p.start()

                for p in ps:
                    p.join()

                result = []  # convert list from shared memory
                for r in manager_result:
                    result.append(r)

                ps = []
                manager_result = manager.list([])

                all_ob = concatenate_all_ob_across_examples([r["all_ob"] for r in result], pa)
                all_action = np.concatenate([r["all_action"] for r in result])
                all_adv = np.concatenate([r["all_adv"] for r in result])

                # Do policy gradient update step, using the first agent
                # put the new parameter in the last 'worker', then propagate the update at the end
                grads = pg_learners[pa.batch_size].get_grad(all_ob, all_action, all_adv)

                grads_all.append(grads)

                all_eprews.extend([r["all_eprews"] for r in result])

                eprews.extend(np.concatenate([r["all_eprews"] for r in result]))  # episode total rewards
                eplens.extend(np.concatenate([r["all_eplens"] for r in result]))  # episode lengths

                all_slowdown.extend(np.concatenate([r["all_slowdown"] for r in result]))
                all_entropy.extend(np.concatenate([r["all_entropy"] for r in result]))

        # assemble gradients
        grads = grads_all[0]
        for i in xrange(1, len(grads_all)):
            for j in xrange(len(grads)):
                grads[j] += grads_all[i][j]

        # propagate network parameters to others
        params = pg_learners[pa.batch_size].get_params()

        rmsprop_updates_outside(grads, params, accums, pa.lr_rate, pa.rms_rho, pa.rms_eps)

        for i in xrange(pa.batch_size + 1):
            pg_learners[i].set_net_params(params)

        timer_end = time.time()

        print "-----------------"
        print "Iteration: \t %i" % iteration
        print "NumTrajs: \t %i" % len(eprews)
        print "NumTimesteps: \t %i" % np.sum(eplens)
        # print "Loss:     \t %s" % np.mean(loss_all)
        print "MaxRew: \t %s" % np.average([np.max(rew) for rew in all_eprews])
        print "MeanRew: \t %s +- %s" % (np.mean(eprews), np.std(eprews))
        print "MeanSlowdown: \t %s" % np.mean(all_slowdown)
        print "MeanLen: \t %s +- %s" % (np.mean(eplens), np.std(eplens))
        print "MeanEntropy \t %s" % (np.mean(all_entropy))
        print "Elapsed time\t %s" % (timer_end - timer_start), "seconds"
        print "-----------------"

        timer_start = time.time()

        max_rew_lr_curve.append(np.average([np.max(rew) for rew in all_eprews]))
        mean_rew_lr_curve.append(np.mean(eprews))
        slow_down_lr_curve.append(np.mean(all_slowdown))

        if iteration % pa.output_freq == 0:
            param_file = open(pa.output_filename + '_' + str(iteration) + '.pkl', 'wb')
            cPickle.dump(pg_learners[pa.batch_size].get_params(), param_file, -1)
            param_file.close()

            pa.unseen = True
            slow_down_cdf.launch(pa, pa.output_filename + '_' + str(iteration) + '.pkl',
                                 render=False, plot=True, repre=repre, end=end)
            pa.unseen = False
            # test on unseen examples

            plot_lr_curve(pa.output_filename,
                          max_rew_lr_curve, mean_rew_lr_curve, slow_down_lr_curve,
                          ref_discount_rews, ref_slow_down)


def main():

    import parameters

    pa = parameters.Parameters()

    pa.simu_len = 50  # 1000
    pa.num_ex = 50  # 100
    pa.num_nw = 10
    pa.num_seq_per_batch = 20
    pa.output_freq = 50
    pa.batch_size = 10

    # pa.max_nw_size = 5
    # pa.job_len = 5
    pa.new_job_rate = 0.3

    pa.episode_max_length = 2000  # 2000

    pa.compute_dependent_parameters()

    pg_resume = None
    # pg_resume = 'data/tmp_450.pkl'

    render = False

    launch(pa, pg_resume, render, repre='image', end='all_done')


if __name__ == '__main__':
    main()
