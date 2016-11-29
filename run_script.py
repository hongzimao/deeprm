# /usr/bin/env python

import os

simu_len = 200

for new_job_rate in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
    for num_seq_per_batch in [20]:
        for num_ex in [100]:
            for num_nw in [10]:

                file_name = 'data/pg_re_rate_' + str(new_job_rate) + '_simu_len_' + str(simu_len) + '_num_seq_per_batch_' + str(num_seq_per_batch) + '_ex_' + str(num_ex) + '_nw_' + str(num_nw)
                log = 'log/pg_re_rate_' + str(new_job_rate) + '_simu_len_' + str(simu_len) + '_num_seq_per_batch_' + str(num_seq_per_batch) + '_ex_' + str(num_ex) + '_nw_' + str(num_nw)

                # run experiment
                os.system('nohup python -u launcher.py --exp_type=pg_re --out_freq=50 --simu_len=' + str(simu_len) + ' --eps_max_len=' + str(simu_len * 4) + ' --num_ex=' + str(num_ex) + ' --new_job_rate=' + str(new_job_rate) + ' --num_seq_per_batch=' + str(num_seq_per_batch) + ' --num_nw=' + str(num_nw) + ' --ofile=' + file_name + ' > ' + log + ' &')

                # plot slowdown
                # it_num = 100
                # os.system('nohup python -u launcher.py --exp_type=test --simu_len=' + str(simu_len) + '--num_ex=' + str(num_ex) + ' --new_job_rate=' + str(new_job_rate) + ' --num_seq_per_batch=' + str(num_seq_per_batch) + ' --pg_re=' + file_name + '_' + str(it_num) + '.pkl' + ' &')
