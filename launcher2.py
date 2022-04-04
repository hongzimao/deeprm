import slow_down_cdf
import parameters
import numpy as np

pa = parameters.Parameters()
pa.simu_len = 20
pa.num_ex = 5
ref_rewards, ref_slow_down = slow_down_cdf.launch(pa, render=True, plot=True)
print '\n---------- Total Discount Rewards ----------'
print 'Random2: ' + str(np.average(ref_rewards['Random2']))
print 'SJF2: ' + str(np.average(ref_rewards['SJF2']))
print 'Packer2: ' + str(np.average(ref_rewards['Packer2']))
print 'Tetris2: ' + str(np.average(ref_rewards['Tetris2']))

# print sd[1]['Random2']
# print np.average(np.concatenate(sd[1]['Random2']))
print '\n---------- Average Job Slowdown ----------'
print 'Random2: ' + str(np.average(np.concatenate(ref_slow_down['Random2'])))
print 'SJF2: ' + str(np.average(np.concatenate(ref_slow_down['SJF2'])))
print 'Packer2: ' + str(np.average(np.concatenate(ref_slow_down['Packer2'])))
print 'Tetris2: ' + str(np.average(np.concatenate(ref_slow_down['Tetris2'])))
