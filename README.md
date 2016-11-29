# DeepRM
HotNets'16 http://people.csail.mit.edu/hongzi/content/publications/DeepRM-HotNets16.pdf

Install prerequisites

```
sudo apt-get update
sudo apt-get install python-numpy python-scipy python-dev python-pip python-nose g++ libopenblas-dev git
pip install --user Theano
pip install --user Lasagne==0.1
sudo apt-get install python-matplotlib
```

In folder RL, create a data/ folder. 

Use `launcher.py` to launch experiments. 


```
--exp_type <type of experiment> 
--num_res <number of resources> 
--num_nw <number of visible new work> 
--simu_len <simulation length> 
--num_ex <number of examples> 
--num_seq_per_batch <rough number of samples in one batch update> 
--eps_max_len <episode maximum length (terminated at the end)>
--num_epochs <number of epoch to do the training>
--time_horizon <time step into future, screen height> 
--res_slot <total number of resource slots, screen width> 
--max_job_len <maximum new job length> 
--max_job_size <maximum new job resource request> 
--new_job_rate <new job arrival rate> 
--dist <discount factor> 
--lr_rate <learning rate> 
--ba_size <batch size> 
--pg_re <parameter file for pg network> 
--v_re <parameter file for v network> 
--q_re <parameter file for q network> 
--out_freq <network output frequency> 
--ofile <output file name> 
--log <log file name> 
--render <plot dynamics> 
--unseen <generate unseen example> 
```


The default variables are defined in `parameters.py`.


Example: 
  - launch supervised learning for policy estimation 
  
  ```
  python launcher.py --exp_type=pg_su --simu_len=50 --num_ex=1000 --ofile=data/pg_su --out_freq=10 
  ```
  - launch policy gradient using network parameter just obtained
  
  ```
  python launcher.py --exp_type=pg_re --pg_re=data/pg_su_net_file_20.pkl --simu_len=50 --num_ex=10 --ofile=data/pg_re
  ```
  - launch testing and comparing experiemnt on unseen examples with pg agent just trained
  
  ```
  python launcher.py --exp_type=test --simu_len=50 --num_ex=10 --pg_re=data/pg_re_1600.pkl --unseen=True
  ```
