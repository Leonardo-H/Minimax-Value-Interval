# Instructions for running experiments

### Environments

```
python 3.7
Tensorflow==1.14.0
Numpy==1.17.2
gym==0.12.5
```



### Prepare Behavior & Target Policy (Optional)

We provide the well-trained policy (TF Model) in the `./CartPole_Model` directory. If you want to generate your own, you can follow this instructions:
```
# Step 1: We build our policy generating code based on OpenAI baselines. So first download baselines
git clone https://github.com/openai/baselines.git
git checkout 1f3c3e3

# Step 2: Install the baselines follow its official instructions

# Step 3: Copy updated files to baselines' directory
cd ./baselines/baselines/deepq
rsync -a [path of CI's code]/GeneratePolicy/* ./

# Step 4: Run DQN to generate policy. You may set your own training hyper-parameters
python train_CartPole.py

# Step 5: Move new model to replace the old one
mv ./CartPole-v0 [path of CI's code]/
```



### Evaluation Experiments

##### Compute Intervals

```shell
# Generate dataset with behavior policy
python CartPole_Gene_Data.py --n-ros 200 --ep-len 1000 --tau 1.0 --seed 100 200 300 400 500

# Compute true value of target policy:
python OnPolicy --tau 0.5 (or 2.5)

# MQL exps (batch run), target_tau = 0.5 or 2.5
python run_mql.py --iter 500 --dataset-seed 100 200 300 400 500 --scale 2.5 --tau 0.5 (or 2.5) --n-ros 200

# CI exps (batch run)
python run_ci_ope.py --iter 500 --dataset-seed 100 200 300 400 500 --scale 2.5 --tau 0.5 (or 2.5) --n-ros 200

# Plot results
python plot_ope.py --alg CI_OPE MQL_Interval --tau 0.5 (or 2.5) --scale 2.5 --n-ros 200 --y-lim 100 400 --save-path ./OPE.png
```



##### Varying the Q-Network Size 

```shell
python run_ci_ope.py --iter 200 --dataset-seed 100 200 300 400 500 --tau 2.5 --n-ros 200 --qs 32 32
python run_ci_ope.py --iter 200 --dataset-seed 100 200 300 400 500 --tau 2.5 --n-ros 200 --qs 32
python run_ci_ope.py --iter 200 --dataset-seed 100 200 300 400 500 --tau 2.5 --n-ros 200 --qs 10
python run_ci_ope.py --iter 200 --dataset-seed 100 200 300 400 500 --tau 2.5 --n-ros 200 --qs 3
python run_ci_ope.py --iter 200 --dataset-seed 100 200 300 400 500 --tau 2.5 --n-ros 200 --qs 1

# plot results
python plot_vary_qsize.py --tau 2.5 --save-path ./CI_Vary_Qsize.png
```



##### Bootstrapping Experiments

```shell
# Notations:
# your_n_ros: number of trajectories
# your_seed: different seeds for resampling dataset

# Generate dataset
python CartPole_Gene_Data.py --n-ros your_n_ros --ep-len 1000 --tau 1.0 --seed 100 200 300 400 500

# run bootstrapping exps with MQL
python run_mql.py --bootstrap --iter 200 --dataset-seed 100 200 300 400 500 --tau 1.5 --n-ros your_n_ros --seed your_seed

# run bootstrapping exps with CI
python run_ci_ope.py --bootstrap --iter 200 --dataset-seed 100 200 300 400 500 --tau 1.5 --n-ros your_n_ros --seed your_seed

# plot results
python plot_bootstrapping.py --alg CI_OPE MQL_Interval --tau 1.5 --scale 2.5 --max-index 1 --iter 100000
```



### Optimization Experiments

```shell
# Generate dataset 
python CartPole_Gene_Data.py --n-ros 200 --ep-len 1000 --tau 0.1 (or 1.0) --seed 100 200 300 400 500

# run optimization with CI
python run_ci_opt.py --iter 1000 --n-ros 200 --ep-len 1000 --tau 0.1 --scale 0.1 (or --tau 1.0 --scale 0.3) --pi-type lower --pi-iter 1 --dataset-seed 100 200 300 400 500
python run_ci_opt.py --iter 500 --n-ros 200 --ep-len 1000 --tau 0.1 --scale 0.1 (or --tau 1.0 --scale 0.3) --pi-type upper --pi-iter 500 --dataset-seed 100 200 300 400 500

# run DQN
python run_dqn.py --iter 500000 --n-ros 200 --ep-len 1000 --tau 0.1 (or 1.0) --tar-freq 1000 --dataset-seed 100 200 300 400 500

# plot results
## Policy Value
python plot_opt.py --x-lim 0 500000 --tau 0.1 --scale 0.1 (or --tau 1.0 --scale 0.3) --dataset 100 200 300 400 500  --plot-true --y-lim 0 300 --save-path ./Policy_Value.png
## IPM
python plot_opt.py --x-lim 0 500000 --tau 0.1 --scale 0.1 (or --tau 1.0 --scale 0.3) --dataset 100 200 300 400 500 --key IPM --save-path ./IPM.png
```





# Experiment Results

Please refer to the paper for detailed results.