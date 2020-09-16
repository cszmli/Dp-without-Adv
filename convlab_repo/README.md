# ConvLab
ConvLab is an open-source multi-domain end-to-end dialog system platform, aiming to enable researchers to quickly set up experiments with reusable components and compare a large set of different approaches, ranging from conventional pipeline systems to end-to-end neural models, in common environments.

## Package Overview
<table>
<tr>
    <td><b> convlab </b></td>
    <td> an open-source multi-domain end-to-end dialog research library </td>
</tr>
<tr>
    <td><b> convlab.agent </b></td>
    <td> a module for constructing dialog agents including RL algorithms </td>
</tr>
<tr>
    <td><b> convlab.env </b></td>
    <td> a collection of environments </td>
</tr>
<tr>
    <td><b> convlab.experiment </b></td>
    <td> a module for running experiments at various levels </td>
</tr>
<tr>
    <td><b> convlab.evaluator </b></td>
    <td> a module for evaluating a dialog session with various metrics </td>
</tr>
<tr>
    <td><b> convlab.modules </b></td>
    <td> a collection of state-of-the-art dialog system component models including NLU, DST, Policy, NLG </td>
</tr>
<tr>
    <td><b> convlab.human_eval </b></td>
    <td> a server for conducting human evaluation using Amazon Mechanical Turk </td>
</tr>
<tr>
    <td><b> convlab.lib </b></td>
    <td> a library of common utilities </td>
</tr>
<tr>
    <td><b> convlab.spec </b></td>
    <td> a collection of experiment spec files </td>
</tr>
</table>

## Installation
ConvLab requires Python 3.6.5 or later. Windows is currently not offically supported.

### Installing via pip

#### Setting up a virtual environment

[Conda](https://conda.io/) can be used to set up a virtual environment with the
version of Python required for ConvLab.  If you already have a Python 3.6 or 3.7
environment you want to use, you can skip to the 'installing via pip' section.

1.  [Download and install Conda](https://conda.io/docs/download.html).

2.  Create a Conda environment with Python 3.6.5

    ```bash
    conda create -n convlab python=3.6.5
    ```

3.  Activate the Conda environment. You will need to activate the Conda environment in each terminal in which you want to use ConvLab.

    ```bash
    source activate convlab
    ```

#### Installing the library and dependencies

Installing the library and dependencies is simple using `pip`.

   ```bash
   pip install -r requirements.txt
   ```
If your Linux system does not have essential building tools installed, you might need to install it by running
 ```bash
 sudo apt-get install build-essential
 ```
ConvLab uses 'stopwords' in nltk, and you need to download it by running
```bash
python -m nltk.downloader stopwords
```

## Running ConvLab
Once you've downloaded ConvLab and installed required packages, you can run the command-line interface with the `python run.py` command.
```bash
$ python run.py {spec file} {spec name} {mode}
```


Running DQN + R(vae-gan):
```bash
# to train a DQN policy with NLU(OneNet), DST(Rule), NLG(Template) on the MultiWOZ environment
$ python -u run.py demo.json rule_dqn train

# to use the policy trained above (this will load up the onenet_rule_dqn_template_t0_s0_*.pt files under the output/onenet_rule_dqn_template_{timestamp}/model directory)
$ python run.py demo.json rule_dqn eval@output/rule_dqn_{timestamp}/model/rule_dqn_t0_s0
```



Running PPO + R(vae-gan):
```bash
# to train a DQN policy with NLU(OneNet), DST(Rule), NLG(Template) on the MultiWOZ environment
$ python -u run.py demo.json rule_ppo train

# to use the policy trained above (this will load up the onenet_rule_dqn_template_t0_s0_*.pt files under the output/onenet_rule_dqn_template_{timestamp}/model directory)
$ python run.py demo.json rule_ppo eval@output/rule_ppo_{timestamp}/model/rule_ppo_t0_s0
```

The data can found here: [zip](https://drive.google.com/file/d/16BbNowEfSUZTRO0BqDbTGeg5ZBKq7xkE/view?usp=sharing).



The pre-trained models can be found here: [zip](https://drive.google.com/file/d/1RdcG4nHlS4NqDtNWUscTU-mp4NZyWwOA/view?usp=sharing).

## Creating a new spec file
A spec file is used to fully specify experiments including a dialog agent and a user simulator. It is a JSON of multiple experiment specs, each containing the keys agent, env, body, meta, search.

We based our implementation on [SLM-Lab](https://github.com/kengz/SLM-Lab/tree/master/slm_lab). For an introduction to these concepts, you should check [these docs](https://kengz.gitbooks.io/slm-lab/content/).

Instead of writing one from scratch, you are welcome to modify the `convlab/spec/demo.json` file. Once you have created a new spec file, place it under `convlab/spec` directory and run your experiments. Note that you don't have to prepend `convlab/spec/` before your spec file name.


