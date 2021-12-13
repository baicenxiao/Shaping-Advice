**Status:** Archive (code is provided as-is, no updates expected)

# Shaping Advice in Deep Multi-Agent Reinforcement Learning

## Summary

Reinforcement learning involves agents interacting with an environment to complete tasks. When rewards provided by the environment are sparse, agents may not receive immediate
feedback on the quality of actions that they take, thereby affecting learning of policies. In this paper, we propose methods to augment the reward signal from the environment with an additional reward termed shaping advice in both single- and multi-agent reinforcement learning. The shaping advice is specified as a difference of potential functions at consecutive time-steps. Each potential function is a function of observations and actions of the agents. The use of potential functions is underpinned by an insight that the total potential when starting from any state and returning to the same state is always equal to zero. The shaping advice needs to be specified only once at the start of training, and can easily be provided by non-experts. Our contributions are twofold. We show through theoretical analyses and experimental validation that the shaping advice does not distract agents from completing tasks specified by the environment reward. Theoretically, we prove that the convergence of policy gradients and value functions when using shaping advice implies the convergence of these quantities in the same environment in the absence of shaping advice. Experimentally, we evaluate proposed algorithms on two tasks in single-agent environments and three tasks in multi-agent environments that have sparse rewards. We observe that using shaping advice results in agents learning policies to complete tasks faster, and obtain higher rewards than algorithms that do not use shaping advice.

## Code description

This code presents a Python implementation of the SAS and SAM algorithm from the paper: 

[Shaping Advice in Deep Reinforcement Learning](https://arxiv.org/abs/2103.15941)
<!-- [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/pdf/1706.02275.pdf). -->
SAM is configured to be run in conjunction with multi-agent reinforcement learning environments from the
[Multi-Agent Particle Environments (MPE)](https://github.com/openai/multiagent-particle-envs).
Different from the original MPE environment where rewards were dense, our work uses a sparse reward structure.

SAS is configured to be run in conjunction with continuous MountainCar from OpenAI gym and a grid world puddle-jump environment.
Note: This code base has been restructured compared to the original paper, and some results may be different.

## Installation

- To install, `cd` into the root directory and type `pip install -e .`

- Known dependencies: Python (3.5.4), OpenAI gym (0.10.5), TensorFlow (1.9.0), numpy (1.15.2)

## Case study: Run SAM-NonUniform in simple-spread Environment

- To run SAM-NonUniform in simple spread, `cd` into the `experiments` directory and run `train.py`:

``python train_spread.py --scenario=simple_spread --num-episodes=60000 --save-dir=./logs/simple_spread/``

- To visualize the play if the saved model is in `./logs/simple_spread/`:

``python train.py --scenario=simple_spread --num-episodes=60000 --display --load-dir=./logs/simple_spread/``

- Here are examples for running SAM on other environments:

``python train_tag.py --scenario=simple_tag --num-adversaries=3 --num-episodes=60000 --save-dir=./logs/simple_tag/``

``python train_adv.py --scenario=simple_adversary --num-adversaries=1 --num-episodes=60000 --save-dir=./logs/simple_adversary/``

- For comparison, you can run [IRCR](https://arxiv.org/abs/2010.12718) or MADDPG alone on MPE with sparse reward:

``python train_IRCR_spread.py --scenario=simple_spread --num-episodes=60000 --save-dir=./logs/simple_spread_IRCR/``

``python train.py --scenario=simple_spread --num-episodes=60000 --save-dir=./logs/simple_spread_MADDPG_alone/``

## Command-line options

### Environment options

- `--scenario`: defines which environment in the MPE is to be used (options: `simple_spread`, `simple_tag`, `simple_adversary`)

- `--max-episode-len` maximum length of each episode for the environment (default: `25`)

- `--num-episodes` total number of training episodes (default: `60000`)

- `--num-adversaries`: number of adversaries in the environment (default: `0`)

### Core training parameters

- `--lr`: learning rate (default: `1e-2`)

- `--gamma`: discount factor (default: `0.95`)

- `--batch-size`: batch size (default: `1024`)

- `--num-units`: number of units in the MLP (default: `64`)

### Checkpointing

- `--exp-name`: name of the experiment, used as the file name to save all results (default: `None`)

- `--save-dir`: directory where intermediate training results and model will be saved (default: `"/tmp/policy/"`)

- `--save-rate`: model is saved every time this number of episodes has been completed (default: `1000`)

- `--load-dir`: directory where training state and model are loaded from (default: `""`)

### Evaluation

- `--restore`: restores previous training state stored in `load-dir` (or in `save-dir` if no `load-dir`
has been provided), and continues training (default: `False`)

- `--display`: displays to the screen the trained policy stored in `load-dir` (or in `save-dir` if no `load-dir`
has been provided), but does not continue training (default: `False`)

- `--benchmark-dir`: directory where benchmarking data is saved (default: `"./benchmark_files/"`)

- `--plots-dir`: directory where training curves are saved (default: `"./learning_curves/"`)

## Code structure

- `./experiments/train.py`: contains code for SAM-Uniform and training MADDPG on the MPE

- `./maddpg/trainer/maddpg.py`: core code for SAM-NonUniform and the MADDPG algorithm

- `./maddpg/trainer/replay_buffer.py`: replay buffer code for MADDPG

- `./AC_Continuous_Mountaincar.ipynb`: Notebook for SAS on the Continuous MountainCar

- `./PG-Puddle-Jump.ipynb`: Notebook for SAS on the puddle-jump grid world

Note: You may freely redistribute and use this sample code, with or without modification, provided you include the original Copyright notice and use restrictions.

## Disclaimer

THE SAMPLE CODE IS PROVIDED "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL BAICEN XIAO OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) SUSTAINED BY YOU OR A THIRD PARTY, HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT ARISING IN ANY WAY OUT OF THE USE OF THIS SAMPLE CODE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

## Acknowledgements

This work was supported by the U.S. Office of Naval Research via Grant N00014-17-S-B001. 

The code of MADDPG is based on the publicly available implementation: https://github.com/openai/maddpg.

## Additional Information

Project Webpage: Feedback-driven Learn to Reason in Adversarial Environments for Autonomic Cyber Systems (http://labs.ece.uw.edu/nsl/faculty/ProjectWebPages/L2RAVE/)


## Paper citation

If you used this code for your experiments or found it helpful, please cite the following paper:

Bibtex:
<pre>
@article{xiao2021shaping,
  title={Shaping Advice in Deep Multi-Agent Reinforcement Learning
},
  author={Xiao, Baicen and Ramasubramanian, Bhaskar and Poovendran, Radha},
  journal={arXiv preprint arXiv:2103.15941},
  year={2021}
}
</pre>