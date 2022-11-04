# Muzero
Pytorch Implementation of MuZero for OpenAI gym environment. It should support any &lt;Discrete> or &lt;Box> configuration for the observation space and action space. You just have to change the game and find the best hyperparameter <br />
<br />
I try to document the code as much as possible next to the muzero paper: https://arxiv.org/abs/1911.08265 <br />
<br />
For example you can use the tutorial.ipynb ( the step for the Dependency/Lib to use Muzero are in it ) <br />
<br />
or<br />
<br />
CLI with your config file (a cartpole example is available in config folder) : <br />
<br />
Training : <br />
```
cd \to\project\folder
python muzero_cli.py train config/cli_cartpole_config.json
```

Inference / Play :<br />
```
cd \to\project\folder
python muzero_cli.py play config/cli_cartpole_config.json
```

## Features

* [x] Residual Network  
* [x] Fully connected network in [PyTorch](https://github.com/pytorch/pytorch)
* [x] Model weights automatically saved at checkpoints
* [x] Commented and 
* [x] Work for any gym environments/games (continous , discrete or any combinaison of the two for action and observation space)
* [x] Tutorial
* [x] Pretrained weights
* [x] Support : Windows , Linux , MacOS
* [x] Single player mode
* [ ] Two player mode and more
* [ ] Need to add a board and atari configuration for muzero base on the paper hyperparameter<br />
* [ ] Need to add multi CPU/GPU and cloud remote compute for aws, google cloud and azure with cluster<br />
