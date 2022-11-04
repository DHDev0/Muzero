
## Muzero

Pytorch Implementation of MuZero for OpenAI gym environment. It should support any &lt;Discrete> or &lt;Box> configuration for the observation space and action space. You will have to change the game in config, choice between a RGB observation(vision_model) or Game state(mlp_model) as observation and find the best hyperparameter <br />
<br />
I try to document the code as much as possible next to the muzero paper: https://arxiv.org/abs/1911.08265 <br />
<br />
## Dependency

The step for the Dependency/Lib to use Muzero are in the first cell of the jupyter notebook.<br />
https://github.com/DHDev0/Muzero/blob/main/tutorial.ipynb

## Jupyter

For example you can use the tutorial.ipynb ( the step for the Dependency/Lib to use Muzero are in it ) <br />
https://github.com/DHDev0/Muzero/blob/main/tutorial.ipynb <br />
(Two approach : self explain variable and code comment)

## CLI

Set your config file (example): https://github.com/DHDev0/Muzero/blob/main/config/<br />
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

* [x] Residual Network for RGB observation using render.
* [x] Fully connected network for game state observation.
* [x] Model weights automatically saved at best selfplay reward.
* [x] Commented with link/page to the paper.
* [x] Work for any gym environments/games. (continous , discrete or any combinaison of the two for action and observation space)
* [x] Tutorial with jupyter notebook.
* [x] Pretrained weights for cartpole.
* [x] Support : Windows , Linux , MacOS.
* [x] Single player mode.
* [ ] Two player mode and more
* [ ] Board and atari configuration for muzero base on the paper hyperparameter<br />
* [ ] Multi CPU/GPU and cloud remote compute for aws, google cloud and azure with cluster<br />

## Subjects

Machine Learning
