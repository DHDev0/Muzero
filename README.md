# Muzero
Pytorch Implementation of MuZero for OpenAI gym environment. It should support any &lt;Discrete> or &lt;Box> configuration for the observation space and action space.

I try to document the code as much as possible next to the muzero paper: https://arxiv.org/abs/1911.08265

For example you can use the tutorial.ipynb

or

CLI with your config file (a cartpole example is available in config folder) :

Training :
```
cd \to\project\folder
python muzero_cli.py train config/cli_cartpole_config.json
```

Inference / Play :
```
cd \to\project\folder
python muzero_cli.py play config/cli_cartpole_config.json
```


|X| TODO: Need to add a board and atari configuration for muzero base on the paper hyperparameter
|X| TODO: Need to add multi CPU/GPU and cloud remote compute for aws, google cloud and azure with cluster
