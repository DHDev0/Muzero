# Muzero
Pytorch Implementation of MuZero for OpenAI gym environment. It should support any &lt;Discrete> or &lt;Box> configuration for the observation space and action space.

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
