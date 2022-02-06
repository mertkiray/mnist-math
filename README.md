# Unnecessarily Complicated Addition and Subtraction of Two Digits

### Requirements & Training - Testing
To run the code, first install the requirements with
```
pip install -r requirements.txt
```

To train on SuMNIST dataset, run the following code on
```
python main.py --config configs/mnistsum_config.txt 
```

To train on DiffSuMNist dataset, run the following code on
```
python main.py --config configs/diffmnistsum_config.txt 
```

The MNIST dataset will be downloaded automatically to a folder specified in the config file ```datadir```.


To view the logs from tensorboard, please run
```tensorboard --logdir runs```.

The experiments will be logged to tensorboard with the experiment name from the config ```expname```.

The checkpoints will be logged to the ```basedir``` into the specific experiment folder.

The frequency of checkpoint saving and logging to tensorboard can be changed by setting
```i_print``` and ```i_weight``` parameters.


To start the training from an existing checkpoint run the following
```python main.py --config configs/mnistsum_config.txt --ckpt {ckpt_path}```

The trained model can be tested by adding ```--test``` parameter and the checkpoint.
Example:

```python main.py --config configs/mnistsum_config.txt --ckpt {ckpt_path} --test```