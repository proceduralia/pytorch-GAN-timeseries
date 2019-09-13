# Financial time series generation using GANs
This repository contains the implementation of a GAN-based method for real-valued financial time series generation. See for instance [Real-valued (Medical) Time Series Generation with Recurrent Conditional GANs](https://arxiv.org/abs/1706.02633).

<div style="text-align:center"> <img src="https://github.com/proceduralia/proceduralia.github.io/blob/master/assets/whiteseries.png?raw=true" width="500" height="300"/> </div>

Main features:
- [Causal Convolution](https://arxiv.org/abs/1609.03499) or LSTM architectures for disciminator and generator
- Non-saturing GAN training (see this [tutorial](https://arxiv.org/abs/1701.00160) for more info) 
- Generation can be unconditioned or conditioned on the difference between the last and the first element of the time series to be generated (i.e., a daily delta)
- Conditional training done by supervised learning on the generator, either alternating optimization steps or combining adversarial and supervised loss

During conditional training, daily deltas that are given as additional input to the generator are sampled from a Gaussian distribution estimated from real data via maximum likelihood.

## Some words on the dataset
Considering the original data provided in csv format, the values for the time series are obtained from the feature **btp_price**. 
Minimal preprocessing, including normalization in the range [-1,1], is done inside `btp_dataset.py`. The resulting dataset has 173 sequences of length 96, for an overall tensor shape of (173 x 96 x 1).
If you use a dataset that is not compatible with this preprocessing, you can just write your own loader.

## Project structure
The files and directories composing the project are:
- `main.py`: runs the training. It can save the model checkpoints and images of generated time series, and features visualizations (loss, gradients) via tensorboard. Run `python main.py -h` to see all the options.
- `generate_dataset.py`: generates a fake dataset using a trained generator. The path of the generator checkpoint and of the output \*.npy file for the dataset must be passed as options. Optionally, the path of a file containing daily deltas (one per line) for conditioning the time series generation can be provided.
- `finetune_model.py`: uses pure supervised training for finetuning a trained generator. *Discouraged*, it is generally better to train in supervised and unsupervised way jointly. 
- `models/`: directory containing the model architecture for both discriminator and generator.
- `utils.py`: contains some utility functions. It also contains a `DatasetGenerator` class that is used for fake dataset generation.
- `main_cgan.py`: runs training with standard conditional GANs. Cannot produce nice results, but it is kept for reference.

By default, during training, model weights are saved into the `checkpoints/` directory, snapshots of generated series into `images/` and tensorboard logs into `log/`.

Use:
```
tensorboard --logdir log
```
from inside the project directory to run tensoboard on the default port (6006).

## Examples
Run training with recurrent generator and convolutional discriminator, conditioning generator on deltas and alternating adversarial and supervised optimization:
```
python main.py --dataset_path some_dataset.csv --delta_condition --gen_type lstm --dis_type cnn --alternate --run_tag cnn_dis_lstm_gen_alternte_my_first_trial
```

Generate fake dataset `prova.npy` using deltas contained in `delta_trial.txt` and model trained for 70 epochs:
```
python generate_dataset.py --delta_path delta_trial.txt --checkpoint_path checkpoints/cnn_conditioned_alternate1_netG_epoch_70.pth --output_path prova.npy
```
Finetune checkpoint of generator with supervised training:
```
python finetune_model.py --checkpoint checkpoints/cnn_dis_lstm_gen_noalt_new_netG_epoch_39.pth --output_path finetuned.pth
```

## Insights and directions for improvement
- As reported in several works in sequence generation using GANs, recurrent discriminators are usually less stable than convolutional discriminators. Thus, I recommend the convolution-based one.
- I did not perform extensive search over hyperparameters and training procedures, being qualitative evaluation the only one easily possible. If a target task is configured (e.g., learning a policy), intuitions and quantitative evaluations can be obtained and used for selecting the best model. 
- There is a bit of a tradeoff between performance on realistic generation and error with respect to input delta. If having mild precision on the delta is not a problem for the final task, its error can be ignored; if one wants to reduce the error on the deltas as much as possible, it is possible either to weight the supervised objective more or to use the supervised fine tuning.
- The training is sometimes prone to mode collapse: the current implementation could benefit from the use of recent GAN variations such as [Wasserstein GANs](https://arxiv.org/abs/1704.00028). It would be sufficient to just change the adversarial part of the training.
- The [standard way](https://arxiv.org/abs/1411.1784) to inject conditions in GANs cannot work without concern for the problem of generation conditioned by deltas: as I observed in some indepedent experiments, causal convolution-based neural networks are able to easily solve the problem of detecting the delta of a given sequence. Therefore, a discriminator that receives the delta as input can easily distinguish between real sequences with correct deltas and fake sequences with incorrect deltas.
