# network_theseus

## Random-guide optimization prior, version 2

Start with [OPTIMIZATION_PRIOR.md](OPTIMIZATION_PRIOR.md) and `run_prior.py` for these experiments. The defaults are cumulative progressive alignment, full selected updates, and diagnostic-only rank measurements. Both CKA and differentiable KNN use label-free held-out checkpoint selection.

The remaining README commands describe legacy Theseus runs, not the optimization-prior presets.

This is the code repo for [Network of Theseus](https://arxiv.org/pdf/2512.04198). 

## Dependencies
All dependencies should be in `requirements.txt` so you should be able to install via:

```
pip install -r requirements.txt
```

after making your preferred python environment (conda, miniconda, venv, etc...)

## Setup

The repo is fairly self-explanatory! Here is the structured breakdown of the most useful files:

```
network_theseus/
├── theseus.py : Main runner for adding models and setting arguments.
├── args.py : A list of argparse arguments that the code uses
├── layer_replace.py : The main layer replacement modules. Includes all replacement from the paper. 
├── finetune.py : where standard training happens -- has training functions for ImageNet classification, language modeling, etc.
├── dataloader.py : a bunch of dataloaders from ImageNet, CIFAR, Wikitext-103, etc.
├── resnet_utils.py : Utility functions for aligning ResNets
├── gpt2_utils.py : Utility functions for aligning across different GPT-2 sizes.
├── layer_registry.py : Dictionary containing mapping of guide network layers to target network layers. Only matters for standard replacement.
├── modules.py : Alignment modules for replacement. Only matters for standard replacement. 
└── rep_sims.py : A list of different representational similarity measures. 
```

## Running ResNet Alignment

To understand the ResNet alignment, take a look at:

* `layer_replace.py --> progressive_align_resnets` : where progressive alignment of ResNet happens
* `layer_replace.py --> match_all_groups` : the alignment trainer
* `resnet_utils.py` : contains all functions for matching across ResNets.

For this set up, I used ResNets from `torchvision`. If you want to go beyond the standard ResNets e.g. ResNet-8, ResNet-20 or ResNet-32, then you will need to write that! You can modify the extraction code in `resnet_utils.py` pretty easily afterward. 

Sample command:

```
python theseus.py --exp_name rn50_rn18_progressive_replace --indices 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 --model rn50 --repdist CKA --replace progressive_align_resnet --rep_batch 512 --ft_batch 256 --ft_optimizer adamw --ft_lr 1e-3 --scheduler --full_rank --progressive_lrs 1e-4 --progressive_epochs 10 --rep_scheduler --ft_epochs 100 --ft_wd 1e-4 --label_smoothing 0.3
```

This runs progressive replacement from ResNet-50 to ResNet-18 using CKA. There are 19 convolutional layers in ResNet-18 listed by the `indices` argument. `--replace` indicates which replacement style this is using! Each progressive stage is tuned for 10 epochs with a learning rate of `1e-4`. 

After running alignment, you will automatically do standard training using `finetune.py`. If you want to just run `finetune.py`, you can pass `--reload_ft` to reload the aligned weights.