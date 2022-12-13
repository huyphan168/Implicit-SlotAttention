<img src="./fixpoint-diagram.jpg" width="600px" style="border: 1px solid #ccc"></img>
# Slot-Attention-Iterative-Refinement
Unofficial Implementation of papers [Object Representations as Fixed Points: Training Iterative Refinement Algorithms with Implicit Differentiation](https://arxiv.org/pdf/2207.00787.pdf), [SLATE](https://arxiv.org/pdf/2110.11405.pdf), and
[Object-Centric Learning with Slot Attention](https://arxiv.org/abs/2006.15055) by Phan Nhat Huy

## To do:
- [] Implement Slot Attention
- [] Implement Implicit Differentiation for Slot Attention
- [] Running experiments with CLEVR dataset
- [] Running experiments with Reasoning dataset

## Installation
Current version of implementation has tested with pytorch 2.0 but it should work with pytorch > 1.0.0. To install all dependencies, just run.
```
python3 -r requirements.txt
```

## Usage
This implementation is fairly simple to use. With a user specificed configuration file, `main.py` will do anything else. The configuration file consists of 3 parts: 
Model, Training and Testing config. 

Combining all parts we have following general configuration file.
```yaml
model:
    model_type: implicit_slate #slate, vanilla_slot
    encoder_type: dvae
    num_slots: 4
    num_heads: 4
    num_layers: 2
    feat_size: 256
    dropout: 0.1
    implicit_diff: True
    max_iter_fwd: 10
training:
    data_type: clevr
    batch_size: 64
    num_workers: 4
    num_epochs: 100
    lr: 0.0001
    weight_decay: 0.0001
    log_interval: 100
    save_interval: 1000
    save_dir: ./checkpoints
    device: cuda
testing:
    data_type: clevr
    batch_size: 64
    num_workers: 4   
```
To reconstruct the results in the paper of Implicit Slot Attention with CLEVR dataset, just run
```
python3 main.py --cfg configs/im_slate_clevr.yaml
```
## Citation
```bibtex
@article{Locatello2020ObjectCentricLW,
  title={Object-Centric Learning with Slot Attention},
  author={Francesco Locatello and Dirk Weissenborn and Thomas Unterthiner and Aravindh Mahendran and Georg Heigold and Jakob Uszkoreit and Alexey Dosovitskiy and Thomas Kipf},
  journal={ArXiv},
  year={2020},
  volume={abs/2006.15055}
}
@inproceedings{
      singh2022illiterate,
      title={Illiterate DALL-E Learns to Compose},
      author={Gautam Singh and Fei Deng and Sungjin Ahn},
      booktitle={International Conference on Learning Representations},
      year={2022},
      url={https://openreview.net/forum?id=h0OYV0We3oh}
}
@article{Chang2022ObjectRA,
  title={Object Representations as Fixed Points: Training Iterative Refinement Algorithms with Implicit Differentiation},
  author={Michael Chang and Thomas L. Griffiths and Sergey Levine},
  journal={ArXiv},
  year={2022},
  volume={abs/2207.00787}
}
```
