# DSDS-H
We propose a hierarchical multimodal NMT method based on dynamic semantic selection (DSDS-H). The method jointly exploits region-level and global image features, and introduces a Token-aware Top-K feature selection Mechanism together with a hierarchical fusion strategy to achieve progressive coarse-to-fine integration.


# Our dependency

* PyTorch version == 1.8.1
* Python version == 3.7.16
* timm version == 0.4.12
* vizseq version == 0.1.15
* nltk verison == 3.6.4
* sacrebleu version == 1.5.1

# Install fairseq
You need to load the complete fairseq framework and place the code into the corresponding files before it can run! ! !

```bash
cd fairseq_mmt
pip install --editable ./
```

# Multi30k data & Flickr30k entities
Multi30k data from [here](https://github.com/multi30k/dataset) and [here](https://www.statmt.org/wmt17/multimodal-task.html)  
flickr30k entities data from [here](https://github.com/BryanPlummer/flickr30k_entities)  
Here, We get multi30k text data from [Revisit-MMT](https://github.com/LividWo/Revisit-MMT)
```bash
cd fairseq_mmt
git clone https://github.com/BryanPlummer/flickr30k_entities.git
cd flickr30k_entities
unzip annotations.zip

# download data and create a directory anywhere
flickr30k
├─ flickr30k-images
├─ test2017-images
├─ test_2016_flickr.txt
├─ test_2017_flickr.txt
├─ test_2017_mscoco.txt
├─ test_2018_flickr.txt
├─ testcoco-images
├─ train.txt
└─ val.txt
```

# Extract image feature
#### 1. Vision Transformer 

  python extract_grid_features.py --dataset train --path ./flickr30k --data_path data-bin
  ```
  script parameters:
  - ```dataset```: choices=['train', 'val', 'test2016', 'test2017', 'testcoco']
  - ```path```:    '/path/to/your/flickr30k'
  ```
#### 2. Faster-R-CNN

  python extract_grid_features.py --dataset train --path ./flickr30k --data_path data-bin
  ```
  script parameters:
  - ```dataset```: choices=['train', 'val', 'test2016', 'test2017', 'testcoco']
  - ```path```:    '/path/to/your/flickr30k'
  ```

# Train and Test
#### 1. Train

    fairseq-train data-bin/en-de \
      --arch my_transformer \
      --task my_translation \
      --valid-subset valid,test2016 \
      --share-decoder-input-output-embed \
      --optimizer adam --adam-betas '(0.9, 0.98)' \
      --clip-norm 0.1 \
      --lr 0.001 \
      --lr-scheduler inverse_sqrt \
      --warmup-init-lr 1e-07 \
      --min-lr 1e-09 \
      --warmup-updates 4000 \
      --max-update 4700 \
      --max-tokens 4096 \
      --dropout 0.3 \
      --attention-dropout 0.1 \
      --activation-dropout 0.1\
      --weight-decay 0.0001 \
      --criterion label_smoothed_cross_entropy \
      --label-smoothing 0.2 \
      --update-freq 4 \
      --eval-bleu \
      --eval-bleu-args '{"beam": 5, "lenpen": 1.2, "max_len_a": 1.2, "max_len_b": 10}' \
      --eval-bleu-detok moses \
      --eval-bleu-remove-bpe \
      --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
      --keep-last-epochs 10 \
      --eval-bleu-print-samples \
      --patience 15 \
      --no-progress-bar \
      --log-format simple \
      --tensorboard-logdir results/en-de/logdir \
      --save-dir checkpoints/your_path \
      --fp16 
    
#### 2. Test

      fairseq-generate data-bin/en-de \
        --path checkpoints/your_model/checkpoint_best.pt \
        --task my_translation \
        --num-workers 4 \
        --batch-size 128 \
        --beam 5 --lenpen 1.2 --max-len-a 1.2 --max-len-b 10 \
        --gen-subset dataset \
        --remove-bpe \
        --scoring sacrebleu \
        --output results/your.out


# Visualization
```
  python draw_high_layer_cross_modal.py
```
