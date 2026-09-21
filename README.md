SPS targets long-tailed recognition, where head classes have many samples and tail classes are underrepresented. The method transfers knowledge from head classes to tail classes, but does it through a purified low-rank prototype space to reduce noisy or semantically mismatched transfer.
The paper uses two main losses:

- **SemLRF**: synthesizes stronger tail-class prototypes by fusing semantically related head-class knowledge inside a low-rank subspace.
- **ProtoCL**: improves global prototype separation with a contrastive objective.

- ## This Bundle

This folder contains the code, bundled CIFAR-100 data, and the selected checkpoint for the CIFAR-100-LT imbalance rate 0.01 experiment.

- `code/`: training and evaluation code.
- `code/Trainer.py`: SPS trainer implementation based on `Trainer1105True.py`.
- `code/main.py`: training entry with the CIFAR-100-LT IR=100 defaults.
- `data/cifar-100-python/`: bundled CIFAR-100 files (`train`, `test`, `meta`).
- `checkpoint/ckpt.best.pth.tar`: selected checkpoint.
- `checkpoint/log.txt`: training log for the selected checkpoint.
-  The checkpoint ckpt.best.pth.tar can be find in 
 https://pan.baidu.com/s/1R5auSd48LcHz7BQmY1O63w?pwd=1b89 提取码: 1b89 
- ## Run

From the `code/` directory, `main.py` resolves the dataset path to the bundled `../data` directory.

Train with the packaged defaults:

```bash
cd code
python main.py
```

Evaluate the selected checkpoint:

```bash
cd code
python test.py --dataset cifar100 -a resnet34 --imbanlance_rate 0.01 \
  --resume ../checkpoint/ckpt.best.pth.tar -b 256 -j 4
```
