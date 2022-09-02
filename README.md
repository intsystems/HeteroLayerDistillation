# Knowledge distillation on heterogeneous models

**Authors**: M. Gorpinich, O. Bakhteev, V. Strijov

This paper investigates the deep learning knowledge distillation problem. Knowledge distillation is a model parameter optimization problem that allows transferring information contained in the model with high complexity, called teacher, to the simpler one, called student. In this paper we propose a cross-layer distillation method that can be applied to significantly heterogeneous models. The variational inference is applied to derive the loss function for metaparameter optimization. Metaparameters are the coefficients of the losses between each pair of layers. The proposed approach is evaluated in the computational experiment on the CIFAR-10 dataset.

## Requirements

```
Python >= 3.5.5
torch == 1.7.1
numpy == 1.18.5
tqdm == 4.59.0
matplotlib == 3.3.2
hyperopt == 0.2.5
scipy == 1.5.2
Pillow == 7.2.0
```

[requirements.txt](https://github.com/Intelligent-Systems-Phystech/HeteroLayerDistillation/blob/master/requirements.txt)

## Main experiments:

[ResNet distillation](https://github.com/Intelligent-Systems-Phystech/HeteroLayerDistillation/blob/master/notebooks/mutual_information_distillation_demo_oleg.ipynb)

[Distillation of two NNs with same structure](https://github.com/Intelligent-Systems-Phystech/HeteroLayerDistillation/blob/master/notebooks/mutual_information_distillation_same_structure.ipynb)

