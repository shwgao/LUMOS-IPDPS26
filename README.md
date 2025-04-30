## Installation

All the basic python packages are listed in the requirements.txt, you can install them from the following commands.
```
conda create -n LOXIA python=3.10
conda activate LOXIA
pip install -r requirements.txt
```

## About directory [applications](./applications)

It includes all the implementation of all the application parts in which some specific components need to be defined.
- dataset.py: It defines how to process the dataset, will return training and validation data loaders.
- launch.py: In this file, the loss function and metric (a specific number which would be used to determine the best model) should be defined.
- model.py: It defines your model that has to inherit `BaseModel` from [base_layers.py](./base_layers.py). For some more complicated models, `build_dependency_graph` may need to be designed as necessary.
- settings.json: All the tunable hyparameters are included in this file.

## About [base_layers.py](./base_layers.py)
Base layers(linear, convolution and base models) are implemented. The algorithm part is mainly comes from paper: [Learning sparse neural networks through $L_0$ regularization](https://arxiv.org/abs/1712.01312).

## About datasets resources
- [google drive]()
- [Cosmoflow](https://arxiv.org/pdf/2007.12856.pdf)
- [sciml-bench](https://github.com/stfc-sciml/sciml-bench/tree/master)
- Some others could be found online.

## Other important files
- train.py: Training.
- inference.py: Validation.
- inference_one_round.py: Test all the applications at one time.
- inference_one_round_sar.py: Test all the model and save the onnx model file.
- utils.py: Other useful functions.

## What's under processing
We are going to include applications(ogbg-ppa and ogbg-molhiv) using graph models(gin and gcn).

How to train these models: using [main_pyg_l0.py](./main_pyg_l0.py).
Other related files locate at directory [test_graph](./applications/test_graph).

What's the next steps:
- Get trained graph models for these two applications.
- Prune the trained models and test the performance.
- Make code structure of graph models same as other applications.

## The steps to get a pruned model
- Train the model by tuning the hyperparameters in `setting.json`.
    * During training, we create masks for kernels and neurons, and then train these masks on the fly.
    * Every model has a different number of layers. Each layer is created with a mask vector that has the same dimension as the number of input (or output) channels of weights.
    * The most important hyperparameters are $N$ (which can be tuned by `initial_budget` and `final_budget`), $temperature$ (which can be tuned by `initial_temp` and `final_temp`), and $lr$. 

- After training and obtaining the ideal model, the next step is to prune out the kernels and neurons which are zero. To prune a model, you only need to call the class method `prune_model`. However, for some complicated models, you may have to design the process of pruning as follows:
    * If the mask value corresponding to a neuron or kernel is 0, then this neuron or kernel needs to be removed from this layer.
    * Every model has to inherit from the base model in which the `build_dependency_graph` class method is used to build `in_mask` and `out_mask` for every layer to indicate which kernel or neuron needs to be removed. For more complicated models, this class method may need to be rewritten as necessary.
    * The basic idea to rewrite `build_dependency_graph` is that you have to specify every layer's `in_mask` and `out_mask` based on the relationship of dependent layers.

- After calling `prune_model` successfully, you will get the post-pruned model with fewer parameters according to the prune rate. You can then test or save the post-pruned model.
    


