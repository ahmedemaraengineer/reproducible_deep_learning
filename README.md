# Reproducible Deep Learning Project
## Sound Classification Model Development

building a complex deep learning model that is full of interacting 
design decisions, data engineering, parameter tweaking and experimentation.
Having access to powerful tools for versioning storing, and analyzing every step 
of the process (MLOps).

<img width="482" alt="image" src="https://user-images.githubusercontent.com/71132701/163660752-6eb110d3-27c0-46cf-851b-dcf7488440a1.png">


[**Notebook**](/Initial_Training.ipynb)


Starting from a jupyter notebook that handles the steps of data engineering, 
data preprocessing, training the model step by step, then I have tried to make this 
project as reproducible as possible. 


### 1- Configuration With Hydra

```bash
pip install hydra-core
```

move all configuration for the training script inside an external configuration file.
This simple step dramatically simplifies development and reproducibility, 
by providing a single entry point for most hyper-parameters of the model 
using a .yaml file that carries all the configurations.

```yaml
data:
    # All parameters related to the dataset
model:
    # All parameters related to the model
    optimizer:
        # Subset of parameters related to the optimizer
trainer:
    # All parameters to be passed at the Trainer object
```

### 2- Data Versioning Using Data Version Control(DVC)

- [ ] Adding data versioning with DVC.
- [ ] Using multiple remotes for a DVC repository.
- [ ] Downloading files from a DVC repository.

```bash
pip install dvc
```

### 3- Dockerization 

Isolating the execution in a container by isolating all the libraries and dependencies 
inside a container that works as an operating system that handles all the requirements 
for the project to be more reproducible in the future.

### 4- Weights and Biases

Then finally I have used weights and biases (wandb) to keep track of all the logs 
that I want to save and also do hyperparamaters tuning and monitoring through
an isolated server with full visualizations that helped me visualize my training process
and choose the best hyperparameters. 

```bash
pip install wandb
```

(wandb) uses something called ("sweep") to do hyperparameter tuning 
using a .yaml file to take the instructions or the range of different hyperparameters that the user 
want to experiment.

```yaml
program: train.py
method: bayes
metric:
  name: val_acc
  goal: maximize
parameters:
  sample_rate: [2000, 4000, 8000]
  # Define other hyper-parameters here...
```


