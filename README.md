# Lithology classification on well log data - Benchmark

Repository with the code for the experiments of the paper **Lithology classification based on well log data: A benchmark for machine learning models**

# Execution

To execute the lithology benchmark, follow these bash commands:

```
cd well-log-lithology-classification
python3 benchmark.py --model <model> --dataset <dataset> --seq_size <n> --run <x>
```

Replace `<model>` and `<dataset>` with the desired model and dataset options respectively. Additionally, <n> represents the sequence size, which must be an integer number (default is 50 for deep learning models), and `<x>` indicates the number of the execution. 

Available Models:
- XGBoost (xgb)
- Random Forest (rf)
- Naive Bayes (nb)
- MLP (mlp)
- BiLSTM (bilstm)
- BiGRU (bigru)
- CNN (cnn)
- ResNet (resnet)
- AdaBoost-Transformer (transformer)
- Hybrid Noise Label Filtering and Correction Framework (HNFCL)

Available Datasets:
- Force (force)
- Geolink (geolink)
    
All execution parameters:
    
| Command          | Description                                                                     | Default Value  | Possible Values              |
|------------------|---------------------------------------------------------------------------------|----------------|------------------------------|
| --model          | Name of the desired model.                                                      | 'xgb'          | Any implemented model name   |
| --dataset        | Desired dataset for training and testing model.                                 | 'force'        | Any implemented dataset name |
| --seq_size       | Input sequence size (only used for deep learning models).                       | 50             | Positive integer             |
| --weighted       | Allow weighted training or not. MLP does not support.                           | False          | True, False                  |
| --save_model     | Wether to save or not the trained model after training.                         | False          | True, False                  |
| --save_dir       | Directory used for saving the model.                                            | trained_models | Any existing path            |
| --output_dir     | Directory used for saving the model evaluation results.                         | results        | Any existing path            |
| --config_dir     | Directory used for loading the config yml files.                                | configs        | Any existing path            |
| --verbose        | Print the evolution of steps during execution                                   | False          | True, False                  |
| --run            | Number of the code execution (useful for multiple runs using the same configs). | 1              | Positive integer             |

For example, to run the benchmark with XGBoost on the Force dataset with a sequence size of 1 for the third execution, the command would be:

python3 benchmark.py --model xgb --dataset force --seq_size 1

# Configs

In 'Frente2_Benchmark_Litologia' directory, the 'config' directory has '.yml' files that hold the necessary parameters to run the code properly.

### `data.yml`: 

#### This file contains necessary information to perform the methods.

| Field          | Description                                                    | Possible Values       | Default Value |
|----------------|----------------------------------------------------------------|-----------------------|---------------|
| split_form     | Method for selecting data split.                               | 'kfold'               | 'kfold'       |
| scaling_method | Method for scaling the data.                                   | 'standard', 'minmax'  | 'standard'    |
| n_splits       | Number of splits for cross-validation (kfold).                 | Positive integer      | 5             |
| shuffle        | Parameter to determine if shuffling of data is desired.        | True, False           | True          |
| test_size      | Ratio of test data size. Must be within [0, 1] interval.       | Float in [0, 1]       | 0.1           |
| val_size       | Ratio of validation data size. Must be within [0, 1] interval. | Float in [0, 1]       | 0.2           |

### `<dataset>.yml` files:

There are three YAML configuration files named `<dataset>`.yml, where `<dataset>` can be replaced with 'force' or 'geolink'. These files require modification in the datadir parameter to specify the directory where the dataset is stored.

Example:

* **force.yml**: Configuration file for the 'force' dataset.
* **geolink.yml**: Configuration file for the 'geolink' dataset.

In each of these YAML files, locate the datadir parameter and set its value to the directory path where the respective dataset is stored.

### `<model>.yml` files:

For the `<model>` files, there are two distinct classes: shallow and deep models.

#### Shallow Models:

The shallow methods are primarily implemented using the [Scikit Learn](https://scikit-learn.org/stable/) package, except for XGBoost, which has its own package. The parameters of the following models need to be modified according to their respective documentation:

- [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html)
- [XGBoost](https://xgboost.readthedocs.io/en/stable/parameter.html)


#### Deep Models:

Deep models can be configured in their own yml file, in the format `{model}.yml`. The Multi-layered perceptron was implemented with scikit-learn MLPClassifier, and its parameters follow its documentation [MLP](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html). Other Deep Learning models (currently **BiLSTM, BiGRU, CNN, ResNet, and Adaboost Transformer**) have common and specific hyperparameter, as follows:

| Common parameters | Description                                                  | Possible Values               |
|-------------------|--------------------------------------------------------------|-------------------------------|
| lr                | Learning rate for optimization.                              | Floating point number         |
| weight_decay      | Weight decay (L2 penalty) for regularization.                | Floating point number         |
| batch_size        | Batch size for training the model.                           | Integer greater than 0        |
| epochs            | Number of epochs for training.                               | Integer greater than 0        |

The following are the specific parameters for the Recurrent Networks (BiLSTM and BiGRU):

| RNN parameters    | Description                                                                                   | Possible Values         |
|-------------------|-----------------------------------------------------------------------------------------------|-------------------------|
| hidden_size       | Number of units in the hidden state of the RNN.                                               | Integer greater than 0  |
| num_layers        | Number of recurrent layers.                                                                   | Integer greater than 0  |
| batch_first       | If true, input and output tensors are provided as (batch_size, sequence_length, feature_dim). | True, False             |
| dropout           | Dropout probability, applied to the input and recurrent layers.                               | Floating point number   |
| bidirectional     | If true, RNNs are bidirectional.                                                              | True, False             |

The following are the specific parameters for the Convolutional Networks (CNN and ResNet):

| CNNs parameters   | Description                                                  | Possible Values               |
|-------------------|--------------------------------------------------------------|-------------------------------|
| num_logs          | Number of logs to be printed during training.                | Integer greater than 0        |

The following are the specific parameters for the Transformer (Adaboost Transformer):

| Transformer parameters | Description                                                  | Possible Values                     |
|------------------------|--------------------------------------------------------------|-------------------------------------|
| n_classifiers          | Number of transformers in the Adaboost Framework             | Integer greater than 0              |
| hidden_dim             | Hidden Dimension in each Transformer Encoder architecture    | Integer greater than 0              |
| dropout                | Dropout parameter for each Transformer Encoder               | Floating point number from 0 to 1   |

## Methodology:

In our paper, we have justified our decision to utilize four key well logs in our approach: *Gamma Ray, Density, Neutron, and Compressional Wave*. We selected these logs because they are properties that are highly influenced by rock types and exhibit a degree of completeness, meaning they are less susceptible to missing data issues.

In employing these four logs within our *shallow models*, as previously described, we performed lithology classification on a point-to-point basis for each instance of data.

In our implementation of the deep models, we adopted a sequential approach with a configurable sequence size denoted as **s**, which can be specified via command line parameters. This sequence encompasses data from the four logs and is constructed exclusively from entries without missing values. A sequence is considered valid if no missing values are encountered between its start and end points. Consequently, our input data instance is characterized by a size of **s x 4**, reflecting the incorporation of all four logs within each sequential segment.


For evaluation, we use nine metrics:

- Accuracy
- Weighted Accuracy
- MCC
- Precision
- Weighted Precision
- Recall
- Weighted Recall
- F1-Score
- Weighted F1-Score

Although we use several metrics, in our paper we decided to use four as the most important ones: *Accuracy, MCC, F1-score, Weighted Precision.*

Overall, summarizing the benchmark proposal we have:

|          |               |
| -------- | --------------|
| Datasets | Force, Geolink|
| Logs     | GR, DTC, NPHI, RHOB |
|Sequence Lengths| 1, 50|
| Data split | 5-fold cross validation|
| Metrics| A, MCC, P, R|
| Models| Adaboost Transformer, BiGRU, XgBoost, MLP |

## Run example

This section is a sample of the output of our benchmark. This will demonstrate how the program outputs, running with the FORCE 2020 dataset.

Using the following command:

    $python3 benchmark.py --model transformer --dataset force --run 1


the program will output the following files:

#### results/transformers_force_50_1_False:

Consists in the results of all metrics used to each fold, as the default training strategy is 5-fold cross validation.

```
FOLD 1
Accuracy: 0.7409353507565337
Weighted Accuracy: 0.3620584336092069
MCC: 0.514034109523071
Precision: 0.5391668288127979
Weighted Precision: 0.6991680055690951
Recall: 0.3620584336092069
Weighted Recall: 0.7409353507565337
F1-Score: 0.39147883643536086
Weighted F1-Score: 0.7091174727974746
Training Time: 6681.516830921173

FOLD 2
Accuracy: 0.7271345387680049
Weighted Accuracy: 0.36040440813606633
MCC: 0.5080825677985306
Precision: 0.5017825866673316
Weighted Precision: 0.6785100344892573
Recall: 0.36040440813606633
Weighted Recall: 0.7271345387680049
F1-Score: 0.39095160089205566
Weighted F1-Score: 0.6931489280924452
Training Time: 6757.57267165184

FOLD 3
Accuracy: 0.7426091081593927
Weighted Accuracy: 0.29061691844567605
MCC: 0.5322054855270353
Precision: 0.34180821946877066
Weighted Precision: 0.701631019733321
Recall: 0.29061691844567605
Weighted Recall: 0.7426091081593927
F1-Score: 0.28679321069183944
Weighted F1-Score: 0.7099570248177435
Training Time: 6428.15208029747

FOLD 4
Accuracy: 0.7035448717948718
Weighted Accuracy: 0.411834177690789
MCC: 0.5003051018096201
Precision: 0.36236749750759545
Weighted Precision: 0.6706353057810158
Recall: 0.2995157655933011
Weighted Recall: 0.7035448717948718
F1-Score: 0.31932589507597037
Weighted F1-Score: 0.6764403200956371
Training Time: 6957.990426540375

FOLD 5
Accuracy: 0.7119490071095856
Weighted Accuracy: 0.3680166906151434
MCC: 0.4982074234156329
Precision: 0.47307580667626525
Weighted Precision: 0.6810864769362498
Recall: 0.3680166906151434
Weighted Recall: 0.7119490071095856
F1-Score: 0.36602751926301624
Weighted F1-Score: 0.6783419580146866
Training Time: 6520.793189525604
```

#### results/transformers_force_50_1_False.png:

This image file demonstrates the prediction of the trained model, comparing it with the ground truth, which consists in the lithology stracted from the well for each depth. In this case, *16_10_5* represents the nomination of the well used in this plot. 

<img src=Frente2_Benchmark_Litologia/results/transformer_force_50_1_False.png>

The first column shows the ground truth lithology labels for the well, while the second column shows the predictions that the Adaboost Transformer has done given the four well logs: *Gamma Ray, Density, Neutron, Compressional Wave.*


# Adding Models/Datasets/Metrics
    
## Models

There are a few steps that should be followed when introducing a model:

1. If it is a **Deep Learning model**, it is important to add a model file into the models folder (`core/models/`).
    - The model class should inherit from the model template in `model_template.py` to get the fit and test functions.
    - The output from the model should be a one hot vector of probabilities, i.e., for each depth of the sequence input sequence, the model should output a vector of k probabilities, where k is the number of lithology classes.

2. If it is a **Traditional ML model**, it is important that it has the fit and predict functions.
    
3. Create a yml file in the `configs` folder for the model hyperparameters.

4. Adapt the code inside `ConfigArgs` class in the `configs/config.py` file to be able to open your model's yml file.
    - **Important Note**: Remember to change the second conditional code in `parse_args` to give the appropriate `input_format` to your model.
    
5. Add the model instantiation to the `select_model` function inside the `init` file in `core/models/`.
    - Remember to import the model, either from your model file or from an outside package.
    
6. Add the model saving code to the `save_model` function inside the `init` file in `core/models/`.

Now you can evaluate your new model!    

## Datasets
    
1. Add a dataset file (in the format `data_{name}.py`) into the data folder (`core/data/`).
    - This file should contain a class that inherits from the `Data` class inside `data.py` file.
    - Your class should overwrite `open_data` function in `Data` class in order for it to return the dataset as a pandas DataFrame with the standard log names.

2. Create a yml file in the `configs` folder for the dataset's hyperparameters in the format `{name}.yml` based on the other datasets' files.

3. Add the dataset class instantiation to the `open_data` function inside the `init` file in `core/data/`.
    - Remember to import the dataset class from the dataset file.

Now you can evaluate models with your new dataset!


## Metrics
    
1. Import the metric function to `benchmark.py`.

2. Add another `f.write` line of code at the end of `evaluate` function.
    - Call your metric function inside the braces.
    - **Important Note**: Your metric function should receive `y_predicted` and `y_true`, where the first is the predicted values by the model and the second is the true label.

Now you can evaluate models with your new metric!
