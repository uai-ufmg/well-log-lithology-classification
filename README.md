# well-log-lithology-classification

Repository with the code for the experiments of the paper **A Benchmark for Lithology Classification Based on Well Log Data**

# Execution

To execute the lithology benchmark program, follow these bash commands:

```
cd Frente2-Benchmark_Litologia
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
| --run            | Number of the code execution (useful for testing multiple different configs).   | 1              | Positive integer             |

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

Deep models can be configured in their own yml file, in the format `{model}.yml`. The Multi-layered perceptron was implemented with scikit-learn MLPClassifier, and its parameters follow its documentation [MLP](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html). Other Deep Learning models (currently **BiLSTM, BiGRU, and CNN**) have common and specific hyperparameter, as follows:

| Common parameters | Description                                                  | Possible Values               |
|-------------------|--------------------------------------------------------------|-------------------------------|
| lr                | Learning rate for optimization.                              | Floating point number         |
| weight_decay      | Weight decay (L2 penalty) for regularization.                | Floating point number         |
| batch_size        | Batch size for training the model.                           | Integer greater than 0        |
| epochs            | Number of epochs for training.                               | Integer greater than 0        |

The following are the specific parameters for the Recurrent Networks (BiLSTM and BiGRU):

| RNN parameters    | Description                                                  | Possible Values               |
|-------------------|--------------------------------------------------------------|-------------------------------|
| hidden_size       | Number of units in the hidden state of the RNN.             | Integer greater than 0       |
| num_layers        | Number of recurrent layers.                                  | Integer greater than 0       |
| batch_first       | If true, input and output tensors are provided as (batch_size, sequence_length, feature_dim). | True, False |
| dropout           | Dropout probability, applied to the input and recurrent layers. | Floating point number      |
| bidirectional     | If true, RNNs are bidirectional.                            | True, False                  |

The following are the specific parameters for the Convolutional Network (CNN):

| CNN parameters    | Description                                                  | Possible Values               |
|-------------------|--------------------------------------------------------------|-------------------------------|
| num_logs          | Number of logs to be printed during training.                | Integer greater than 0       |

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
| Metrics| A, F1, MCC, wP|
| Models| XgBoost, MLP, BiGRU, CNN|

## Run example

This section is a sample of the output of our benchmark. As XgBoost was the baseline, this will demonstrate how the program outputs, running with the FORCE 2020 dataset.

Using the following command:

    $python3 benchmark.py --model xgb --dataset force  --run 1


the program will output the following files:

#### results/xgb_force_1:

Consists in the results of all metrics used to each fold, as the default training strategy is 5-fold cross validation.

```
FOLD 1
Accuracy: 0.7521060611866197
Weighted Accuracy: 0.43768505453534845
MCC: 0.5276648686490152
Precision: 0.5780822814294108
Weighted Precision: 0.6908293726931463
Recall: 0.43768505453534845
Weighted Recall: 0.7521060611866197
F1-Score: 0.45217145808192005
Weighted F1-Score: 0.6961306615449118

FOLD 2
Accuracy: 0.7245828074603843
Weighted Accuracy: 0.34601576240255755
MCC: 0.4918392722087577
Precision: 0.4433642179411221
Weighted Precision: 0.650603933620286
Recall: 0.34601576240255755
Weighted Recall: 0.7245828074603843
F1-Score: 0.3643292385413993
Weighted F1-Score: 0.6685090987351411

FOLD 3
Accuracy: 0.7301236457064052
Weighted Accuracy: 0.3016504591203111
MCC: 0.49266254182745317
Precision: 0.4413012231305751
Weighted Precision: 0.6772869581114742
Recall: 0.3016504591203111
Weighted Recall: 0.7301236457064052
F1-Score: 0.30866079654879414
Weighted F1-Score: 0.674751992497194

FOLD 4
Accuracy: 0.6743926615797492
Weighted Accuracy: 0.30834495473096113
MCC: 0.44272613458327964
Precision: 0.29443032044097206
Weighted Precision: 0.6200932499722089
Recall: 0.22425087616797176
Weighted Recall: 0.6743926615797492
F1-Score: 0.239633925937933
Weighted F1-Score: 0.6317931307655439

FOLD 5
Accuracy: 0.699553824224318
Weighted Accuracy: 0.35952303383630696
MCC: 0.4648916232455772
Precision: 0.4746699047870535
Weighted Precision: 0.6409600459721931
Recall: 0.35952303383630696
Weighted Recall: 0.699553824224318
F1-Score: 0.36190986127565594
Weighted F1-Score: 0.6470138832408625
```

#### plots/15_9_13_xgb.png:

This image file demonstrates the prediction of the trained model, comparing it with the ground truth, which consists in the lithology stracted from the well for each depth. In this case, *15_9_13* represents the nomination of the well used in this plot. 

<img src=Frente2_Benchmark_Litologia/plots/15_9_13_xgb.png>

The rightmost column demonstrate the predictions that our XgBoost model has done over the four well logs: *Gamma Ray, Density, Neutron, Compressional Wave.*


# Adding Models/Datasets/Metrics
    
## Models

There are a few steps that should be followed when introducing a model:

1. If it is a **Deep Learning model**, it is important to add a model file into the models folder (`core/models/`).
    - The model class should inherit from the model template in `model_template.py` to get the fit and test functions.
    - The output from the model should be a one hot vector of probabilities, i.e., for each depth of the sequence input sequence, the model should output a vector of *k* probabilities, where *k* is the number of lithology classes.

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
