=====
Usage
=====

Step 1 - Write specification file
#################################

The specification file is essentially a YAML file but with extension `.nni.yml`

There are 4 parts (sections) in the configuration file.

******************
Datasource Section
******************

This is where you will specify the (python) callable that `sknni` would invoking to the training and
test dataset.

The callable should return 2 values where each value is a `tuple` of two items. The first tuple
consists of training data `(X_train, y_train)` and the second tuple consists of test data `(X_test, y_test)`.

An example callable would look like this::

    import numpy as np

    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split

    class ACustomDataSource(object):
        def __init__(self):
            pass

        def __call__(self, test_size:float=0.25):
            digits = load_digits()
            X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, random_state=99, test_size=test_size)

            return (X_train, y_train), (X_test, y_test)

In the above example, the callable generates the train and test dataset. The callable can even have paramaters for e.g. in this
example you could optionally pass the fraction of data to be used for testing purposes.

Now let's see how you would specify in the specification file.

.. code-block:: yaml

    # Datasource is how you specify which callable
    # sknni will invoke to get the data
    dataSource:
        reader: yourmodule.ACustomDataSource
        params:
            test_size: 0.30

Make sure that during the exeuction of the experiment your datasource (i.e. in this case `yourmodule.ACustomDataSource`)
is available in the PYTHONPATH.

Here is an additional example showing the usage of an built-in datasource reader

.. code-block:: yaml

    dataSource:
        reader: sknni.datasource.NpzClassificationSource
        params:
            dir_path: /Users/ksachdeva/Desktop/Dev/myoss/scikit-nni/examples/data/multiclass-classification


**************************
Pipline definition Section
**************************

Below is the example of the section. You simply specify the list of steps of your typical scikit-learn Pipeline.

Note - The sequence of steps is very important.

What you **MUST** ensure is that the full qualified name of your scikit-learn preprocessors, transformers and
estimators is correctly specified. `sknni` uses reflection and introspection to create the instances so if you have a
typo in the names and/or they are not available in your PYTHONPATH you will get an error at experiment execution time.

.. code-block:: yaml

    sklearnPipeline:
        name: normalizer_svc
        steps:
            normalizer:
                type: sklearn.preprocessing.Normalizer
            svc:
                type: sklearn.svm.SVC

In above example, there are 2 steps. The first step is to normalize the data and the second step is train a classifier using Support
Vector Machine.

********************
Search Space Section
********************

This section corresponds to the search space for your hyperparameters. When you ```nnictrl``` this is typically
specified in search-space.json file.

Here are the important things to note about this section -

- The syntax is the same (except we are using YAML here instead of JSON) for specifiying parameter types and ranges.
- You **MUST** specifiy the parameters corresponding to the step in your scikit pipeline.
- You **MUST** use the names of the parameters that are same as the ones accepted by scikit-learn components (i.e. preprocessors, estimators etc).

Below is an example of this section.

.. code-block:: yaml

    nniConfigSearchSpace:
        - normalizer:
            norm:
                _type: choice
                _value: [l2, l1]
        - svc:
            C:
                _type: uniform
                _value: [0.1,0.0]
            kernel:
                _type: choice
                _value: [linear,rbf,poly,sigmoid]
            degree:
                _type: choice
                _value: [1,2,3,4]
            gamma:
                _type: uniform
                _value: [0.01,0.1]
            coef0:
                _type: uniform
                _value: [0.01,0.1]

Note that `sklearn.svm.SVC` takes C, kernel, degree, gamman and coef0 is the paramaters and hence we have used here
the same names (keys) in the search space specification. You can add as many or as little parameters to search for.

******************
NNI Config Section
******************

This is the simplest of all sections as there is nothing new here from sknni perspective. You just copy-paste
here your NNI's config.yaml here. You do not have to specify `codedir` and `command` field in the `trial` subsection as
this is added by the sknni in the generated configuration files.

Here is an example.


.. code-block:: yaml

    # This is exactly same as the one that of NNI
    # except that you do not have to specify the command
    # and code fields. They are automatically added by the sknni generator
    nniConfig:
        authorName: default
        experimentName: example_sklearn-classification
        trialConcurrency: 1
        maxExecDuration: 1h
        maxTrialNum: 100
        trainingServicePlatform: local
        useAnnotation: false
        tuner:
            builtinTunerName: TPE
            classArgs:
                optimize_mode: maximize
        trial:
            gpuNum: 0

You can look at the various examples in the repository to learn how to define your own specification file.


Step 2 - Generate your experiment
#################################

.. code-block:: bash

    sknni generate-experiment --spec example/basic_svc.nni.yml --output-dir experiments


Above command will create a directory experiments/svc-classification will following files

    - The original specification file i.e. basic_svc.nni.yml (used during experiment run as well)
    - Generated Microsoft NNI's config.yml
    - Generated Microsoft NNI's search-space.json


Step 3 - Run your experiment
#################################

This is same as running `nnitctl`

.. code-block:: bash

    nnictl create --config experiments/svc-classification/config.yml


