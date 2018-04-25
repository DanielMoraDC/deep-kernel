# New hybrid architectures for deep learning

This work is the implementation for my Thesis in the Master of Articial Intelligence. It combines neural networks and kernel methods to build hybrid architectures. These networks are tested on structured and non-structured data and we provide an starting point for new regularization techniques for these architectures.

Three different novel types of learning have been provided for these networks. For simplicity, they have been placed in separate branches of this repository. These are:

- [Cyclic Layerwise Training](https://github.com/DaniUPC/deep-kernel/tree/layerwise-cycling).
- [Incremental Layerwise Training](https://github.com/DaniUPC/deep-kernel). Master branch.
- [Alternate Layerwise Training](https://github.com/DaniUPC/deep-kernel/tree/layerwise-alternate).

Details about these methods can be found in the [report of the Thesis](https://github.com/DaniUPC/deep-kernel/blob/master/Thesis.pdf).

## Cloning the repo

After cloning the repo, in order to initialize the submodules, you must type:

```bash
git submodule update --init
```

## Reproducibility

This code works on Python 3 and uses Tensorflow. The dependencies for the project can be imported into a virtual environment easily.

First, make sure *virtualenv * is installed:

```bash
sudo apt-get install python3-pip
sudo pip3 install virtualenv 
```

Then we can create an environment with:

```bash
virtualenv env --python=python3 
```

Activate the virtual environment created:

```bash
source env/bin/activate 
```

Then install all requirements:

```bash
pip install -r requirements.txt
```

Finally, examples in the *examples* folder can be executed straightaway. Note that they need datasets to be preprocessed beforehand. This can be easily done by following the instructions in [the submodule](https://github.com/DaniUPC/protodata).

## Future tasks

Here are a list of tasks to be done in the near future:

- Create submodule for shared classes in the different branches.
- Split the branches into separate forked repositories.
- Implement new regularization methods and test them in depth.
