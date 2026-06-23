# How to Run the Project

This repository contains the Python emulator/experiments and the SystemVerilog RTL implementation for the master thesis ReCA classifier project.

The project is divided into three main parts:

* `python/` — Python emulator and experiment scripts
* `rtl/` — SystemVerilog RTL hardware design files
* `testbench/` — SystemVerilog testbench files

Large datasets, Quartus generated folders, and ModelSim simulation output files are not included in this repository.

---

## 1. Python Emulator and Experiments

The Python files are located in the `python/` folder.

Main Python file:

```bash
python3 python/The_Golden_one.py
```

This file represents the main Python model/emulator used for the ReCA-based classifier experiments.

There are also optional experiment folders, for example:

```text
python/Optional_Adam/
python/Optional_Proposed_WTA_classifier_Binary-MNIST/
```

These folders contain additional Python experiment versions, such as Adam-based experiments and proposed WTA classifier experiments.

To run one of the optional files, use:

Google colab
```

---

## 2. Dataset Files

The Python scripts may require MNIST-related data files to be available locally.

Large dataset files are not included in this GitHub repository because they can be too large.

Examples of files that may be needed locally:

```text
mnist_train_hw.txt
mnist_val_hw.txt
mnist_test_hw.txt
```

If these files are required, place them in the correct local folder expected by the Python script or update the file path inside the script.





---

## 7. Repository Purpose

This repository is intended to document and preserve the main source code used in the master thesis project.

It includes:

* Python ReCA classifier models and experiments
* Basic documentation for running and understanding the project

The repository does not include large datasets or generated tool output files.
