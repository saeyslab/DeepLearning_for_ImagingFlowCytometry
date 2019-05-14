# Deep Learning for Imaging Flow Cytometry

This repository contains the code used to generate results for "Include link to paper". A framework around Tensorflow-Keras is provided that makes it easy to train and evaluate any deep learning architecture on imaging flow cytometry data.

## Citation
If you use this package please cite the following publication:

## Installation

### Prerequisites
This installation requires Python (tested on version 3.6 and 3.7) and git.

To use GPU functionality, please refer to the [TensorFlow guide](https://www.tensorflow.org/install/gpu).

### Framework installation
Clone this repository:
```
git clone https://github.com/saeyslab/X
```
Use the package manager pip to install all dependencies:
```
pip install -r requirements.txt
```
## Usage
### Data
#### Images
Images should be fed to the network in the form of HDF5 files containing images and accompanying masks.

The HDF5-file is required to have the following structure:
- channel_1 (HDF5 Group)
  - images (HDF5 Dataset, uint16 2D-arrays)
  - masks (HDF5 Dataset, boolean 2D-arrays)
- channel_9
  - images
  - masks
- ...

The datasets in all the groups need to have a corresponding order. This means that `data["channel_1/images"][0]` and `data["channel_9/images"][0]` are both images of the same cell. Also `data["channel_1/masks"][0]` is the mask for `data["channel_1/images"][0]`.

In case your data is acquired on an Amnis-platform a [tool](https://github.com/saeyslab) is available to convert CIF-files (as exported from IDEAS software) to the required HDF5-files.
#### Labels
For training, a textfile needs to be provided containing the class labels for each input image. The labels are assumed to be in the same order as the HDF5-file. 
#### Cross-validation splits
For cross-validation, a directory structure containing cross-validation splits needs to be provided. The structure is as follows:
- 0
  - train.txt
  - val.txt
- 1
  - train.txt
  - val.txt
- ...

The txt-files contain `int`s separated by newlines. Each int refers to a cell in the HDF5 file.

### Deep learning
The framework contains 5 functionalities:
- cross-validating (`cv`),
- training (`train`),
- predicting (`predict`),
- embedding (`embed`),
- and hyper-parameter searching (`param`).

The basic command to run any of the functionalities is
```
python main.py function_name json_config run_dir data_root 
```
with `python main.py` fixed,
- `function_name` any of the names in brackets above,
- `json_config` a json-file containing all options,
- `run_dir` path to a dir where output can be stored,
- and `data_root` path to root dir containing data.

#### Functionalities overview
##### Cross-validation (`cv`)
In order to perform cross-validation you need to provide the HDF5-file with images and masks, the labels and CV directory structure as explained under [Data](###Data).

The framework will store several items in `run_dir`:
- best performing model per fold,
- current model,
- file with all metrics.

##### Training (`train`)
In order to perform training you need to provide the HDF5-file with images and masks and the labels as explained under [Data](###Data).

The framework will store several items in `run_dir`:
- best model,
- file with all metrics

#### JSON-config
All parameters required to use a functionality are read from a JSON-config file. Some [example configs](./configs/) are available in this repository.

#### Data root
This is the path the first common parent directory of the CV directory, HDF5-file and labels-file. In the config file, make sure to specify all directories and files relative to the data root.

This setup makes it easy to re-use JSON-configs accross machines, as long as the directory tree starting from the data root is the same for every machine.