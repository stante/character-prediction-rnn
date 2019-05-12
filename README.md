# character-prediction-rnn

## Installation
The easiest way to get the provided code running is by creating a python environment based on the provided 
`environment.yml` file.

```sh
$ conda env create -f environment.yml
```

This creates a conda environment called `neural-style-transfer`, which shall be activated prior to use of the command 
line tool.

```sh
$ conda activate character-prediction-rnn
```

If the conda environment already exists, it can be activated with the following command.

```
$ conda env update -f environment.yml
```

## Usage
There are two command line application. One is for learning from a text corpus and saving the model  to a file 
(`learn-text`) and the other one is for reading the learned model and creating text predictions (`predict-text`).
```sh
Usage: learn-text.py [OPTIONS] TEXT_FILE WRITE_MODEL                                                           
                                                                                                               
Options:                                                                                                       
  --epochs INTEGER                                                                                             
  --help            Show this message and exit.  
```

```sh
Usage: predict-text.py [OPTIONS] READ_MODEL                                                                                                                                                                                      
                                                                                                                                                                                                                                 
Options:
  --n INTEGER
  --help       Show this message and exit.
```
## Examples

## License
character-prediction-rnn is Copyright © 2019 Alexander Stante. It is free software, and may be redistributed under the 
terms specified in the [LICENSE](/LICENSE) file.