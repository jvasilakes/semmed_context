# Modeling SemRepFact

First we need to generate a default experiment config file that we can later modify to run our experiments.

```
python config.py new path/to/config.yaml
```

This default config file will have some missing values, which we'll need to fill in manually. You can see which
values need filling in with

```
python config.py validate path/to/config.yaml
```

One of the things the above script says we need is a valid `Data.datadir`.
To get this, we need to prepare the dataset. We preprocess by splitting into train, val, and test, and
encoding each example. We also save each split as a .tar.gz archive to save space, and so we can
use `webdataset` to read it quickly during training and evaluation.

```
python utils/preprocess_dataset.py path/to/dataset/factuality/converted path/to/dataset/tar
```

The resulting `path/to/dataset/tar` can then be specified in the config file under `Data.datadir`.
You can either do this by manually editing the config file, or by running

```
python config.py update -f path/to/config.yaml -p Data.datadir path/to/dataset/tar
```

Hint: the `update` command allows you to programmatically update multiple config files at once. For example,
the following updates the `Data.datadir` parameter for all yaml files in `config_dir/`.

```
python config.py update -f path/to/config_dir/*.yaml -p Data.datadir path/to/dataset/tar
```

You should fill in or modify the other values in the config file as you wish for your experiment.
Documentation is included in the config file by default. Once you have a complete config file
(i.e., `python config.py validate` doesn't return any warnings or errors), you can run experiments.

```
python run.py train path/to/config.yaml
python run.py validate path/to/config.yaml --split {train,val,test}
python run.py predict path/to/config.yaml --split {train,val,test}
```
