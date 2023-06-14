# Variable Length Embeddings

## Requirements

Setting up conda environments with the following command
```
cd variable-length-embedding
conda env create -f environment.yml
conda activate vle
```

## Running

To run simple training, run the following
```
PYTHONPATH=. python main.py --base configs/training_config.yaml --devices 0,
```

