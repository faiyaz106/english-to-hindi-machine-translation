from pathlib import Path

def get_config():
    return {
        "batch_size": 32,
        "num_epochs": 40,
        "lr": 10**-4,
        "seq_len": 250,   # Maximum sequence length 
        "d_model": 512,     # Embedding vector size 
        "h":2,            # Number of self-Attention Head (In vanilla Transformer it's 8)
        "N":2,            # Number of repeating Encoder and Decoder Block (In vanilla Transformer it's 6)
        "dropout":0.1,
        "d_ff":256,      # In vanilla transformer it's 2048
        "datasource": 'cfilt/iitb-english-hindi',   
        "lang_src": "en",
        "lang_tgt": "hi",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": "latest",
        "tokenizer_method":"bpe", 
        "bpe_tokenizer_file": "bpe_tokenizer_{0}.json",
        "wordlevel_tokenizer_file": "wl_tokenizer_{0}.json",
        "experiment_name": "runs/tmodel"
    }

def get_weights_file_path(config, epoch: str):
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
    model_folder = config['model_folder']
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])

