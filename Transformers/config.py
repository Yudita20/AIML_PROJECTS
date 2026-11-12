from pathlib import Path

def get_config():
    config = {
        'src_lang': 'en',
        'tgt_lang': 'it',
        'seq_len': 350,
        'batch_size': 8,
        'd_model': 512,
        'num_epochs': 20,
        'learning_rate': 1e-4,
        'model_folder': 'weights',
        'model_filename' : 'tmodel_',
        'prelaod': None,
        'tokenizer_filw' : 'tokenizer_{0}.json',
        'experiment_name' : 'runs/tmodel'
    }
    return config


def get_weights_file_path(config, epoch: str):
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_filename = f"{model_basename}_{epoch}.pt"
    return str(Path('') / model_folder / model_filename)