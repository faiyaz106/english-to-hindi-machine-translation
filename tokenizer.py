from tokenizers import Tokenizer
from tokenizers.models import WordLevel 
from tokenizers.trainers import WordLevelTrainer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from config import get_config
from pathlib import Path

def get_all_sentences(ds, lang):
    for item  in ds:
        yield item['translation'][lang]

def word_level_tokenizer(config, ds,lang):
    tokenizer_path = Path(config['wordlevel_tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer =Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]","[PAD]","[SOS]","[EOS]"], min_frequency=2,vocab_size=1000000)
        tokenizer.train_from_iterator(get_all_sentences(ds,lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def bpe_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['bpe_tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        trainer = BpeTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],min_frequency=2, vocab_size=50000)
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer.train_from_iterator(get_all_sentences(ds,lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

if __name__ == "__main__":
    import sys
    from datasets import load_dataset
    config = get_config()
    ds_raw = load_dataset(config["datasource"])
    ds_raw = ds_raw['train']
    lang = sys.argv[2]
    if sys.argv[1]=="bpe":
        bpe_tokenizer(config, ds_raw, lang)
        print("BPE Tokenizer created sucessfully")
    if sys.argv[1]=="wordlevel":
        word_level_tokenizer(config, ds_raw, lang)
        print("Wordlevel Tokenizer created sucessfully")