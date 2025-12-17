from nltk import Nonterminal
from data_builder.cfg import load_cfg_nltk, generate_random_sentence

N_SENTENCES = 500000
RES_PATH = "data/cfg_artif_data_1.txt"
CFG_PATH = "data_builder/english_cfg.cfg"
cf_grammar = load_cfg_nltk(CFG_PATH)

tgt_file = open(RES_PATH, "w+")
for _ in range(N_SENTENCES):
    words = generate_random_sentence(cf_grammar, Nonterminal('S'))
    tgt_file.write(" ".join(words) + "\n")
