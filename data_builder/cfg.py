import random
from typing import List
from nltk import CFG, Nonterminal

def load_cfg_nltk(path: str):
    '''
    Load a CFG file in simple BNF with quoted terminals and '|' alternations.
    Lines starting with '#' are treated as comments.
    '''
    lines = []
    with open(path, 'r', encoding='utf-8') as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            lines.append(line)
    grammar_text = '\n'.join(lines)
    return CFG.fromstring(grammar_text)

def generate_random_sentence(grammar: CFG, symbol: Nonterminal = Nonterminal('S'), max_depth: int = 12) -> List[str]:
    '''
    Randomly expand from 'symbol' until only terminals remain.
    Depth-limited to avoid infinite recursion.
    '''
    if max_depth <= 0:
        return []
    prods = grammar.productions(lhs=symbol)
    if not prods:
        return [str(symbol)]
    prod = random.choice(prods)
    out: List[str] = []
    for sym in prod.rhs():
        if isinstance(sym, Nonterminal):
            out.extend(generate_random_sentence(grammar, sym, max_depth - 1))
        else:
            out.append(sym)
    return out
