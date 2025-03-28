import re, collections

class BPE:
    # Initialization
    def __init__(self, vocab_size=512):
        self.vocab_size = vocab_size
        self.merges = []
        self.vocab = {}
        self.special_tokens = {'<tab>', '<space>', '<newline>', '</w>'}

    def get_stats(self, vocab):
        """Count pair frequency"""
        pairs = collections.defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i + 1]] += freq
        return pairs

    def merge_vocab(self, pair, v_in):
        """Merging pairs of chars"""
        v_out = {} #vocabulary out
        bigram = re.escape(' '.join(pair)) #connverts a pair to bigram, also escapes special characters (re.escape)
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)') #ensures bigram is preceeded and followed by a space
        for word in v_in:
            w_out = p.sub(''.join(pair), word) #substitute pair of chars with joined pair
            v_out[w_out] = v_in[word] #add to vocab out
        return v_out

    #train tokenizer on corpus
    def train(self, text):
        symbols = set()
        vocab = collections.Counter()

        # handling of spaces and newlines
        words = re.findall(r'\S+|\n|\s|\t', text)
        for word in words:
            if word == ' ':
                vocab['<space>'] += 1
                symbols.add('<space>')
            elif word == '\n':
                vocab['<newline>'] += 1
                symbols.add('<newline>')
            elif word == '\t':
                vocab['<tab>'] += 1
                symbols.add('<tab>')
            else:
                chars = list(word) + ['</w>']
                vocab[' '.join(chars)] += 1
                symbols.update(chars)
        symbols.add('</w>')  # end-of-word symbol

        initial_symbols = symbols.copy()
        num_merges = self.vocab_size - len(initial_symbols)

        merges = []
        # Selecting the most common pair and appending it to merges
        for _ in range(num_merges):
            pairs = self.get_stats(vocab)
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            vocab = self.merge_vocab(best_pair, vocab)
            merges.append(best_pair)

        self.merges = merges

        # Final vocabulary
        self.vocab = set(initial_symbols)
        for merge in merges:
            self.vocab.add(''.join(merge))

        print("Final vocab size:", len(self.vocab))

    def encode(self, text):
        """Encode text into tokens"""
        #Using findall to get all words and instances of \s \n
        words = re.findall(r'\S+|\n|\s|\t', text)
        encoded_tokens = []

        for word in words:
            #Handling of special tokens
            if word == ' ':
                encoded_tokens.append('<space>')
            elif word == '\n':
                encoded_tokens.append('<newline>')
            elif word == '\t':
                encoded_tokens.append('<tab>')
            else:
                chars = list(word) + ['</w>']
                tokens = chars

                #Initial pairs
                pairs = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
                while True:
                    pair_freq = {}

                    for pair in pairs:
                        if pair in self.merges:  # does the pair exist in merge rules
                            pair_freq[pair] = self.merges.index(
                                pair)

                    if not pair_freq:
                        break
                    best_pair = min(pair_freq, key=pair_freq.get)
                    tokens = self.merge_pair(tokens, best_pair)
                    if len(tokens) == 1:
                        break
                    pairs = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
                encoded_tokens.extend(tokens)

        return encoded_tokens

    def merge_pair(self, tokens, pair):
        """Merge pairs of tokens"""
        merged = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == pair:
                merged.append(''.join(pair))
                i += 2
            else:
                merged.append(tokens[i])
                i += 1
        return merged

    def decode(self, tokens):
        """Decode from tokens"""
        text = ''.join(tokens)
        text = text.replace('</w>', '').replace('<space>', ' ').replace('<newline>', '\n').replace('<tab>','\t')
        return text

    def build_token_ids(self):
        """This function assigns ids to tokens"""
        # Include special tokens explicitly to avoid KeyError
        full_vocab = self.vocab.union(self.special_tokens)
        self.token_to_id = {token: idx for idx, token in enumerate(sorted(full_vocab))}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}

    def encode_to_ids(self, text):
        """Encode "encoded" tokens to ids"""
        tokens = self.encode(text)
        return [self.token_to_id[token] for token in tokens]

    def decode_from_ids(self, ids):
        "Decode from ids to tokens"
        tokens = [self.id_to_token[id_] for id_ in ids]
        return self.decode(tokens)

    def get_vocab_size(self):
        return len(self.vocab)
