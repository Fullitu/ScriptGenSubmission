from collections import defaultdict
import re
import time


class WordPiece:
    def __init__(self):  # add for package: unk_char, whitespace_char etc.
        starting_tokens = [' ', '\n', '\t']
        self.vocab = set()
        self.vocab_size = 0
        self.tokens_to_ids_dict = defaultdict(int)
        self.ids_to_strings_dict = defaultdict(str)

        # add basic tokens
        for c in starting_tokens:
            self._add_token(c)

        # add unk token
        self.tokens_to_ids_dict['[UNK]'] = -1
        self.ids_to_strings_dict[-1] = '[UNK]'

    def _get_splits(self, word_freq):
        splits = {}

        for word in word_freq.keys():
            splits[word] = []
            for i, c in enumerate(word):
                c = c if i == 0 else f'##{c}'
                splits[word].append(c)
                self._add_token(c)

        return splits

    def _calculate_scores(self, word_freq, splits):
        pair_freq = defaultdict(int)
        letter_freq = defaultdict(int)

        for word, split in splits.items():
            freq = word_freq[word]

            if len(word) == 1:
                letter_freq[split[0]] += freq
                continue  # single characters words can't form pairs

            # update frequency count
            for i in range(len(split) - 1):
                pair_freq[(split[i], split[i + 1])] += freq
                letter_freq[split[i]] += freq

            letter_freq[split[-1]] += freq  # add freq of last token

        scores = {pair: pair_freq[pair] for pair in pair_freq.keys()}

        return scores

    def _add_token(self, x):
        if x not in self.vocab:
            self.vocab.add(x)
            self.vocab_size += 1

            id = len(self.vocab)

            self.tokens_to_ids_dict[x] = id
            self.ids_to_strings_dict[id] = x.strip('##')

    def _marge(self, a, b, splits):
        ab_marged = a + b.strip('##')
        self._add_token(ab_marged)

        for word, split in splits.items():
            if len(word) == 1:
                continue

            new_split = []
            skip = False  # determines if item should be skipped
            for i in range(len(split) - 1):
                # skip this item if previous iteration detected a pair
                if skip == True:
                    skip = False
                    continue

                if split[i] == a and split[i + 1] == b:
                    new_token = ab_marged
                    skip = True
                else:
                    new_token = split[i]
                new_split.append(new_token)

                # just apreciate my first try for a sec ;-;
                # new_split.append(ab_marged if split[i] == a and split[i + 1] == b else split[i])

            # append last item if it wasn't in a pair
            if skip == False:
                new_split.append(split[-1])

            splits[word] = new_split

    def _pre_tokenize(self, text):
        return re.findall(r'\S+|\n|\t|\s', text)  # split by whitespaces, new lines and tabs

    def _preprocess(self, text):
        tokens_raw = self._pre_tokenize(text)
        preprocessed = defaultdict(int)

        for token in tokens_raw:
            if sum(c.isdigit() or not c.isalnum() for c in token) / len(token) > 0.3:
                continue

            preprocessed[token] += 1

        return preprocessed

    def train(self, text, vocab_size=512, show_progress=False):
        start = time.time()
        if show_progress:
            last_action = start
            print('training started')
            print()

        word_freq = self._preprocess(text)
        if show_progress:
            print('preprocessing done in ' + str(time.time() - last_action) + ' seconds')
            print()
            last_action = time.time()

        splits = self._get_splits(word_freq)
        if show_progress:
            print('initial split done' + str(time.time() - last_action) + ' seconds')
            print()
            last_action = time.time()

        if vocab_size < len(self.vocab):
            print(
                f"Warning: Very small vocab_size inputted, in order to fit basic letter tokens vocab_size has been expanded to {len(self.vocab)}")

        while len(self.vocab) < vocab_size:

            if show_progress and len(self.vocab) % 100 == 0:
                print(f'{len(self.vocab)} / {vocab_size} tokens added')

            scores = self._calculate_scores(word_freq, splits)
            best_pair = None if scores == {} else sorted(scores.items(), key=lambda x: x[1], reverse=True)[0][0]

            # fail safe condition
            if best_pair == None:
                print("Warning: all passible comibations reached, consider decreasing vocab_size")
                break

            self._marge(best_pair[0], best_pair[1], splits)

        if show_progress:
            print()
            print('training completed')
            print("total time: " + str(time.time() - start))

    def encode_word(self, word):
        i = len(word)
        result = []
        while i > 0:
            current_substring = word[:i]
            if result != []: current_substring = '##' + current_substring

            if current_substring in self.vocab:
                result.append(self.tokens_to_ids_dict[current_substring])
                word = word[i:]
                i = len(word)
            elif i == 1:
                result = [self.tokens_to_ids_dict['[UNK]']]
                return result
            else:
                i -= 1
        return result

    def encode(self, text):
        pre_processed_data = self._pre_tokenize(text)

        encoded = [self.encode_word(word) for word in pre_processed_data]
        encoded = [item for sublist in encoded for item in sublist]

        return encoded

    def encode_to_tokens(self, text):
        self.encode(text)
        return [self.ids_to_strings_dict[id] for id in self.encode(text)]

    def decode(self, ids):
        decoded = ''

        for id in ids:
            if id == -1:  # unk token
                continue
            decoded += self.ids_to_strings_dict[id]
        # TODO: fix issue where double whitespace appears with sequences of unknown tokens
        return decoded