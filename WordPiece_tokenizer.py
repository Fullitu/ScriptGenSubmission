from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import BertPreTokenizer
from tokenizers.decoders import WordPiece as WordPieceDecoder
import time

# Initialize a blank Unigram tokenizer
tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
tokenizer.pre_tokenizer = BertPreTokenizer()
tokenizer.decoder = WordPieceDecoder()


trainer = WordPieceTrainer(
    vocab_size=512,
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
)

#files = ["seinfeld_scripts.txt"]
files = ["cleaned_text/combined.txt"]
start = time.time()
tokenizer.train(files, trainer)
tokenizer.save("WordPiece_tokenizer.json")
stop = time.time()
print("Tokenizer saved: ", start - stop)


tokenizer = Tokenizer.from_file("bpe_tokenizer.json")

input_string = "Hello world, this is a test!"
input_string2 = (
    "This is the Hugging Fa\n"
    "ce Cou\trse. This chapter is about tokenization. "
    "This section shows several tokenizer algorithms. "
    "Hopefully, you will be able to understand how they are trained "
    "and generate tokens."
)

encoded = tokenizer.encode(input_string2)
decoded = tokenizer.decode(encoded.ids)


print("Token IDs:", encoded.ids)
print("Tokens:", encoded.tokens)
print("Decoded:  ", decoded)
print("Cleaned Decoded:", decoded)