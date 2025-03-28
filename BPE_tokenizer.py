from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace, ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
import random
import os
# Initialize a blank BPE tokenizer
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
tokenizer.decoder = ByteLevelDecoder()


trainer = BpeTrainer(
    vocab_size=512,
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
)


files = ["cleaned_text/combined.txt"]
tokenizer.train(files, trainer)
tokenizer.save("bpe_tokenizer.json")
print("Tokenizer saved to 'bpe_tokenizer.json'")
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
print("Decoded: ", decoded)