from tokenizers import Tokenizer
from tokenizers.models import Unigram
from tokenizers.trainers import UnigramTrainer
from tokenizers.pre_tokenizers import Metaspace
from tokenizers.decoders import Metaspace as MetaspaceDecoder

# Initialize a blank Unigram tokenizer
tokenizer = Tokenizer(Unigram())
tokenizer.pre_tokenizer = Metaspace()
tokenizer.decoder = MetaspaceDecoder()

trainer = UnigramTrainer(
    vocab_size=512,
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
)

#files = ["seinfeld_scripts.txt"]
files = ["cleaned_text/combined.txt"]
tokenizer.train(files, trainer)
tokenizer.save("Unigram_tokenizer.json")


tokenizer = Tokenizer.from_file("bpe_tokenizer.json")

input_string = "Hello world, this is a test!"
input_string2 = (
    "This is the Hugging Fa\n"
    "ce Cou\trse. This chapter is about tokenization. "
    "This section shows several tokenizer algorithms. "
    "Hopefully, you will be able to understand how they are trained "
    "and generate tokens."
)
encoded = tokenizer.encode(input_string)
decoded = tokenizer.decode(encoded.ids)

print("Token IDs:", encoded.ids)
print("Tokens:", encoded.tokens)
print("Decoded: ", decoded)