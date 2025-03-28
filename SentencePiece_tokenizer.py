import sentencepiece as spm
import json, os
from helper_functions import load_data


# Training sentencepiece tokenizer
#spm_input = "full_text.txt"
# spm_input = ["cleaned_text/combined.txt"]
spm_model_prefix = 'sentencepiece_tokenizer'

# Preprocessing to preserve newlines and tabs
with open("cleaned_text/combined.txt", "r", encoding="utf-8") as f:
    data = f.read()
data = data.replace("\n", " [NEWLINE] ").replace("\t", " [TAB] ")
if not data.strip():
    raise ValueError("Your preprocessed file is empty!")
with open("cleaned_text/combined_preprocessed.txt", "w", encoding="utf-8") as f:
    for line in data.split(". "):
        f.write(line.strip() + "\n")


spm_input = ["cleaned_text/combined_preprocessed.txt"]

vocab_size = 512

#training
spm.SentencePieceTrainer.train(
    input=spm_input,
    model_prefix="sentencepiece",
    vocab_size=vocab_size,
    model_type='unigram',
    character_coverage=1.0,
    user_defined_symbols=["[CLS]", "[SEP]", "[PAD]", "[MASK]", "[NEWLINE]", "[TAB]"]
)

def restore_formatting(text):
    return text.replace("[NEWLINE]", "\n").replace("[TAB]", "\t")

def prepare_input(text):
    return text.replace("\n", " [NEWLINE] ").replace("\t", " [TAB] ")

def encode_text(text, tokenizer, out_type=int):
    return tokenizer.encode(prepare_input(text), out_type=out_type)

def decode_text(token_ids, tokenizer):
    return restore_formatting(tokenizer.decode(token_ids))

# Loading the trained tokenizer
sp = spm.SentencePieceProcessor()
sp.load('sentencepiece.model')

# Testing
input_string = "Hello world, this is a test!"
input_string2 = (
    "This is the Hugging Fa\n"
    "ce Cou\trse. This chapter is about tokenization. "
    "This section shows several tokenizer algorithms. "
    "Hopefully, you will be able to understand how they are trained "
    "and generate tokens."
)

tokens = encode_text(input_string2, sp)
decoded_text = decode_text(tokens, sp)

print(tokens)
print("Decoded:", decoded_text)
print(encode_text("Hello\nworld\tagain!",sp, out_type=str))


#
# vocab = []
# for i in range(sp.get_piece_size()):
#     token = sp.id_to_piece(i)
#     vocab.append({"id": i, "token": token})
#
# tokenizer_json = {
#     "vocab_size": sp.get_piece_size(),
#     "vocab": vocab,
#     "special_tokens": ["[CLS]", "[SEP]", "[PAD]", "[MASK]"]
# }
#
# # Save the JSON file
# with open("Sentencepiece_tokenizer.json", "w", encoding="utf-8") as json_file:
#     json.dump(tokenizer_json, json_file, ensure_ascii=False, indent=4)
#
# print("Tokenizer saved as JSON!")
