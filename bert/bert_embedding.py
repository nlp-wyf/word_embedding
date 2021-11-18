import torch
from scipy.spatial.distance import cosine
from transformers import BertTokenizer, BertModel

import matplotlib.pyplot as plt

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define a new example sentence with multiple meanings of the word "bank"
text = "After stealing money from the bank vault, the bank robber was seen " \
       "fishing on the Mississippi river bank."

# Add the special tokens.
marked_text = "[CLS] " + text + " [SEP]"

# Split the sentence into tokens.
tokenized_text = tokenizer.tokenize(marked_text)

# Map the token strings to their vocabulary indeces.
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

# Display the words with their indeces.
for tup in zip(tokenized_text, indexed_tokens):
    print('{:<12} {:>6,}'.format(tup[0], tup[1]))

# Mark each of the 22 tokens as belonging to sentence "1".
segments_ids = [1] * len(tokenized_text)

print(indexed_tokens)
print(segments_ids)

# Convert inputs to PyTorch tensors
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])

# Load pre-trained model (weights)
# 'output_hidden_states' Whether the model returns all hidden-states.
model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)

# Put the model in "evaluation" mode, meaning feed-forward operation.
model.eval()

# Run the text through BERT, and collect all of the hidden states produced from all 12 layers.
with torch.no_grad():
    outputs = model(tokens_tensor, segments_tensors)

    # Evaluating the model will return a different number of objects based on
    # how it's  configured in the `from_pretrained` call earlier. In this case,
    # because we set `output_hidden_states = True`, the third item will be the
    # hidden states from all layers. See the documentation for more details:
    # https://huggingface.co/transformers/model_doc/bert.html#bertmodel
    hidden_states = outputs[2]

    print("Number of layers:", len(hidden_states), "  (initial embeddings + 12 BERT layers)")

    layer_i = 0
    print("Number of batches:", len(hidden_states[layer_i]))

    batch_i = 0
    print("Number of tokens:", len(hidden_states[layer_i][batch_i]))

    token_i = 0
    print("Number of hidden units:", len(hidden_states[layer_i][batch_i][token_i]))

    # Concatenate the tensors for all layers. We use `stack` here to
    # create a new dimension in the tensor.
    token_embeddings = torch.stack(hidden_states, dim=0)

    print(token_embeddings.size())
    # torch.Size([13, 1, 22, 768])

    # Remove dimension 1, the "batches".
    token_embeddings = token_embeddings.squeeze(dim=1)

    print(token_embeddings.size())
    # torch.Size([13, 22, 768])

    # Swap dimensions 0 and 1.
    token_embeddings = token_embeddings.permute(1, 0, 2)

    print(token_embeddings.size())
    # torch.Size([22, 13, 768])

    # Stores the token vectors, with shape [22 x 3,072]
    token_vecs_cat = []

    # `token_embeddings` is a [22 x 12 x 768] tensor.

    # For each token in the sentence...
    for token in token_embeddings:
        # `token` is a [12 x 768] tensor

        # Concatenate the vectors (that is, append them together) from
        # the last four layers.
        # Each layer vector is 768 values, so `cat_vec` is length 3072.
        cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)

        # Use `cat_vec` to represent `token`.
        token_vecs_cat.append(cat_vec)

    print('Shape is: %d x %d' % (len(token_vecs_cat), len(token_vecs_cat[0])))
    # Shape is: 22 x 3072

    # Stores the token vectors, with shape [22 x 768]
    token_vecs_sum = []

    # `token_embeddings` is a [22 x 12 x 768] tensor.

    # For each token in the sentence...
    for token in token_embeddings:
        # `token` is a [12 x 768] tensor

        # Sum the vectors from the last four layers.
        sum_vec = torch.sum(token[-4:], dim=0)

        # Use `sum_vec` to represent `token`.
        token_vecs_sum.append(sum_vec)

    print('Shape is: %d x %d' % (len(token_vecs_sum), len(token_vecs_sum[0])))
    # Shape is: 22 x 768

    # `hidden_states` has shape [13 x 1 x 22 x 768]

    # `token_vecs` is a tensor with shape [22 x 768]
    token_vecs = hidden_states[-2][0]

    # Calculate the average of all 22 token vectors.
    sentence_embedding = torch.mean(token_vecs, dim=0)

    print("Our final sentence embedding vector of shape:", sentence_embedding.size())
    # Our final sentence embedding vector of shape: torch.Size([768])


# After stealing money from the bank vault, the bank robber was seen fishing on the Mississippi river bank.
for i, token_str in enumerate(tokenized_text):
    print(i, token_str)

# 为了确认这些向量的值是上下文相关的，我们可以检查一下例句中 "bank" 这个词的向量
print('First 5 vector values for each instance of "bank".')
print('')
print("bank vault   ", str(token_vecs_sum[6][:5]))
print("bank robber  ", str(token_vecs_sum[10][:5]))
print("river bank   ", str(token_vecs_sum[19][:5]))

# 很明显值不同，但是通过计算向量之间的余弦相似度可以更精确的进行比较
# Calculate the cosine similarity between the word bank
# in "bank robber" vs "bank vault" (same meaning).
same_bank = 1 - cosine(token_vecs_sum[10], token_vecs_sum[6])

# Calculate the cosine similarity between the word bank
# in "bank robber" vs "river bank" (different meanings).
diff_bank = 1 - cosine(token_vecs_sum[10], token_vecs_sum[19])

print('Vector similarity for  *similar*  meanings:  %.2f' % same_bank)  # 0.94
print('Vector similarity for *different* meanings:  %.2f' % diff_bank)  # 0.69
