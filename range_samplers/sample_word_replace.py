import torch
import random
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer
)


def sample_word_replace(range_center, mask_model, mask_tokenizer, num_masks, sample_size, device):
    """
    Sample sentences with word replacements using a masked language model.
    Args:
        range_center (str): The sentence to sample around.
        mask_model (str): The masked language model to use.
        mask_tokenizer (str): The tokenizer for the masked language model.
        num_masks (int): The number of words to mask in each sentence.
        sample_size (int): The number of sentences to sample.
        device (str): The device to run the masked language model on.
    Returns:
        list: Sentences sampled with word replacements.
    """
    # Load the masked language model
    mlm_model = AutoModelForMaskedLM.from_pretrained(mask_model).to(device)
    mlm_tokenizer = AutoTokenizer.from_pretrained(mask_tokenizer)

    # Mask the input sentence
    attempts = 0
    new_sentences = [range_center]
    words = range_center.split()
    num_words = len(words)

    if num_words <= num_masks:
        # If there are not enough words to mask, return the original text repeated m times
        raise ValueError("Not enough words to mask")

    while len(new_sentences) < sample_size:
        masked_words = words[:]
        k = random.randint(1, num_masks)
        masked_indices = random.sample(range(num_words), k)
        original_masked_words = [
            words[idx] for idx in masked_indices
        ]  # Keep original words for comparison
        for index in masked_indices:
            masked_words[index] = "[MASK]"
        masked_sentence = " ".join(masked_words)

        # Tokenize the masked sentence
        inputs = mlm_tokenizer(masked_sentence, return_tensors="pt").to(device)

        # Get predictions
        with torch.no_grad():
            outputs = mlm_model(**inputs)
            predictions = outputs.logits

        # Replace '[MASK]' tokens by selecting from the top 6, excluding the original word
        mask_token_index = (inputs.input_ids[0] == mlm_tokenizer.mask_token_id).nonzero(
            as_tuple=True
        )[0]
        for i, mask_index in enumerate(mask_token_index):
            mask_word_logits = predictions[0, mask_index]
            top_tokens = torch.topk(mask_word_logits, 6, dim=0).indices
            # Decode tokens to filter out the original word
            decoded_tokens = [mlm_tokenizer.decode([tok_id]) for tok_id in top_tokens]
            filtered_tokens = [
                tok_id
                for tok, tok_id in zip(decoded_tokens, top_tokens)
                if tok != original_masked_words[i]
            ]

            if filtered_tokens:
                # Select a random token from filtered tokens that is not the original masked word
                selected_token = random.choice(filtered_tokens)
                inputs.input_ids[0, mask_index] = selected_token
            else:
                # If all top tokens are the original word, use the top token by default
                raise ValueError("All top tokens are the original word")

        # Decode the modified sentence
        new_sentence = mlm_tokenizer.decode(
            inputs.input_ids.squeeze(0), skip_special_tokens=True
        )

        # Check for duplicates
        if new_sentence != range_center and new_sentence not in new_sentences:
            new_sentences.append(new_sentence)
        else:
            attempts += 1
            if attempts >= 100:
                raise ValueError("Could not generate enough unique sentences")
    return new_sentences