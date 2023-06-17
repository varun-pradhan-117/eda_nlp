import nlpaug.augmenter.word as naw
from transformers import MarianMTModel, MarianTokenizer


# The base class for data augmentation. It doesn't perform any operation.
class Augmenter:
    def augment(self, text):
        raise NotImplementedError  # This should be implemented in each subclass


# This class performs synonym augmentation. It replaces words in the input with their synonyms.
class SynonymAugmenter(Augmenter):
    def __init__(self):
        self.aug = naw.SynonymAug(aug_src='wordnet')  # Use WordNet as the source of synonyms

    def augment(self, batch_texts):
        # For each text in the batch, replace some words with their synonyms
        augmented_texts = [self.aug.augment(text) for text in batch_texts]
        # Join the lists of words into strings
        augmented_texts = [' '.join(text) for text in augmented_texts]
        return augmented_texts


# This class performs backtranslation augmentation. It translates the input to French and then back to English.
class BacktranslationAugmenter(Augmenter):
    def __init__(self, device):
        self.device = device  # The PyTorch device (CPU or GPU) to use for model computations
        # Load the tokenizer and model for English-to-French translation
        self.tokenizer_src = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-fr')
        self.model_src = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-fr')
        # Load the tokenizer and model for French-to-English translation
        self.tokenizer_tgt = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-fr-en')
        self.model_tgt = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-fr-en')
        # Move the models to the specified device
        self.model_src = self.model_src.to(device)
        self.model_tgt = self.model_tgt.to(device)

    def translate(self, batch_texts, model, tokenizer):
        """Translate a batch of texts using the provided model and tokenizer."""
        # Encode the texts for the model
        encoded_texts = tokenizer(batch_texts, truncation=True, max_length=512, padding='longest',
                                  return_tensors='pt').to(self.device)
        # Use the model to generate translations of the texts
        translation_logits = model.generate(**encoded_texts, max_new_tokens=512)
        # Decode the model's output into human-readable text
        decoded_translations = tokenizer.batch_decode(translation_logits, skip_special_tokens=True)
        return decoded_translations

    def augment(self, batch_texts):
        """Backtranslate a batch of texts."""
        # Translate the texts to French
        fr_texts = self.translate(batch_texts, self.model_src, self.tokenizer_src)
        # Translate the French texts back to English
        en_texts = self.translate(fr_texts, self.model_tgt, self.tokenizer_tgt)
        return en_texts
