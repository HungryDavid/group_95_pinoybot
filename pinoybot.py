"""
pinoybot.py

PinoyBot: Filipino Code-Switched Language Identifier

This module provides the main tagging function for the PinoyBot project, which identifies the language of each word in a code-switched Filipino-English text. The function is designed to be called with a list of tokens and returns a list of tags ("ENG", "FIL", "CS", or "OTH").

Model training and feature extraction should be implemented in a separate script. The trained model should be saved and loaded here for prediction.
"""

import os
import joblib
from typing import List

MODEL_PATH = 'model.pkl'
VECTORIZER_PATH = 'vectorizer.pkl'

def extract_features(tokens, index):
    word = tokens[index]
    word_lower = word.lower()
    
    features = {
        'word.lower': word_lower,
        'word.length': len(word),
        'word.is_capitalized': word[0].isupper() if word else False,
        'word.is_all_caps': word.isupper(),
        'word.is_all_lower': word.islower(),
        'word.contains_digit': any(char.isdigit() for char in word),
        'word.contains_punct': any(not char.isalnum() for char in word),
        'word.prefix_2': word_lower[:2] if len(word_lower) >= 2 else word_lower,
        'word.prefix_3': word_lower[:3] if len(word_lower) >= 3 else word_lower,
        'word.suffix_2': word_lower[-2:] if len(word_lower) >= 2 else word_lower,
        'word.suffix_3': word_lower[-3:] if len(word_lower) >= 3 else word_lower,
        'word.has_fil_prefix': any(word_lower.startswith(p) for p in ['nag', 'mag', 'pag', 'pa', 'na', 'um', 'in', 'ka']),
        'word.has_eng_suffix': any(word_lower.endswith(s) for s in ['ing', 'ed', 's', 'tion', 'ment', 'ity']),
    }
    
    if index > 0:
        prev_word = tokens[index - 1]
        features.update({
            'prev_word.lower': prev_word.lower(),
            'prev_word.is_capitalized': prev_word[0].isupper() if prev_word else False,
        })
    else:
        features['START'] = True

    if index < len(tokens) - 1:
        next_word = tokens[index + 1]
        features.update({
            'next_word.lower': next_word.lower(),
            'next_word.is_capitalized': next_word[0].isupper() if next_word else False,
        })
    else:
        features['END'] = True
        
    return features

def tag_language(tokens: List[str] =["Love", "kita", "."]) -> List[str]:
    """
    Tags each token in the input list with its predicted language.
    Args:
        tokens: List of word tokens (strings).
    Returns:
        tags: List of predicted tags ("ENG", "FIL", "CS", or "OTH"), one per token.
    """
    # 1. check if model exist then load using joblib
    model = None 
    vectorizer = None
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH): 
        try:
            model = joblib.load(MODEL_PATH)
            vectorizer = joblib.load(VECTORIZER_PATH)
        except Exception as e:
            print(f"Could not load files: {e}")
            
    # extra safety features: if the model does not exist/model path is wrong then the tags for each token will be null or unevaluated
    if model is None or vectorizer is None:
        return ['NULL' for _ in tokens]
        
    # extra safety features: if tokens is empty then tags will also be empty
    if not tokens:
        return []
    
    # 2. extract feature dictionaries for every token in the incoming sequence
    # we call extract_features to extract certain features exactly the same as to how we extract features from the dataset
    features_list = [extract_features(tokens, i) for i in range(len(tokens))]
    X_feat = vectorizer.transform(features_list)

    # 3. predict using the loaded sklearn model pipeline
    predicted = model.predict(X_feat)

    # 4. convert predictions to a list of strings using list comprehensions
    tags = [str(tag) for tag in predicted]
    
    # 5. return tags
    return tags

if __name__ == "__main__":
    # Example usage
    sentence = input("Enter a sentence: ")
    example_tokens = sentence.split()
    print(f"Tokens: {example_tokens}")
    tags = tag_language(example_tokens)
    print(f"Tags:   {tags}")