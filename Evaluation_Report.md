# PinoyBot Evaluation Report

**Group Number:** 95
**Group Name:** PinoyBot Team
**Group Members:** 
- Matthew Fabregas (Section: [Your Section])
- [Member 2 Name] (Section: [Member 2 Section])
- [Member 3 Name] (Section: [Member 3 Section])

*Note: This report was generated with the assistance of an AI tool (Gemini 3.1 Pro).*

---

## 1. Performance Metrics

We tested our bot against the test sentences dataset. Below are the computed metrics for each category (True Positives, False Positives, False Negatives, Precision, Recall, and F1-Score):

| Category | TP | FP | FN | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **FIL** (Filipino) | 2139 | 36 | 61 | 0.9834 | 0.9723 | 0.9778 |
| **ENG** (English) | 326 | 46 | 34 | 0.8763 | 0.9056 | 0.8907 |
| **CS** (Code-Switched) | 0 | 12 | 0 | 0.0000 | 0.0000 | 0.0000 |
| **OTH** (Other) | 382 | 54 | 53 | 0.8761 | 0.8782 | 0.8772 |

**Overall Accuracy:** 95.06%

---

## 2. Most Common Misclassifications

The confusion matrix reveals that the model struggles primarily with distinguishing proper nouns, borrowed words, and punctuation. The most frequent misclassifications are:

1. **True: FIL -> Pred: OTH (38 occurrences)**
   - **Example:** Token `'Yan'` in context `"... talagang ayoda Yan Ang Sabi ..."`
   - **Reasoning:** Capitalized Filipino words often get mistaken for named entities or "Other" (OTH).
2. **True: OTH -> Pred: ENG (35 occurrences)**
   - **Example:** Token `'Migs'` in context `"... ni Doc Migs , yun ..."`
   - **Reasoning:** Proper nouns (like names) that end in typical English suffixes (like "s") trigger English rules.
3. **True: OTH -> Pred: FIL (18 occurrences)**
   - **Example:** Token `'NASA'` in context `"... Sabi ng NASA pang government ..."`
4. **True: ENG -> Pred: FIL (18 occurrences)**
   - **Example:** Token `'dad'` in context `"... toward my dad . ..."`
   - **Reasoning:** Short English words without clear English morphological markers (like -ing, -ed) are defaulted to FIL.
5. **True: ENG -> Pred: OTH (16 occurrences)**
   - **Example:** Token `'cobra'` in context `"... kasi ang cobra ang mag ..."`
6. **True: FIL -> Pred: CS (12 occurrences)**
   - **Example:** Token `'nag-isip'` in context `"... ito kaya nag-isip ako ng ..."`
   - **Reasoning:** The hyphen in "nag-isip" likely confused the model, as hyphens are common in Code-Switched words (e.g., "nag-decide").

---

## 3. Analysis of Failure Points

Based on the common misclassifications and the feature extraction logic in `pinoybot.py`, the following factors contributed to the incorrect predictions:

1. **Over-reliance on Simple Heuristics for Suffixes:**
   - The feature `'word.has_eng_suffix'` checks if a word ends in "s", "ing", "ed", etc. This causes proper nouns or acronyms (like "Migs") to be falsely flagged as English simply because they end in "s". 
2. **Capitalization Confusion:**
   - The model uses `word.is_capitalized` as a feature. While this is intended to catch proper nouns (OTH), it backfires when users capitalize regular Filipino words (e.g., "Yan") for emphasis or at the start of a sentence. This leads to FIL words being classified as OTH.
3. **Hyphenated Words:**
   - The model misclassified pure Filipino hyphenated words (like "nag-isip") as Code-Switched (CS). In code-switching, prefixes are often attached to English verbs with a hyphen (e.g., "nag-drive"). The model learned to associate the hyphen heavily with CS, causing false positives on legitimate FIL words.
4. **Lack of Contextual Depth:**
   - While the model looks at the previous and next word, Logistic Regression and Naive Bayes are relatively simple models that cannot capture deep semantic context. For example, "dad" is treated as Filipino because it lacks explicit English morphological features and the model relies more on character-level rules.
5. **Class Imbalance:**
   - The model completely failed to predict `CS` (Code-Switched) correctly in the test set (Precision/Recall of 0). The training set might not have had enough `CS` examples, or the features extracted were not distinct enough to separate `CS` from `FIL` or `ENG`.
