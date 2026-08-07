import os
import joblib
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from pinoybot import extract_features, MODEL_PATH, VECTORIZER_PATH

def evaluate():
    print("Loading model...")
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    print("Model loaded.")

    with open('test sentences/test/test_data.txt', 'r', encoding='utf-8') as f:
        data_lines = f.read().strip().split('\n')
    
    with open('test sentences/test/test_labels.txt', 'r', encoding='utf-8') as f:
        label_lines = f.read().strip().split('\n')
        
    all_true_labels = []
    all_pred_labels = []
    
    misclassifications = []
    
    for i, (data_line, label_line) in enumerate(zip(data_lines, label_lines)):
        tokens = data_line.split('|')
        true_labels = label_line.split('|')
        
        if len(tokens) != len(true_labels):
            print(f"Mismatch in line {i+1}: {len(tokens)} tokens, {len(true_labels)} labels")
            continue
            
        features_list = [extract_features(tokens, j) for j in range(len(tokens))]
        X_feat = vectorizer.transform(features_list)
        predicted = model.predict(X_feat)
        pred_labels = [str(tag) for tag in predicted]
        
        all_true_labels.extend(true_labels)
        all_pred_labels.extend(pred_labels)
        
        for j, (t, p, tok) in enumerate(zip(true_labels, pred_labels, tokens)):
            if t != p:
                misclassifications.append({
                    'sentence_idx': i,
                    'token_idx': j,
                    'token': tok,
                    'true': t,
                    'pred': p,
                    'context': ' '.join(tokens[max(0, j-2):min(len(tokens), j+3)])
                })

    print("Classification Report:")
    print(classification_report(all_true_labels, all_pred_labels, digits=4))
    
    print("\nConfusion Matrix:")
    labels_order = ['FIL', 'ENG', 'CS', 'OTH']
    cm = confusion_matrix(all_true_labels, all_pred_labels, labels=labels_order)
    print(labels_order)
    print(cm)
    
    print("\nMost common misclassifications:")
    from collections import Counter
    misclass_pairs = [f"True: {m['true']} -> Pred: {m['pred']}" for m in misclassifications]
    counter = Counter(misclass_pairs)
    for pair, count in counter.most_common(10):
        print(f"{pair}: {count}")
        example = next(m for m in misclassifications if f"True: {m['true']} -> Pred: {m['pred']}" == pair)
        print(f"   Example: Token '{example['token']}' in context '... {example['context']} ...'\n")
        
    print("\nDetailed breakdown per class:")
    precision, recall, fscore, support = precision_recall_fscore_support(all_true_labels, all_pred_labels, labels=labels_order)
    for idx, label in enumerate(labels_order):
        tp = cm[idx, idx]
        fp = cm[:, idx].sum() - tp
        fn = cm[idx, :].sum() - tp
        print(f"Category: {label}")
        print(f"  TP: {tp}, FP: {fp}, FN: {fn}")
        print(f"  Precision: {precision[idx]:.4f}")
        print(f"  Recall: {recall[idx]:.4f}")
        print(f"  F1-Score: {fscore[idx]:.4f}")

if __name__ == '__main__':
    evaluate()
