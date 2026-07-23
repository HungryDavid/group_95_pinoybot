import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report
from pinoybot import extract_features

def train_model(excel_file_path: str, model_output_path: str = 'model.pkl', vectorizer_output_path: str = 'vectorizer.pkl'):
    df = pd.read_excel(excel_file_path, engine='openpyxl')
    
    df.columns = df.columns.str.strip() # strip whitespace from column names
    df = df.dropna(subset=['word', 'lang', 'sentence_id']) # get rid of rows with missing values in critical columns
    df['word'] = df['word'].astype(str)
    df['lang'] = df['lang'].astype(str).str.strip().str.upper()
    df['sentence_id'] = df['sentence_id'].astype(float)
    
    valid_labels = ['FIL', 'ENG', 'CS', 'OTH']
    df = df[df['lang'].isin(valid_labels)]

    sentences_map = {}
    for sid, group in df.groupby('sentence_id', sort=False): # group by sentence_id while preserving the original order
        sentences_map[sid] = {
            'tokens': group['word'].tolist(),
            'labels': group['lang'].tolist()
        }

    sids = list(sentences_map.keys()) # map by sentence so we can split them by sentence and not by words
    sids_train, sids_temp = train_test_split(sids, test_size=0.3, random_state=42) # split the map so train gets 70% and temp gets 30%
    sids_val, sids_test = train_test_split(sids_temp, test_size=0.5, random_state=42) # split temp so val gets 15% and test gets 15%

    def build_xy(sid_list): # function to create feature matrix
        X, y = [], []
        for sid in sid_list:
            data = sentences_map[sid]
            for i in range(len(data['tokens'])):
                X.append(extract_features(data['tokens'], i)) 
                y.append(data['labels'][i])
        return X, y
    # create feature dictionaries and labels for training, validation, and testing sets
    X_train_dicts, y_train = build_xy(sids_train) 
    X_val_dicts, y_val = build_xy(sids_val)
    X_test_dicts, y_test = build_xy(sids_test)

    # vectorize the feature dictionaries into a sparse matrix format suitable for model training 
    vectorizer = DictVectorizer(sparse=True)
    X_train = vectorizer.fit_transform(X_train_dicts)
    X_val = vectorizer.transform(X_val_dicts)
    X_test = vectorizer.transform(X_test_dicts)

    # initializes the model and trains it using the 70% training features and labels.
    # here we use a VotingClassifier that combines a Decision Tree and a Multinomial Naive Bayes classifier, with the Naive Bayes given more weight in the voting process.
    dt_clf = DecisionTreeClassifier(class_weight='balanced', random_state=42)
    nb_clf = MultinomialNB(alpha=0.5)
    clf = VotingClassifier(
        estimators=[
            ('dt', dt_clf), 
            ('nb', nb_clf)
        ],
        voting='soft',
        weights=[1, 2]
    )
    clf.fit(X_train, y_train)

    y_pred_val = clf.predict(X_val)
    print("\nValidation Classification Report:")
    print(classification_report(y_val, y_pred_val, zero_division=0))

    y_pred_test = clf.predict(X_test)
    print("\nTest Classification Report:")
    print(classification_report(y_test, y_pred_test, zero_division=0))

    joblib.dump(vectorizer, vectorizer_output_path)
    joblib.dump(clf, model_output_path)
    joblib.dump(sentences_map, 'sentences_map.pkl')
    print(f"Model successfully saved to {model_output_path}")
    print(f"Vectorizer successfully saved to {vectorizer_output_path}")

if __name__ == "__main__":
    excel_path = "Shared Dataset.xlsx"
    if not os.path.exists(excel_path):
        excel_files = [f for f in os.listdir('.') if f.endswith('.xlsx') or f.endswith('.xls')]
        if excel_files:
            excel_path = excel_files[0]

    if os.path.exists(excel_path):
        train_model(excel_path)
    else:
        print(f"Error: Dataset file not found.")