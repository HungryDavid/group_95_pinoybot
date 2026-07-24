import pandas as pd
import re

df = pd.read_excel('Shared Dataset.xlsx')

issues = []

# Valid labels
valid_labels = {'ENG', 'FIL', 'CS', 'OTH'}

def is_punctuation_or_number(word):
    if pd.isna(word):
        return False
    word_str = str(word)
    # Check if string contains only punctuation or digits
    return bool(re.match(r'^[\W\d_]+$', word_str))

for index, row in df.iterrows():
    word_id = row['word_id']
    word = row['word']
    lang = row['lang']
    row_num = index + 2 # Excel row is 1-indexed, and header is row 1
    
    # Check for blank
    if pd.isna(lang) or str(lang).strip() == '':
        issues.append({
            'row': row_num,
            'word': str(word),
            'original': '[BLANK]',
            'suggested': 'Needs manual classification',
            'reason': 'Blank classification'
        })
        continue
    
    lang_str = str(lang)
    
    # Check for invalid tag or leading/trailing spaces
    if lang_str not in valid_labels:
        suggested = lang_str.strip().upper()
        if suggested in valid_labels:
            issues.append({
                'row': row_num,
                'word': str(word),
                'original': f"'{lang_str}'",
                'suggested': suggested,
                'reason': 'Typo or extra spaces in tag'
            })
        else:
            issues.append({
                'row': row_num,
                'word': str(word),
                'original': f"'{lang_str}'",
                'suggested': 'ENG, FIL, CS, or OTH',
                'reason': 'Invalid tag'
            })
        continue
    
    # Check for punctuation or number that is not OTH
    if is_punctuation_or_number(word) and lang_str != 'OTH':
        issues.append({
            'row': row_num,
            'word': str(word),
            'original': lang_str,
            'suggested': 'OTH',
            'reason': 'Punctuation or number should be OTH'
        })

# Now write to markdown
with open(r'C:\Users\Matthew Fabregas\.gemini\antigravity-ide\brain\47c8c49a-5e6b-4c68-bf0a-af99802afa19\dataset_review.md', 'w', encoding='utf-8') as f:
    f.write("# Dataset Review\n\n")
    f.write("Here are the rows that might need your review based on the project specifications:\n\n")
    f.write("| Excel Row | Word | Original Tag | Suggested Tag | Reason |\n")
    f.write("|---|---|---|---|---|\n")
    
    if not issues:
        f.write("| - | - | - | - | No issues found! |\n")
    else:
        for issue in issues[:1000]:
            # escape pipes for markdown table
            word_safe = issue['word'].replace('|', '\|')
            f.write(f"| {issue['row']} | {word_safe} | {issue['original']} | {issue['suggested']} | {issue['reason']} |\n")
            
        if len(issues) > 1000:
            f.write(f"\n**Note:** Showing first 1000 issues out of {len(issues)} total.\n")
