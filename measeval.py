import glob
import re
from collections import Counter

import pandas as pd
import sklearn_crfsuite
import spacy
from quantities import units as u
from sklearn_crfsuite import metrics


def get_text_data(text_files):
    """Return dataframe of text files"""

    docs = {'docId': [], 'sent': [], 'ents': [], 'nounPhrases': []}

    for t_file in text_files:
        with open(t_file, 'r') as f:
            doc_id = re.findall(r'txt/(.*).txt', f.name)[0]
            text = nlp(f.read())

            for sent in text.sents:
                if 'CD' in [word.tag_ for word in sent]:
                    docs['docId'].append(doc_id)
                    docs['sent'].append(sent)

                    ents = []

                    for ent in sent.ents:
                        token_span = (ent.start, ent.end)
                        ent_type = (ent.label_, token_span)

                        ents.append(ent_type)

                    docs['ents'].append(ents)

                    noun_phrases = []

                    for nc in sent.noun_chunks:
                        token_span = (nc.start, nc.end)

                        noun_phrases.append((nc.text, token_span))

                    docs['nounPhrases'].append(noun_phrases)

    doc_df = pd.DataFrame(docs, columns=['docId', 'sent', 'ents', 'nounPhrases'])

    return doc_df


def is_unit(token):
    """Check if token is in units list"""

    if 'μ' in token.text:
        token.text.replace('μ', 'u')

    return token.lower_ in units or token.lemma_ in units


def get_labels(doc_df, annot_df):
    """Return matched up labels from annot_df to text"""

    labels = {'labels': []}

    for _, t_row in doc_df.iterrows():
        labeled_spans = [['O', (word.idx, (word.idx + len(word)))] for word in t_row['sent']]
        match_doc_id = annot_df['docId'] == t_row['docId']

        for _, a_row in annot_df[match_doc_id].iterrows():
            if a_row['annotType'] == 'Qualifier':
                continue

            start, end = a_row['startOffset'], a_row['endOffset']
            annot_type = a_row['annotType'].lower()

            if annot_type == tags['QNT'].lower():
                annot_type = 'QNT'
            elif annot_type == tags['MSE'].lower():
                annot_type = 'MSE'
            elif annot_type == tags['MSP'].lower():
                annot_type = 'MSP'

            for i, item in enumerate(labeled_spans):
                if item[0] != 'O':
                    continue

                span = (item[1][0], item[1][1])

                if (start <= span[0] < end) or (start < span[1] <= end):
                    labeled_spans[i][0] = annot_type

        labels['labels'].append([label for label, span in labeled_spans])

    return labels['labels']


def word2features(sent, ents, nouns, i):
    """Generate features from words"""

    word = sent[i]

    features = {
        'bias': 1.0,
        'word.lower': word.lower_,
        'word.lemma': word.lemma_,
        'word.is_upper': word.is_upper,
        'word.is_title': word.is_title,
        'word.is_digit': word.is_digit,
        'word.like_num': word.like_num,
        'word.is_unit': is_unit(word),
        'postag': word.tag_,
        'dep': word.dep_
    }

    for ent in ents:
        tok_span = ent[1]

        if tok_span[0] <= word.i < tok_span[1]:
            features['word.is_ent'] = ent[0]
            break

    for noun in nouns:
        tok_span = noun[1]

        if tok_span[0] <= word.i < tok_span[1]:
            features['word.is_noun_phrase'] = list(noun[0])
            break

    if i > 0:
        prev_word = word.nbor(-1)
        features.update({
            '-1:word.lower': prev_word.lower_,
            '-1:word.lemma': prev_word.lemma_,
            '-1:word.is_title': prev_word.is_title,
            '-1:word.is_upper': prev_word.is_upper,
            '-1:word.is_digit': prev_word.is_digit,
            '-1:word.like_num': prev_word.like_num,
            '-1:word.is_unit': is_unit(prev_word),
            '-1:postag': prev_word.tag_,
            '-1:dep': prev_word.dep_
        })
    else:
        features['BOS'] = True

    if i < len(sent) - 1:
        next_word = word.nbor()
        features.update({
            '+1:word.lower': next_word.lower_,
            '+1:word.lemma': next_word.lemma_,
            '+1:word.is_title': next_word.is_title,
            '+1:word.is_upper': next_word.is_upper,
            '+1:word.is_digit': next_word.is_digit,
            '+1:word.like_num': next_word.like_num,
            '+1:word.is_unit': is_unit(next_word),
            '+1:postag': next_word.tag_,
            '+1:dep': next_word.dep_
        })
    else:
        features['EOS'] = True

    window_2 = {'lower': [], 'lemma': [], 'pos': [], 'dep': []}
    window_4 = {'lower': [], 'lemma': [], 'pos': [], 'dep': []}
    for x in range(-4, 5):
        if x == 0 or (i + x < 0) or (i + x > len(sent) - 1):
            continue

        window_word = word.nbor(x)

        if x in range(-2, 3):
            window_2['lower'].append(window_word.lower_)
            window_2['pos'].append(window_word.pos_)
            window_2['dep'].append(window_word.dep_)

        window_4['lower'].append(window_word.lower_)
        window_4['pos'].append(window_word.pos_)
        window_4['dep'].append(window_word.dep_)

    features.update({
            'window_2_words:lower': window_2['lower'],
            'window_2_words:pos': window_2['pos'],
            'window_2_words:dep': window_2['dep'],
            'window_4_words:lower': window_4['lower'],
            'window_4_words:pos': window_4['pos'],
            'window_4_words:dep': window_4['dep']
    })

    return features


def sent2features(sent, ents, nouns):
    """Generate features for each sentence"""
    return [word2features(sent, ents, nouns, i) for i in range(len(sent))]


def get_mismatched_metrics(actual, prediction):
    """Get the metrics for the labels which were mismatched"""
    mismatch = {
            'O -> QNT': 0, 'O -> MSE': 0, 'O -> MSP': 0,
            'QNT -> O': 0, 'MSE -> O': 0, 'MSP -> O': 0,
            'QNT -> MSE': 0, 'QNT -> MSP': 0,
            'MSE -> QNT': 0, 'MSE -> MSP': 0,
            'MSP -> QNT': 0, 'MSP -> MSE': 0
    }

    labels = list(zip(actual, prediction))

    for act, pred in labels:
        for idx in range(len(act)):
            if act[idx] == 'O' and pred[idx] == 'QNT':
                mismatch['O -> QNT'] += 1
            elif act[idx] == 'O' and pred[idx] == 'MSE':
                mismatch['O -> MSE'] += 1
            elif act[idx] == 'O' and pred[idx] == 'MSP':
                mismatch['O -> MSP'] += 1
            elif act[idx] == 'QNT' and pred[idx] == 'O':
                mismatch['QNT -> O'] += 1
            elif act[idx] == 'MSE' and pred[idx] == 'O':
                mismatch['MSE -> O'] += 1
            elif act[idx] == 'MSP' and pred[idx] == 'O':
                mismatch['MSP -> O'] += 1
            elif act[idx] == 'QNT' and pred[idx] == 'MSE':
                mismatch['QNT -> MSE'] += 1
            elif act[idx] == 'QNT' and pred[idx] == 'MSP':
                mismatch['QNT -> MSP'] += 1
            elif act[idx] == 'MSE' and pred[idx] == 'QNT':
                mismatch['MSE -> QNT'] += 1
            elif act[idx] == 'MSE' and pred[idx] == 'MSP':
                mismatch['MSE -> MSP'] += 1
            elif act[idx] == 'MSP' and pred[idx] == 'QNT':
                mismatch['MSP -> QNT'] += 1
            elif act[idx] == 'MSP' and pred[idx] == 'MSE':
                mismatch['MSP -> MSE'] += 1

            idx += 1

    return mismatch


def print_state_features(state_features):
    """Print state features"""
    for (attr, label), weight in state_features:
        print("%0.6f %-8s %s" % (weight, label, attr))


# *************** MAIN ****************

tags = {'QNT': 'Quantity', 'MSE': 'MeasuredEntity', 'MSP': 'MeasuredProperty'}

nlp = spacy.load('en_core_web_sm')

units = ['%']
for key, val in u.__dict__.items():
    if isinstance(val, type(u.l)):
        if key not in units and key.lower() not in nlp.Defaults.stop_words:
            units.append(key.lower())

        if val.name not in units and val.name.lower() not in nlp.Defaults.stop_words:
            units.append(val.name.lower())


# ------------ Getting data for train and test files ------------
train_text_files = glob.glob('data/train/txt/*.txt')
train_doc_df = get_text_data(train_text_files)

train_annot_files = glob.glob('data/train/tsv/*.tsv')
train_annot_df = pd.concat([pd.read_csv(a_file, sep='\t', header=0) for a_file in train_annot_files])

test_text_files = glob.glob('data/test/txt/*.txt')
test_doc_df = get_text_data(test_text_files)

test_annot_files = glob.glob('data/test/tsv/*.tsv')
test_annot_df = pd.concat([pd.read_csv(a_file, sep='\t', header=0) for a_file in test_annot_files])


# ------------ Get features and labels ------------
X_train = [sent2features(row['sent'], row['ents'], row['nounPhrases']) for _, row in train_doc_df.iterrows()]
y_train = get_labels(train_doc_df, train_annot_df)

X_test = [sent2features(row['sent'], row['ents'], row['nounPhrases']) for _, row in test_doc_df.iterrows()]
y_test = get_labels(test_doc_df, test_annot_df)


# ------------ Training ------------
crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True
)

crf.fit(X_train, y_train)


# ------------ Evaluation ------------
labels = list(crf.classes_)
labels.remove('O')

y_pred = crf.predict(X_test)

sorted_labels = sorted(
    labels,
    key=lambda name: (name[1:], name[0])
)

print(metrics.flat_classification_report(
    y_test, y_pred, labels=sorted_labels, digits=3
))

# mismatched = get_mismatched_metrics(y_test, y_pred)
#
# for k, v in mismatched.items():
#     print(k + ":", v)

# print("Top positive:")
# print_state_features(Counter(crf.state_features_).most_common(30))
#
# print("\nTop negative:")
# print_state_features(Counter(crf.state_features_).most_common()[-30:])

