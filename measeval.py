import glob
import re
import pandas as pd
import scipy.stats
import spacy
from quantities import units as u
import sklearn
import sklearn_crfsuite
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV
from sklearn_crfsuite import metrics


def get_text_data(text_files):
    """Return dataframe of text files"""

    docs = {'docId': [], 'sent': []}

    for t_file in text_files:
        with open(t_file, 'r') as f:
            doc_id = re.findall(r'txt/(.*).txt', f.name)[0]
            text = nlp(f.read())

            for sent in text.sents:
                if 'CD' in [word.tag_ for word in sent]:
                    docs['docId'].append(doc_id)
                    docs['sent'].append(sent)

    doc_df = pd.DataFrame(docs, columns=['docId', 'sent'])

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


def word2features(sent, i):
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

    return features


def sent2features(sent):
    """Generate features for each sentence"""
    return [word2features(sent, i) for i in range(len(sent))]


tags = {'QNT': 'Quantity', 'MSE': 'MeasuredEntity', 'MSP': 'MeasuredProperty'}

nlp = spacy.load('en_core_web_sm')

units = ['%']
for key, val in u.__dict__.items():
    if isinstance(val, type(u.l)):
        if key not in units and key.lower() not in nlp.Defaults.stop_words:
            units.append(key.lower())

        if val.name not in units and val.name.lower() not in nlp.Defaults.stop_words:
            units.append(val.name.lower())

pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 500)


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
X_train = [sent2features(sent) for sent in train_doc_df['sent']]
y_train = get_labels(train_doc_df, train_annot_df)

X_test = [sent2features(sent) for sent in test_doc_df['sent']]
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
print(metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=labels))

sorted_labels = sorted(
    labels,
    key=lambda name: (name[1:], name[0])
)

print(metrics.flat_classification_report(
    y_test, y_pred, labels=sorted_labels, digits=3
))

crf = sklearn_crfsuite.CRF(
    algorithm='pa',
    max_iterations=100,
    all_possible_transitions=True
)

crf.fit(X_train, y_train)

y_pred = crf.predict(X_test)
print(metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=labels))
print(metrics.flat_classification_report(
    y_test, y_pred, labels=sorted_labels, digits=3
))

# params_space = {
#     'c1': scipy.stats.expon(scale=0.5),
#     'c2': scipy.stats.expon(scale=0.05),
# }
#
# # use the same metric for evaluation
# f1_scorer = make_scorer(metrics.flat_f1_score,
#                         average='weighted', labels=labels)
#
# # search
# rs = RandomizedSearchCV(crf, params_space,
#                         cv=3,
#                         verbose=1,
#                         n_jobs=-1,
#                         n_iter=50,
#                         scoring=f1_scorer)
# rs.fit(X_train, y_train)
#
# print('best params:', rs.best_params_)
# print('best CV score:', rs.best_score_)
# print('model size: {:0.2f}M'.format(rs.best_estimator_.size_ / 1000000))

