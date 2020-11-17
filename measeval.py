import glob
import re
import pandas as pd
import sklearn
import sklearn_crfsuite


def get_text_data(text_files) -> pd.DataFrame:
    docs = {'docId': [], 'text': []}

    for t_file in text_files:
        with open(t_file, 'r') as f:
            doc_id = re.findall(r'txt/(.*).txt', f.name)[0]
            text = f.read()

            docs['docId'].append(doc_id)
            docs['text'].append(text)

    doc_df = pd.DataFrame(docs, columns=['docId', 'text'])

    return doc_df


pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 500)

train_text_files = glob.glob('data/train/txt/*.txt')
train_doc_df = get_text_data(train_text_files)

train_annot_files = glob.glob('data/train/tsv/*.tsv')
train_annot_df = pd.concat([pd.read_csv(a_file, sep='\t', header=0) for a_file in train_annot_files])

test_text_files = glob.glob('data/test/txt/*.txt')
test_doc_df = get_text_data(test_text_files)

test_annot_files = glob.glob('data/test/tsv/*.tsv')
test_annot_df = pd.concat([pd.read_csv(a_file, sep='\t', header=0) for a_file in test_annot_files])



