from quantulum3 import parser   # must have installed numpy, scipy, and sklearn
import spacy    # install en_core_web_sm


nlp = spacy.load('en_core_web_sm')

with open('data/text/S0012821X12004384-952.txt', 'r') as file:
    text = file.read()

doc = nlp(text)

target_sentences = []

for sentence in doc.sents:
    for token in sentence:
        if 'NUM' in token.pos_:
            target_sentences.append(sentence)
            break

print(target_sentences)


# print(token.text)
# print(token.text, token.idx, token.ent_type_, token.pos_, token.dep_)

# spacy.displacy.serve(list(doc.sents), style='ent')

# print(' '.join([t.text for t in token.subtree]))


# a = parser.parse('The ruler is 4 meters per hour long.')
#
# for x in range(len(a)):
#     print(a[x].surface, a[x].span, a[x].unit)
