import spacy
import collections
import pandas as pd
import nltk
from nltk import pos_tag, word_tokenize, RegexpParser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# Features: NER and Part-of-Speech related features from SpaCy.

nlp = spacy.load('en_core_web_sm')

#Helper function 1
def list_count(lst):
    """
    :function: count the elements of a list -- the number of words with a respective POS or NER labels in a sentence.
    :input: lst: list of tuples, where tuple has two elements -- a word and its POS or NER label
    :return: lst_count: list of dictionaries, where
    the dictionary consists of keys -- the elements are words and their POS or NER labels
    and values -- how many times each word and its POS or NER label occurs
    If a sentence has no POS or NER labels, return an empty list

    """

    dic_counter = collections.Counter()

    for x in lst:
        dic_counter[x] += 1

    dic_counter = collections.OrderedDict(
        sorted(dic_counter.items(),
               key=lambda x: x[1], reverse=True))

    lst_count = [{key: value} for key, value in dic_counter.items()]

    return lst_count


#Helper function 2
def column_tag(lst_dics_tuples, tag):
    """
    :function: new column for each POS or NER tag category
    :input: lst_dics_tuples: list of dictionaries with tuples
            tag: POS or NER label from a list
    :return: tag: new column for each POS or NER label with their counts

    """

    if len(lst_dics_tuples) > 0:
        tag_type = []

        for dic_tuples in lst_dics_tuples:
            for tuple in dic_tuples:
                type, n = tuple[1], dic_tuples[tuple]
                tag_type = tag_type + [type] * n
                dic_counter = collections.Counter()
                for x in tag_type:
                    dic_counter[x] += 1
        return dic_counter[tag]

    else:
        return 0


# Feature 1. Part-of-Speech for adverbs and adjectives
def pos_features(df, speech_sents):
    """
    :function: add two new columns with two POS: adjectives and adverbs, and their counts per sentence.
    Two helper functions -- list_count, column_tag -- are needed
    :input: df: entire DataFrame
            speech_sents: Series of sentences in DataFrame
    :return: df: new DataFrame with two new features

    """

    df['pos'] = speech_sents.apply(lambda x: [(tag.text, tag.pos_)
                                              for tag in nlp(x)])

    df['pos'] = df['pos'].apply(lambda x: list_count(x))

    # extract features
    tags_set = []

    for lst in df['pos'].tolist():
        for dic in lst:
            for k in dic.keys():
                tags_set.append(k[1])

    tags_set = list(set(tags_set))

    for feature in tags_set:
        df['pos_' + feature] = df['pos'].apply(lambda x: column_tag(x, feature))

    # keeping only adverbs and adjectives and dropping other pos
    for feature in df.columns:
        if feature != 'pos_ADV' and feature != 'pos_ADJ' and 'pos' in feature:
            df = df.drop(feature, axis=1)

    return df


# Feature 2. Tenses for verbs, modal verbs
def verbs_features(df, speech_sents):
    """
    :function: add several new columns with features for verb tenses and the presence of modal verbs,
    and their counts per sentence.
    Two helper functions -- list_count, column_tag -- are needed
    :input: df: entire DataFrame
            speech_sents: Series of sentences in DataFrame
    :return: df: new DataFrame with features for each verb tense and for modal verbs

    """

    df['verb_tag'] = speech_sents.apply(lambda x: [(tag.text, tag.tag_)
                                                   for tag in nlp(x)])

    df['verb_tag'] = df['verb_tag'].apply(lambda x: list_count(x))

    # extract features
    verbs_set = []

    for lst in df['verb_tag'].tolist():
        for dic in lst:
            for k in dic.keys():
                verbs_set.append(k[1])

    verbs_set = list(set(verbs_set))

    for feature in verbs_set:
        df['verb_tag_' + feature] = df['verb_tag'].apply(lambda x: column_tag(x, feature))

    # out of all detailed POS tags, keeping only verbs-related ones
    for f in df.columns:
        if f != 'verb_tag_VB' and f != 'verb_tag_VBZ' and f != 'verb_tag_VBP' and f != 'verb_tag_VBD' and f != 'verb_tag_VBN' and f != 'verb_tag_VBG' and f != 'verb_tag_MD' and 'verb_tag' in f:
            df = df.drop(f, axis=1)

    return df


# Feature 3. NER features
def ner_features(df, speech_sents):
    """
    :function: add several new columns with NER labels, and their counts per sentence.
    Two helper functions -- list_count, column_tag -- are needed
    :input: df: entire DataFrame
            speech_sents: Series of sentences in DataFrame
    :return: df: new DataFrame with new features for each NER label

    """

    df['ner'] = speech_sents.apply(lambda x: [(tag.text, tag.label_)
                                              for tag in nlp(x).ents])
    # count tags
    df['ner'] = df['ner'].apply(lambda x: list_count(x))

    # extract features
    tags_set = []

    for lst in df['ner'].tolist():
        for dic in lst:
            for k in dic.keys():
                tags_set.append(k[1])

    tags_set = list(set(tags_set))

    for feature in tags_set:
        df['ner_' + feature] = df['ner'].apply(lambda x: column_tag(x, feature))

    df = df.drop(['ner'], axis=1)

    return df

# Feature 4. Syntactic features

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def syntactic_features(df, speech_sents):
    """
    :function: add syntactic features -- 1) the number of productions, 2) the number of VP groups per sentence,
    and 3) the depth of a sentence tree
    :input: df: entire DataFrame
            speech_sents: Series of sentences in DataFrame
    :return: df: new DataFrame with three syntactic features

    """

    a, b, c, d, e = [], [], [], [], []
    for x, y in enumerate(speech_sents):
        tagged = pos_tag(word_tokenize(y))
        chunker = RegexpParser(r"""
            NBAR:
            {<NN.*|JJ>*<NN.*>}  
            VP:
            {<V.*>}  
            NP:
            {<NBAR>}
            {<NBAR><IN><NBAR>}  
        """)

        a.append(chunker.parse(tagged))
        b.append(len(chunker.parse(tagged).productions()))
        e.append(chunker.parse(tagged).productions())
        c.append(chunker.parse(tagged).height())

    df.loc[:, 'Speech_parsed'] = a
    df.loc[:, 'Productions_count'] = b
    df.loc[:, 'Tree_depth'] = c

    for i in e:
        vp = []
        for u in i:
            if str(u).startswith('VP'):
                vp.append(u)
        d.append(len(vp))

    df.loc[:, 'VP_count'] = d

    df = df.drop(['Speech_parsed'], axis=1)

    return df

# Feature 5. Discourse connectives

def add_connectives(df, speech_sents):
    """
    :function: add a boolean feature based on presence/absence of a claim connective from the pre-defined list
    :input: df: entire DataFrame
            speech_sents: Series of sentences in DataFrame
    :return: df: DataFrame with a new feature Claim_Connective

    """

    connectives = ['so that', 'as a result', 'therefore', 'thus', 'thereby', 'in the end', 'hence', 'accordingly',
                   'in this way']
    lst = []

    for sent in speech_sents:
        if any(w in sent for w in connectives):
            lst.append(1)
        else:
            lst.append(0)
    df['Claim_Connective'] = lst

    return df


# Feature 6. Semantic features: sentiment of a sentence

def add_sentiment(df, speech_sents):
    """
    :function: add a feature with a sentiment score for each sentence
    :input: df: entire DataFrame
            speech_sents: Series of sentences in DataFrame
    :return: df: DataFrame with a new feature Sentiment

    """

    analyzer = SentimentIntensityAnalyzer()

    senti = []

    for sent in speech_sents:
        vs = analyzer.polarity_scores(sent)
        senti.append([list(vs.values())[3]])

    senti_arr = np.array(senti)
    df['Sentiment'] = senti_arr

    return df

# Feature 7. Personal pronouns

def add_personal(df, speech_sents):
    """
    :function: add two boolean features based on the presence/absence of any pronoun from two given lists.
    :input: df: entire DataFrame
            speech_sents: Series of sentences in DataFrame
    :return: df: DataFrame with two new features Pronoun_Singular and Pronoun_Plural

    """

    singular = [' i ', ' me ', ' my ', ' myself ', ' mine ']
    plural = [' we ', ' our ', ' ours ', ' ourselves ']
    lst_sing = []
    lst_plur = []

    for sent in speech_sents:
        if any(w in sent for w in singular):
            lst_sing.append(1)
        else:
            lst_sing.append(0)
    df['Pronoun_Singular'] = lst_sing

    for sent in speech_sents:
        if any(w in sent for w in plural):
            lst_plur.append(1)
        else:
            lst_plur.append(0)
    df['Pronoun_Plural'] = lst_plur

    return df

