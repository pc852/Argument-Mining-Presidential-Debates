import string

def preproc(sentence):

    """
    :function: lowercasing and punctuation removal for input text
    :input: sentence: raw string
    :return: sentence: preprocessed string

    """
    sentence = sentence.lower()
    sentence = ''.join([i for i in sentence if i not in string.punctuation])
    return sentence
