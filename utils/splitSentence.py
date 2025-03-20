import pyperclip
import pandas as pd
from polyglot.text import Text

def segmentSentence(text):
    """
    :param text: str
    :return: list of str
    """
    sentences = Text(text).sentences
    sentenceList = []
    for sentence in sentences:
        sentenceList.append(sentence)
    return sentenceList
