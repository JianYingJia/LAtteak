import re

class SplitWord:
    def __init__(self, *args, **kwargs):
        self.sentences = args[0]
        self.file_path = file_path
        self.sentencesList = sentencesList
        if self.sentences is None:
            self.handle_none_optional_param()

    def split_text(text):
        if all('\u4e00' <= char <= '\u9fff' for char in text):  # 判断是否全为中文
            result = [char for char in text]
        else:
            result = text.split()  # 非中文按空格分割
        return result

text1 = "你是谁"
text2 = "i love you\\.."

print(split_text(text1))  # 输出: ['你', '是', '谁']
print(split_text(text2))  # 输出: ['i', 'love', 'you']

