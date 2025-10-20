"""
Tokenization service with support for English and Chinese (Traditional/Simplified)

Requirements:
- jieba: Chinese word segmentation
- opencc-python-reimplemented: Traditional/Simplified Chinese conversion
"""
import re
import string
from typing import List
from functools import lru_cache

import jieba
import opencc

# Initialize jieba
jieba.initialize()

# Initialize OpenCC converters
# Traditional to Simplified (for normalization)
T2S_CONVERTER = opencc.OpenCC('t2s')
# Simplified to Traditional (for output)
S2T_CONVERTER = opencc.OpenCC('s2t')


# Basic Chinese stopwords (commonly used)
# Note: If network allows, you can also use NLTK stopwords:
#   from nltk.corpus import stopwords
#   nltk.download('stopwords', quiet=True)
#   ENGLISH_STOPWORDS = set(stopwords.words('english'))
#   CHINESE_STOPWORDS = set(stopwords.words('chinese'))  # NLTK does support Chinese
CHINESE_STOPWORDS = {
    #繁體
    '的', '了', '和', '是', '就', '都', '而', '及', '與', '著',
    '或', '一個', '沒有', '我們', '你們', '他們', '這個', '那個',
    '什麼', '怎麼', '為什麼', '因為', '所以', '但是', '如果', '已經',
    '可以', '這樣', '那樣', '還是', '不是', '也是', '會', '要', '說',
    # 簡體
    '的', '了', '和', '是', '就', '都', '而', '及', '与', '着',
    '或', '一个', '没有', '我们', '你们', '他们', '这个', '那个',
    '什么', '怎么', '为什么', '因为', '所以', '但是', '如果', '已经',
    '可以', '这样', '那样', '还是', '不是', '也是', '会', '要', '说',
}

# Basic English stopwords
# Note: If using NLTK, replace with: set(stopwords.words('english'))
ENGLISH_STOPWORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
    'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
    'to', 'was', 'will', 'with', 'the', 'this', 'but', 'they', 'have',
    'had', 'what', 'when', 'where', 'who', 'which', 'why', 'how'
}


class Tokenizer:
    """
    Tokenizer with support for English and Chinese (Traditional/Simplified)

    Features:
    - jieba for Chinese word segmentation
    - opencc for Traditional/Simplified conversion
    - Optional stopword filtering
    - Traditional Chinese as default normalization
    """

    def __init__(
        self,
        filter_stopwords: bool = False,
        normalize_to_traditional: bool = True
    ):
        """
        Initialize tokenizer

        Args:
            filter_stopwords: Remove common stopwords
            normalize_to_traditional: Normalize to Traditional Chinese (default: True)
        """
        self.filter_stopwords = filter_stopwords
        self.normalize_to_traditional = normalize_to_traditional

        # Build stopwords set
        self.stopwords = CHINESE_STOPWORDS | ENGLISH_STOPWORDS if filter_stopwords else set()

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text with support for mixed English and Chinese

        Args:
            text: Input text

        Returns:
            List of tokens
        """
        if not text:
            return []

        # Lowercase for English words
        text = text.lower()

        # Use jieba to cut the text
        tokens = jieba.lcut(text)

        # Filter out pure punctuation and whitespace
        tokens = [
            token.strip() for token in tokens
            if token.strip() and not self._is_only_punctuation_and_whitespace(token)
        ]

        # Normalize Chinese characters
        tokens = self._normalize_chinese(tokens)

        # Filter stopwords if enabled
        if self.filter_stopwords:
            tokens = self._filter_stopwords(tokens)

        return tokens

    def _normalize_chinese(self, tokens: List[str]) -> List[str]:
        """
        Normalize Chinese tokens to Traditional or Simplified
        """
        normalized = []
        for token in tokens:
            # Check if token contains Chinese characters
            if re.search(r'[\u4e00-\u9fff]', token):
                if self.normalize_to_traditional:
                    # Convert any simplified to traditional
                    token = S2T_CONVERTER.convert(token)
                else:
                    # Convert any traditional to simplified
                    token = T2S_CONVERTER.convert(token)
            normalized.append(token)

        return normalized

    def _filter_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Filter out stopwords
        """
        return [token for token in tokens if token not in self.stopwords]

    @staticmethod
    def _is_only_punctuation_and_whitespace(s: str) -> bool:
        """
        Check if string contains only punctuation and whitespace
        """
        pattern = string.punctuation + string.whitespace + '，。、；：？！""''（）【】《》…—·'
        return all(char in pattern for char in s)


# Create default tokenizer instances
# Default tokenizer (Traditional Chinese, no stopword filtering)
default_tokenizer = Tokenizer(
    filter_stopwords=False,
    normalize_to_traditional=True
)

# Tokenizer with stopword filtering
tokenizer_with_stopwords = Tokenizer(
    filter_stopwords=True,
    normalize_to_traditional=True
)


def get_tokenizer(filter_stopwords: bool = False) -> Tokenizer:
    """
    Factory function to get tokenizer

    Args:
        filter_stopwords: Whether to filter out stopwords

    Returns:
        Tokenizer instance
    """
    if filter_stopwords:
        return tokenizer_with_stopwords
    return default_tokenizer

