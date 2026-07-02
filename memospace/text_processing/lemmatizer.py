"""
词形还原（Lemmatization）模块

提供文本预处理功能，包括：
- 词形还原
- 停用词过滤
- 分词
"""

import re
from typing import List, Set
from dataclasses import dataclass

# 尝试导入可选的 NLTK 库
try:
    import nltk
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import wordnet, stopwords
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    nltk = None
    WordNetLemmatizer = None
    wordnet = None
    stopwords = None

# 停用词列表（内置版本，不依赖 NLTK）
DEFAULT_STOPWORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'as', 'is', 'are', 'was', 'were', 'been',
    'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare', 'ought',
    'used', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her',
    'us', 'them', 'my', 'your', 'his', 'its', 'our', 'their', 'mine', 'yours',
    'hers', 'ours', 'theirs', 'this', 'that', 'these', 'those', 'what', 'which',
    'who', 'whom', 'whose', 'where', 'when', 'why', 'how', 'all', 'each',
    'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
    'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just',
    'also', 'now', 'here', 'there', 'then', 'once', 'if', 'because', 'while',
    'after', 'before', 'since', 'although', 'though', 'yet', 'still', 'even',
    'already', 'always', 'never', 'ever', 'often', 'sometimes', 'usually',
    'about', 'into', 'through', 'during', 'between', 'under', 'over', 'above',
    'below', 'up', 'down', 'out', 'off', 'again', 'further', 'any', 'many',
    'much', 'several', 'less', 'more', 'most', 'least', 'fewest', 'fewer',
    'more', 'most', 'best', 'better', 'worse', 'worst', 'well', 'bad', 'good',
}

# 简单的词形还原规则（不依赖 NLTK）
SIMPLE_LEMMATIZATION_RULES = [
    (r'ies$', 'y'),
    (r'es$', ''),
    (r's$', ''),
    (r'ed$', ''),
    (r'ing$', ''),
    (r'ly$', ''),
    (r'ful$', ''),
    (r'ness$', ''),
]


@dataclass
class ProcessedText:
    """处理后的文本"""
    original: str
    tokens: List[str]
    lemmas: List[str]
    filtered_lemmas: List[str]  # 去掉停用词后的词干


class Lemmatizer:
    """词形还原器"""
    
    def __init__(self, use_nltk: bool = True, custom_stopwords: Set[str] = None):
        """
        初始化词形还原器
        
        Args:
            use_nltk: 是否尝试使用 NLTK（如果可用）
            custom_stopwords: 自定义停用词列表
        """
        self.use_nltk = use_nltk and NLTK_AVAILABLE
        self.custom_stopwords = custom_stopwords or set()
        
        # 初始化停用词
        self.stopwords = set(DEFAULT_STOPWORDS)
        self.stopwords.update(self.custom_stopwords)
        
        # 初始化 NLTK 组件（如果可用）
        self._nltk_initialized = False
        self._lemmatizer = None
        if self.use_nltk:
            self._init_nltk()
    
    def _init_nltk(self):
        """初始化 NLTK 组件"""
        try:
            if not nltk.data.find('corpora/wordnet.zip'):
                nltk.download('wordnet', quiet=True)
            if not nltk.data.find('corpora/stopwords.zip'):
                nltk.download('stopwords', quiet=True)
            if not nltk.data.find('corpora/averaged_perceptron_tagger.zip'):
                nltk.download('averaged_perceptron_tagger', quiet=True)
            
            self._lemmatizer = WordNetLemmatizer()
            
            # 添加 NLTK 的停用词
            nltk_stopwords = set(stopwords.words('english'))
            self.stopwords.update(nltk_stopwords)
            
            self._nltk_initialized = True
        except Exception:
            self.use_nltk = False
            self._nltk_initialized = False
    
    def _get_wordnet_pos(self, tag: str) -> str:
        """将 NLTK POS 标签转换为 WordNet POS 标签"""
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN
    
    def _simple_lemmatize(self, token: str) -> str:
        """简单的词形还原（不依赖 NLTK）"""
        token = token.lower()
        for pattern, replacement in SIMPLE_LEMMATIZATION_RULES:
            if re.search(pattern, token):
                lemma = re.sub(pattern, replacement, token)
                if len(lemma) > 1:  # 确保还原后不是太短
                    return lemma
        return token
    
    def _tokenize(self, text: str) -> List[str]:
        """简单的分词"""
        # 保留字母数字和下划线，转为小写
        text = text.lower()
        tokens = re.findall(r'\b[a-z0-9_]+\b', text)
        return tokens
    
    def process(self, text: str) -> ProcessedText:
        """
        处理文本：分词、词形还原、停用词过滤
        
        Args:
            text: 输入文本
            
        Returns:
            ProcessedText 对象
        """
        tokens = self._tokenize(text)
        
        if self.use_nltk and self._nltk_initialized:
            # 使用 NLTK 进行词形还原
            pos_tags = nltk.pos_tag(tokens)
            lemmas = []
            for token, pos in pos_tags:
                wordnet_pos = self._get_wordnet_pos(pos)
                lemma = self._lemmatizer.lemmatize(token, pos=wordnet_pos)
                lemmas.append(lemma)
        else:
            # 使用简单的词形还原
            lemmas = [self._simple_lemmatize(token) for token in tokens]
        
        # 过滤停用词
        filtered_lemmas = [
            lemma for lemma in lemmas
            if lemma not in self.stopwords and len(lemma) > 1
        ]
        
        return ProcessedText(
            original=text,
            tokens=tokens,
            lemmas=lemmas,
            filtered_lemmas=filtered_lemmas
        )
    
    def process_batch(self, texts: List[str]) -> List[ProcessedText]:
        """
        批量处理文本
        
        Args:
            texts: 文本列表
            
        Returns:
            ProcessedText 列表
        """
        return [self.process(text) for text in texts]
