def tokenize(message, stem='lemm'):
    """Text processing.
    
    Args:
        message(str): Message content.
        stem(str): stem or lemm.
        
    Returns:
        list: Cleaned tokens.
    """
    # 1. Cleaning

    # 2. Normalization
    text = re.sub(r"[^a-zA-Z0-9]", " ", message.lower())

    # 3. Tokenization
    tokens = word_tokenize(text)

    # 4. Stop Word Removal
    stop_words = stopwords.words("english")
    tokens = list(filter(lambda w: w not in stop_words, tokens))

    # 5. Part of Speech Tagging / Named Entity Recognition

    # 6. Stemming or Lemmatization
    # Because the targets are not roots, we should use Lemmatization

    clean_tokens = []
    if stem == 'stem':
        stemmer = PorterStemmer()
        for tok in tokens:
            clean_tok = stemmer.stem(tok).strip()
            clean_tokens.append(clean_tok)
    else:
        lemmatizer = WordNetLemmatizer()
        for tok in tokens:
            clean_tok = lemmatizer.lemmatize(tok).strip()
            clean_tokens.append(clean_tok)

    return clean_tokens
