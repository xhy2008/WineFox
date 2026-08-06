"""Check which punctuation characters are in the Kokoro vocab."""
vocab = {}
with open(r'e:\winefox\voice-test\third_party\kokoro-cpp-src\dict\vocab.txt', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.rstrip('\n')
        tab = line.find('\t')
        if tab != -1:
            token = line[:tab]
            token = token.replace('\\n', '\n').replace('\\r', '\r').replace('\\t', '\t')
            try:
                vocab[token] = int(line[tab+1:])
            except ValueError:
                pass

puncts = [
    ',', '.', '!', '?', ';', ':', '"', '(', ')',
    '—', '…', '“', '”', '~', '·', '•', '-', '–',
    '【', '】', '《', '》', '「', '」', '『', '』',
    '〈', '〉', '〔', '〕', '〖', '〗', '～',
]
for p in puncts:
    if p in vocab:
        print(f'  {repr(p)} (U+{ord(p):04X}) -> id={vocab[p]}  [IN VOCAB]')
    else:
        print(f'  {repr(p)} (U+{ord(p):04X}) -> NOT in vocab  [dropped]')
