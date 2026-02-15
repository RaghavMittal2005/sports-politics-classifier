import json
from collections import Counter
from feature_extraction import FeatureExtractor

def analyze_dataset():
    with open('dataset.json','r') as f:
        data=json.load(f)
    
    sports_texts=[]
    politics_texts=[]
    
    for item in data:
        if item['label']=='sports':
            sports_texts.append(item['text'])
        else:
            politics_texts.append(item['text'])
    
    print("="*60)
    print("Dataset Analysis")
    print("="*60)
    
    print("\n1. CLASS DISTRIBUTION")
    print(f"Total samples: {len(data)}")
    print(f"Sports samples: {len(sports_texts)}")
    print(f"Politics samples: {len(politics_texts)}")
    print(f"Balance: {len(sports_texts)/len(data)*100:.1f}% sports, {len(politics_texts)/len(data)*100:.1f}% politics")
    
    extractor=FeatureExtractor()
    extractor.build_vocabulary(sports_texts+politics_texts)
    
    print("\n2. VOCABULARY")
    print(f"Total unique words: {len(extractor.vocabulary)}")
    
    sports_words=set()
    politics_words=set()
    
    for text in sports_texts:
        tokens=extractor.tokenize(text)
        sports_words.update(tokens)
    
    for text in politics_texts:
        tokens=extractor.tokenize(text)
        politics_words.update(tokens)
    
    sports_only=sports_words-politics_words
    politics_only=politics_words-sports_words
    common=sports_words.intersection(politics_words)
    
    print(f"Sports-only words: {len(sports_only)}")
    print(f"Politics-only words: {len(politics_only)}")
    print(f"Common words: {len(common)}")
    
    print("\n3. DOCUMENT LENGTH")
    sports_lengths=[len(extractor.tokenize(t)) for t in sports_texts]
    politics_lengths=[len(extractor.tokenize(t)) for t in politics_texts]
    
    print(f"Sports - Avg: {sum(sports_lengths)/len(sports_lengths):.1f} words, Min: {min(sports_lengths)}, Max: {max(sports_lengths)}")
    print(f"Politics - Avg: {sum(politics_lengths)/len(politics_lengths):.1f} words, Min: {min(politics_lengths)}, Max: {max(politics_lengths)}")
    
    print("\n4. TOP SPORTS WORDS")
    sports_word_freq=Counter()
    for text in sports_texts:
        tokens=extractor.tokenize(text)
        sports_word_freq.update(tokens)
    
    for word,count in sports_word_freq.most_common(15):
        print(f"  {word}: {count}")
    
    print("\n5. TOP POLITICS WORDS")
    politics_word_freq=Counter()
    for text in politics_texts:
        tokens=extractor.tokenize(text)
        politics_word_freq.update(tokens)
    
    for word,count in politics_word_freq.most_common(15):
        print(f"  {word}: {count}")
    
    print("\n6. DISTINCTIVE SPORTS WORDS (sports only)")
    sports_distinctive=[w for w in sports_only if sports_word_freq[w]>=3]
    print(f"  {', '.join(sorted(sports_distinctive)[:20])}")
    
    print("\n7. DISTINCTIVE POLITICS WORDS (politics only)")
    politics_distinctive=[w for w in politics_only if politics_word_freq[w]>=3]
    print(f"  {', '.join(sorted(politics_distinctive)[:20])}")
    
    print("\n8. SAMPLE DOCUMENTS")
    print("\nSports example:")
    print(f"  {sports_texts[0]}")
    print("\nPolitics example:")
    print(f"  {politics_texts[0]}")
    
    print("\n"+"="*60)

if __name__=='__main__':
    analyze_dataset()