import pandas as pd
import numpy as np
import random
import os
import re
from datasets import load_dataset
import itertools

# Constants
FAKE_PREPENDS = ["🚨 जरूर शेयर करें!", "⚠️ बहुत जरूरी खबर", "👆 सच्ची खबर", "ਜ਼ਰੂਰ ਸ਼ੇਅਰ ਕਰੋ!", "ਬਹੁਤ ਜ਼ਰੂਰੀ ਖਬਰ"]
FAKE_APPENDS = ["सभी को भेजें", "आगे फॉरवर्ड करें", "ਅੱਗੇ ਭੇਜੋ"]
EMOJIS = ["🚨", "⚠️", "😱", "🔴", "💥", "🙏", "😢", "🇮🇳"]
TRUE_PREPENDS = ["क्या आपने सुना?", "ਕੀ ਤੁਸੀਂ ਸੁਣਿਆ?"]
TRUE_SOURCES = ["(Source: ANI)", "(Source: PTI)", "(Source: News18)", "(Source: Wikipedia)"]

def get_sentences(text):
    # Basic sentence tokenization (split by full stop or pipe character often used in Hindi/Punjabi)
    sentences = re.split(r'[।\|\.]', str(text))
    return [s.strip() for s in sentences if s.strip()]

def transform_to_whatsapp_style(text, label, lang_hint="hi"):
    """Transforms an article into a WhatsApp forward style."""
    sentences = get_sentences(text)
    
    if label == 1:  # FAKE
        # Truncate to first 2-3 sentences
        num_sentences = min(len(sentences), random.randint(2, 3))
        truncated_text = "। ".join(sentences[:num_sentences]) + "।" if sentences else ""
        
        # Prepend
        prepend = random.choice(FAKE_PREPENDS)
        
        # Append
        append = random.choice(FAKE_APPENDS)
        
        # Forwarded flag
        forwarded = "Forwarded many times\n" if random.random() < 0.6 else ""
        
        # Emojis
        num_emojis = random.randint(1, 3)
        emoji_str = "".join(random.choices(EMOJIS, k=num_emojis))
        
        final_text = f"{forwarded}{prepend} {truncated_text} {emoji_str}\n\n{append}"
        return final_text, lang_hint
        
    else:  # TRUE_OR_REAL
        # Truncate to 2-4 sentences
        num_sentences = min(len(sentences), random.randint(2, 4))
        truncated_text = "। ".join(sentences[:num_sentences]) + "।" if sentences else ""
        
        # Optional prepend
        prepend = f"{random.choice(TRUE_PREPENDS)}\n" if random.random() < 0.3 else ""
        
        # Optional source
        source = f"\n\n{random.choice(TRUE_SOURCES)}" if random.random() < 0.4 else ""
        
        final_text = f"{prepend}{truncated_text}{source}"
        return final_text, lang_hint

def fetch_hf_articles(lang, count):
    print(f"Fetching {count} articles for {lang} from HuggingFace Wikipedia...")
    dataset_name = f"20231101.{lang}"
    try:
        ds = load_dataset("wikimedia/wikipedia", dataset_name, split="train", streaming=True)
        articles = []
        for item in itertools.islice(ds, count):
            text = item["text"].strip()
            if len(text) > 50: # filter out very short stubs
                articles.append(text)
        return articles
    except Exception as e:
        print(f"Error fetching {lang} articles: {e}")
        return []

def main():
    np.random.seed(42)
    random.seed(42)
    
    base_dir = "/Users/parthsaini/Desktop/FIANL"
    out_dir = os.path.join(base_dir, "whatsapp_fake_news", "data", "whatsapp_dataset")
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Fetch 20,000 Hindi articles and 10,000 Punjabi articles
    hi_articles = fetch_hf_articles("hi", 20000)
    pa_articles = fetch_hf_articles("pa", 10000)
    
    all_data = []
    
    # Process Hindi
    print("Processing Hindi articles...")
    for i, text in enumerate(hi_articles):
        # Even indices -> FAKE (1), Odd -> REAL (0)
        label = 1 if i % 2 == 0 else 0
        wa_text, lang = transform_to_whatsapp_style(text, label, "hi")
        all_data.append({
            "text": wa_text,
            "label": label,
            "language": lang,
            "source_type": "transformed_wikipedia"
        })
        
    # Process Punjabi
    print("Processing Punjabi articles...")
    for i, text in enumerate(pa_articles):
        label = 1 if i % 2 == 0 else 0
        wa_text, lang = transform_to_whatsapp_style(text, label, "pa")
        all_data.append({
            "text": wa_text,
            "label": label,
            "language": lang,
            "source_type": "transformed_wikipedia"
        })
        
    df = pd.DataFrame(all_data)
    
    # Drop empty or tiny texts
    df = df[df['text'].str.len() > 20]
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 5. Split (80% train, 10% valid, 10% test)
    total = len(df)
    train_end = int(total * 0.8)
    valid_end = int(total * 0.9)
    
    train_df = df.iloc[:train_end]
    valid_df = df.iloc[train_end:valid_end]
    test_df = df.iloc[valid_end:]
    
    # Save
    train_df.to_csv(os.path.join(out_dir, "whatsapp_train.csv"), index=False)
    valid_df.to_csv(os.path.join(out_dir, "whatsapp_valid.csv"), index=False)
    test_df.to_csv(os.path.join(out_dir, "whatsapp_test.csv"), index=False)
    
    # Print stats
    print("\n=== Dataset Statistics ===")
    print(f"Total size: {total}")
    print(f"Train: {len(train_df)}, Valid: {len(valid_df)}, Test: {len(test_df)}")
    
    print("\nClass Distribution (Total):")
    print(df['label'].value_counts(normalize=True))
    
    print("\nLanguage Distribution (Total):")
    print(df['language'].value_counts(normalize=True))
    
    print("\nSource Type Distribution (Total):")
    print(df['source_type'].value_counts(normalize=True))
    
    df['text_len'] = df['text'].apply(lambda x: len(str(x)))
    print(f"\nAverage text length (chars): {df['text_len'].mean():.2f}")
    
    print("\nSample Fake Message:")
    sample_fake = df[df['label'] == 1]
    if not sample_fake.empty:
        print(sample_fake.iloc[0]['text'])
    
    print("\nSample Real Message:")
    sample_real = df[df['label'] == 0]
    if not sample_real.empty:
        print(sample_real.iloc[0]['text'])

if __name__ == "__main__":
    main()
