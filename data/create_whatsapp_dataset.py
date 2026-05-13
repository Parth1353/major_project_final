import pandas as pd
import numpy as np
import random
import os
import re

# Constants
FAKE_PREPENDS = ["🚨 जरूर शेयर करें!", "⚠️ बहुत जरूरी खबर", "👆 सच्ची खबर", "ਜ਼ਰੂਰ ਸ਼ੇਅਰ ਕਰੋ!", "ਬਹੁਤ ਜ਼ਰੂਰੀ ਖਬਰ"]
FAKE_APPENDS = ["सभी को भेजें", "आगे फॉरवर्ड करें", "ਅੱਗੇ ਭੇਜੋ"]
EMOJIS = ["🚨", "⚠️", "😱", "🔴", "💥", "🙏", "😢", "🇮🇳"]
TRUE_PREPENDS = ["क्या आपने सुना?", "ਕੀ ਤੁਸੀਂ ਸੁਣਿਆ?"]
TRUE_SOURCES = ["(Source: ANI)", "(Source: PTI)", "(Source: News18)"]

def mock_dataset_if_not_exists(filepath):
    """Generates a mock dataset if the original article dataset is missing for testing."""
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    
    print(f"Original dataset not found at {filepath}. Generating mock data for testing...")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    mock_data = []
    # Mock some FAKE articles
    for _ in range(7500):
        sentences = [
            "प्रधानमंत्री ने आज एक बड़ी घोषणा की है।",
            "अब सभी का कर्ज माफ होगा।",
            "बैंक खाते में 15 लाख आएंगे।",
            "यह खबर बिल्कुल सच है, टीवी पर भी आया है।"
        ]
        text = " ".join(random.sample(sentences, random.randint(2, 4)))
        mock_data.append({"text": text, "label": 1})
    
    # Mock some TRUE articles
    for _ in range(7500):
        sentences = [
            "चुनाव आयोग ने तारीखों का ऐलान कर दिया है।",
            "यह मतदान 5 चरणों में होगा।",
            "मुख्य चुनाव आयुक्त ने प्रेस कॉन्फ्रेंस में यह जानकारी दी।",
            "सभी दलों ने अपनी तैयारियां तेज कर दी हैं।"
        ]
        text = " ".join(random.sample(sentences, random.randint(2, 4)))
        mock_data.append({"text": text, "label": 0})
        
    df = pd.DataFrame(mock_data)
    df.to_csv(filepath, index=False)
    return df

def get_sentences(text):
    # Basic sentence tokenization (split by full stop or pipe character often used in Hindi/Punjabi)
    sentences = re.split(r'[।\|\.]', text)
    return [s.strip() for s in sentences if s.strip()]

def transform_to_whatsapp_style(text, label):
    """Transforms an article into a WhatsApp forward style."""
    sentences = get_sentences(str(text))
    
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
        
        # Language detection logic (simplified for source_type tagging)
        language = "pa" if any(char >= '\u0A00' and char <= '\u0A7F' for char in truncated_text) else "hi"
        
        final_text = f"{forwarded}{prepend} {truncated_text} {emoji_str}\n\n{append}"
        return final_text, language
        
    else:  # TRUE_OR_REAL
        # Truncate to 2-4 sentences
        num_sentences = min(len(sentences), random.randint(2, 4))
        truncated_text = "। ".join(sentences[:num_sentences]) + "।" if sentences else ""
        
        # Optional prepend
        prepend = f"{random.choice(TRUE_PREPENDS)}\n" if random.random() < 0.3 else ""
        
        # Optional source
        source = f"\n\n{random.choice(TRUE_SOURCES)}" if random.random() < 0.4 else ""
        
        language = "pa" if any(char >= '\u0A00' and char <= '\u0A7F' for char in truncated_text) else "hi"
        
        final_text = f"{prepend}{truncated_text}{source}"
        return final_text, language

def generate_synthetic_data():
    """Generates 500 synthetic fake WhatsApp forwards using templates."""
    print("Generating synthetic WhatsApp data...")
    synthetic_data = []
    
    names = ["रमेश", "सुरेश", "अमित", "ਕਮਲ", "ਗੁਰਪ੍ਰੀਤ"]
    amounts = ["₹50,000", "1 लाख", "10 Lakh", "₹5000", "₹10,000"]
    dates = ["15 अगस्त", "31st March", "ਅੱਜ ਰਾਤ", "कल से"]
    cities = ["दिल्ली", "मुंबई", "ਚੰਡੀਗੜ੍ਹ", "ਲੁਧਿਆਣਾ", "पटना"]
    links = ["http://bit.ly/fake-link", "www.offer.com", "http://free-money.in"]
    
    templates = {
        "Health": [
            "डॉक्टरों ने बताया कि {city} में एक नई बीमारी फैली है। {date} तक घर में रहें।",
            "ਡਾਕਟਰਾਂ ਨੇ ਦੱਸਿਆ ਕਿ {city} ਵਿੱਚ ਨਵੀਂ ਬਿਮਾਰੀ ਹੈ। {date} ਤੋਂ ਪਹਿਲਾਂ ਇਲਾਜ ਕਰਵਾਓ।",
            "रोज सुबह गरम पानी पीने से कैंसर नहीं होता! {city} के बड़े अस्पताल का दावा।",
            "अदरक और लहसुन का रस पीने से सभी वायरस मर जाते हैं।",
            "ਇਸ ਦਵਾਈ ਨਾਲ ਹਰ ਰੋਗ ਠੀਕ ਹੋ ਜਾਵੇਗਾ। ਅੱਜ ਹੀ ਖਰੀਦੋ।"
        ],
        "Gov_Scheme": [
            "सरकार दे रही है {amount} सीधे बैंक खाते में। अभी क्लिक करें {link}",
            "ਸਰਕਾਰ ਦੇ ਰਹੀ ਹੈ {amount} ਸਭ ਨੂੰ। ਫਾਰਮ ਭਰੋ {link}",
            "मोदी सरकार की नई योजना! {date} से पहले आवेदन करें और पाएं {amount}।",
            "छात्रों को फ्री लैपटॉप! इस लिंक {link} पर रजिस्टर करें।",
            "{city} के सभी वासियों के लिए मुफ्त राशन। शेयर करें।"
        ],
        "Religion": [
            "यह चमत्कार {city} के मंदिर में हुआ है। 10 लोगों को भेजें।",
            "ਇਹ ਚਮਤਕਾਰ {city} ਦੇ ਗੁਰਦੁਆਰੇ ਵਿੱਚ ਹੋਇਆ। ਅੱਗੇ ਸ਼ੇਅਰ ਕਰੋ।",
            "इस तस्वीर को देखने के बाद आपको {amount} का लाभ होगा।",
            "भगवान का प्रकोप! {date} को प्रलय आने वाली है।",
            "अगर आप सच्चे भक्त हैं तो इसे इग्नोर न करें।"
        ],
        "Politics": [
            "नेता {name} ने कहा कि वह चुनाव जीतने पर सबको {amount} देंगे।",
            "ਨੇਤਾ {name} ਨੇ ਕਿਹਾ ਕਿ ਪੰਜਾਬ ਵਿੱਚ ਸਭ ਕੁਝ ਮੁਫਤ ਹੋਵੇਗਾ।",
            "{city} में आज बड़ा हंगामा। पुलिस ने किया लाठीचार्ज। वीडियो देखें {link}",
            "जानें कैसे {name} ने किया करोड़ों का घोटाला।",
            "ब्रेकिंग: {date} से चुनाव रद्द कर दिए गए हैं!"
        ],
        "Scam": [
            "लाखों कमाएं घर बैठे! मुझे {amount} भेजें और डबल पाएं।",
            "ਲੱਖ ਕਮਾਓ ਘਰ ਬੈਠੇ! ਅੱਜ ਹੀ {link} ਤੇ ਜਾਓ।",
            "आपका बैंक अकाउंट ब्लॉक हो गया है। KYC अपडेट करने के लिए {link} पर क्लिक करें।",
            "बधाई हो {name}! आपने {amount} की लॉटरी जीती है।",
            "फ्री मोबाइल रिचार्ज {date} तक उपलब्ध। अभी क्लिक करें {link}"
        ]
    }
    
    # 5 categories, 5 templates each, 20 variations each = 500 total
    for category, category_templates in templates.items():
        for template in category_templates:
            for _ in range(20):
                text = template.format(
                    name=random.choice(names),
                    amount=random.choice(amounts),
                    date=random.choice(dates),
                    city=random.choice(cities),
                    link=random.choice(links)
                )
                
                # Apply WhatsApp styling
                text, language = transform_to_whatsapp_style(text, label=1)
                
                synthetic_data.append({
                    "text": text,
                    "label": 1,
                    "language": language,
                    "source_type": "synthetic_forward"
                })
                
    return synthetic_data

def main():
    np.random.seed(42)
    random.seed(42)
    
    base_dir = "/Users/parthsaini/Desktop/FIANL"
    dataset_filepath = os.path.join(base_dir, "fixed_full_article_dataset", "full_article_train.csv")
    out_dir = os.path.join(base_dir, "whatsapp_fake_news", "data", "whatsapp_dataset")
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Load data
    df = mock_dataset_if_not_exists(dataset_filepath)
    
    # 2. Transform articles
    print(f"Transforming {len(df)} articles to WhatsApp style...")
    transformed_data = []
    
    for _, row in df.iterrows():
        text, label = row['text'], row['label']
        wa_text, lang = transform_to_whatsapp_style(text, label)
        transformed_data.append({
            "text": wa_text,
            "label": label,
            "language": lang,
            "source_type": "transformed_article"
        })
        
    transformed_df = pd.DataFrame(transformed_data)
    
    # 3. Generate synthetic
    synthetic_data = generate_synthetic_data()
    synthetic_df = pd.DataFrame(synthetic_data)
    
    # 4. Combine
    final_df = pd.concat([transformed_df, synthetic_df], ignore_index=True)
    
    # Shuffle
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 5. Split (80% train, 10% valid, 10% test)
    total = len(final_df)
    train_end = int(total * 0.8)
    valid_end = int(total * 0.9)
    
    train_df = final_df.iloc[:train_end]
    valid_df = final_df.iloc[train_end:valid_end]
    test_df = final_df.iloc[valid_end:]
    
    # Save
    train_df.to_csv(os.path.join(out_dir, "whatsapp_train.csv"), index=False)
    valid_df.to_csv(os.path.join(out_dir, "whatsapp_valid.csv"), index=False)
    test_df.to_csv(os.path.join(out_dir, "whatsapp_test.csv"), index=False)
    
    # Print stats
    print("\n=== Dataset Statistics ===")
    print(f"Total size: {total}")
    print(f"Train: {len(train_df)}, Valid: {len(valid_df)}, Test: {len(test_df)}")
    
    print("\nClass Distribution (Total):")
    print(final_df['label'].value_counts(normalize=True))
    
    print("\nLanguage Distribution (Total):")
    print(final_df['language'].value_counts(normalize=True))
    
    print("\nSource Type Distribution (Total):")
    print(final_df['source_type'].value_counts(normalize=True))
    
    final_df['text_len'] = final_df['text'].apply(lambda x: len(str(x)))
    print(f"\nAverage text length (chars): {final_df['text_len'].mean():.2f}")
    
    print("\nSample Fake Message:")
    print(final_df[final_df['label'] == 1].iloc[0]['text'])
    
    print("\nSample Real Message:")
    print(final_df[final_df['label'] == 0].iloc[0]['text'])

if __name__ == "__main__":
    main()
