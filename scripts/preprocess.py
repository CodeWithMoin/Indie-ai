import os
import re
from sklearn.model_selection import train_test_split
import csv

def clean_text(text):
    """Clean and normalize text data"""
    text = re.sub(r'[^\w\s]', '', text.strip())  # Remove special chars
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    return text.lower()

def create_csv_files(base_dir, pair):
    """Create CSV files for each split from the text files"""
    src_lang, tgt_lang = pair.split('-')
    pair_dir = os.path.join(base_dir, f"data/samanantar/{pair}")
    
    for split in ["train", "valid", "test"]:
        src_path = os.path.join(pair_dir, f"{split}.{src_lang}")
        tgt_path = os.path.join(pair_dir, f"{split}.{tgt_lang}")
        csv_path = os.path.join(pair_dir, f"{split}.csv")
        
        with open(src_path) as src_file, open(tgt_path) as tgt_file, open(csv_path, "w", newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([src_lang, tgt_lang])  # Write header
            
            for src_line, tgt_line in zip(src_file, tgt_file):
                writer.writerow([src_line.strip(), tgt_line.strip()])

def process_language_pair(base_dir, pair):
    src_lang, tgt_lang = pair.split('-')
    pair_dir = os.path.join(base_dir, f"data/samanantar/{pair}")
    
    # Input files
    src_input = os.path.join(pair_dir, f"train.{src_lang}")
    tgt_input = os.path.join(pair_dir, f"train.{tgt_lang}")

    # Load data
    with open(src_input, "r", encoding="utf-8") as f:
        src_lines = [clean_text(line) for line in f]
    with open(tgt_input, "r", encoding="utf-8") as f:
        tgt_lines = [clean_text(line) for line in f]

    # Split dataset
    src_train, src_temp, tgt_train, tgt_temp = train_test_split(
        src_lines, tgt_lines, test_size=0.2, random_state=42
    )
    src_val, src_test, tgt_val, tgt_test = train_test_split(
        src_temp, tgt_temp, test_size=0.5, random_state=42
    )

    # Save splits
    def save_split(data, path):
        full_path = os.path.join(base_dir, path)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write("\n".join(data))

    save_split(src_train, f"data/samanantar/{pair}/train.{src_lang}")
    save_split(tgt_train, f"data/samanantar/{pair}/train.{tgt_lang}")
    save_split(src_val, f"data/samanantar/{pair}/valid.{src_lang}")
    save_split(tgt_val, f"data/samanantar/{pair}/valid.{tgt_lang}")
    save_split(src_test, f"data/samanantar/{pair}/test.{src_lang}")
    save_split(tgt_test, f"data/samanantar/{pair}/test.{tgt_lang}")

    # After saving splits, create CSV files
    create_csv_files(base_dir, pair)

def clean_and_split_dataset():
    lang_pairs = [
        'en-as', 'en-bn', 'en-gu', 'en-hi',
        'en-kn', 'en-ml', 'en-mr', 'en-or',
        'en-pa', 'en-ta', 'en-te'
    ]

    script_dir = os.path.dirname(__file__)
    base_dir = os.path.dirname(script_dir)
    print(f"📂 Project root: {base_dir}")

    for pair in lang_pairs:
        print(f"\n{'='*40}")
        print(f"🚀 Processing language pair: {pair.upper()}")
        print(f"{'='*40}")
        
        pair_dir = os.path.join(base_dir, f"data/samanantar/{pair}")
        if not os.path.exists(pair_dir):
            print(f"⚠️  Directory not found: {pair_dir}")
            continue
            
        try:
            process_language_pair(base_dir, pair)
            print(f"🎉 Successfully processed {pair.upper()}")
        except Exception as e:
            print(f"❌ Error processing {pair}: {str(e)}")
            continue

if __name__ == "__main__":
    print("Starting multilingual dataset processing...")
    clean_and_split_dataset()
    print("\nAll language pairs processed!")