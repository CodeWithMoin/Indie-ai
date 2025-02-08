from transformers import MarianMTModel, MarianTokenizer
import torch

def evaluate_model():
    device = torch.device("mps")
    
    # Load model
    model = MarianMTModel.from_pretrained("models/fine-tuned-marianmt-en-hi").to(device)
    tokenizer = MarianTokenizer.from_pretrained("models/fine-tuned-marianmt-en-hi")
    
    # Load test data
    with open("data/samanantar/en-hi/test.en") as f:
        test_samples = [line.strip() for line in f][:5]  # First 5 samples
    
    # Translate
    model.eval()
    for sample in test_samples:
        inputs = tokenizer(sample, return_tensors="pt").to(device)
        outputs = model.generate(**inputs)
        translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"Input: {sample}")
        print(f"Translation: {translation}\n")

if __name__ == "__main__":
    evaluate_model()