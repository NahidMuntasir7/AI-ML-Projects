import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load the model and tokenizer from Hugging Face
model_name = "TextLabRUET/xlm-r_based_bangla_sentence_classifier"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Set device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Mapping predicted class to Bangla sentence types
class_mapping = {
    0: "Assertive Sentence (বর্ণনামূলক বাক্য)",
    1: "Interrogative Sentence (প্রশ্নবোধক বাক্য)",
    2: "Imperative Sentence (অনুজ্ঞাসূচক বাক্য)",
    3: "Optative Sentence (প্রার্থনা সূচক বাক্য)",
    4: "Exclamatory Sentence (বিস্ময়সূচক বাক্য)"
}

# Function for prediction
def predict_bangla_sentence(sentence):
    # Tokenize the input sentence
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=128)
    
    # Move input tensors to the same device as the model
    inputs = {key: val.to(device) for key, val in inputs.items()}
    
    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the predicted class
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=-1).item()
    
    # Return the predicted sentence type
    sentence_type = class_mapping.get(predicted_class, "Unknown Sentence Type")
    return f"Predicted Class: {sentence_type}"

# Create Gradio UI
iface = gr.Interface(
    fn=predict_bangla_sentence,
    inputs=gr.Textbox(lines=2, placeholder="Enter a sentence in any language...", label="Sentence"),
    outputs=gr.Textbox(lines=2, label="Output"),
    title="Multilingual Sentence Classifier⚡",
    description = (
        "This model, trained on a curated Bangla dataset by **TextLab RUET**, classifies sentences into five categories: Assertive, Interrogative, Imperative, Optative, and Exclamatory. \n\n"
        "Although it was fine-tuned on Bangla sentences, the model **XLM-R** (a multilingual transformer model based on the BERT architecture) leverages transfer learning from 100+ languages, enabling it to classify sentences in languages like Bangla, English, Spanish, French, Arabic, Chinese, Japanese, and more.\n\n"
        "**Note**: While we aim for accuracy, the model may occasionally misclassify sentences due to dataset limitations. We appreciate your understanding.\n\n"
        "**Enter a sentence in any language below to see how our model interprets it!** 🤖"
    ),

    theme="compact",
)

# Launch the Gradio app
iface.launch(share=True)
