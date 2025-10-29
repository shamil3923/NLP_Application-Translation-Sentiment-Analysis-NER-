from deep_translator import GoogleTranslator
from transformers import MarianMTModel, MarianTokenizer, pipeline
import torch
import streamlit as st

# Cache loaded tokenizers and models to avoid reloading for each call
_tokenizers: dict[str, object] = {}
_models: dict[str, object] = {}


@st.cache_resource
def load_marian_model(model_name: str):
    """Load and cache MarianMT model and tokenizer."""
    print(f"📥 Downloading model: {model_name}...")
    print("⏳ This will take a few minutes on first use (model is ~300-400MB)...")
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    print(f"✅ Model loaded successfully!")
    return tokenizer, model


# ==================== TASK 1: TRANSLATION ====================

def translate_google(text: str, target_lang: str = "fr") -> str:
    """Translate English text using Google Translate API.
    
    target_lang: short code like 'fr', 'hi', 'ta', 'si'.
    """
    try:
        result = GoogleTranslator(source="en", target=target_lang).translate(text)
        return result
    except Exception as e:
        return f"Translation failed: {e}"


def translate_marian(text: str, target_lang: str = "fr") -> str:
    """Translate English text using transformer-based translation models.
    
    Uses Helsinki-NLP MarianMT for most languages, and specialized models for Tamil.
    target_lang: short code like 'fr', 'hi', 'ta', 'si'.
    """
    # Map language codes to available translation models
    model_mapping = {
        "fr": "Helsinki-NLP/opus-mt-en-fr",
        "hi": "Helsinki-NLP/opus-mt-en-hi",
        "ta": "Helsinki-NLP/opus-mt-en-dra",  # Dravidian languages (includes Tamil)
        "si": "Helsinki-NLP/opus-mt-en-mul",  # Multilingual (limited quality for Sinhala)
        "es": "Helsinki-NLP/opus-mt-en-es",  # Spanish (additional)
        "de": "Helsinki-NLP/opus-mt-en-de",  # German (additional)
    }
    
    # Check if language is supported
    if target_lang not in model_mapping:
        unsupported_msg = f"⚠️ No high-quality transformer model available for {target_lang.upper()}. Please use Google Translate instead."
        print(unsupported_msg)
        return unsupported_msg
    
    model_name = model_mapping[target_lang]
    
    try:
        # Load model using cached function (downloads only once)
        tokenizer, model = load_marian_model(model_name)
        
        # For Tamil, we need to use the Dravidian model with special handling
        if target_lang == "ta":
            # The Dravidian model can translate to multiple Dravidian languages
            # We need to prepend the target language token
            text_with_prefix = f">>tam<< {text}"
            inputs = tokenizer(text_with_prefix, return_tensors="pt", padding=True, truncation=True, max_length=512)
        elif target_lang == "si":
            # The multilingual model needs language code prefix for Sinhala
            text_with_prefix = f">>sin<< {text}"
            inputs = tokenizer(text_with_prefix, return_tensors="pt", padding=True, truncation=True, max_length=512)
        else:
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        # Translate
        with torch.no_grad():  # Disable gradient calculation for inference
            translated = model.generate(**inputs)
        result = tokenizer.decode(translated[0], skip_special_tokens=True)
        
        # Add quality warning for Sinhala
        if target_lang == "si":
            warning = "\n\n⚠️ **Quality Note**: This uses a multilingual model with limited Sinhala support. Translation quality may be poor. Compare with Google Translate for accurate results."
            result = result + warning
        
        return result
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"❌ Error details: {error_details}")
        return f"❌ Translation failed. Please use Google Translate instead."


def translate(text: str, target_lang: str = "fr", model_type: str = "google") -> str:
    """Main translation function that routes to the appropriate translator.
    
    model_type: 'google' or 'marian'
    """
    if model_type == "google":
        return translate_google(text, target_lang)
    elif model_type == "marian":
        return translate_marian(text, target_lang)
    else:
        return "Invalid model type. Choose 'google' or 'marian'."


def supported_languages() -> dict:
    """Return supported language codes and their names."""
    return {
        "fr": "French",
        "hi": "Hindi",
        "ta": "Tamil",
        "si": "Sinhala",
    }


# ==================== TASK 2: SENTIMENT ANALYSIS ====================

@st.cache_resource
def load_sentiment_model():
    """Load and cache the sentiment analysis model."""
    print("📥 Loading sentiment analysis model (one-time download)...")
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

def analyze_sentiment(text: str) -> dict:
    """Analyze sentiment of the given text.
    
    Returns a dictionary with label and score.
    """
    try:
        # Use cached model
        sentiment_analyzer = load_sentiment_model()
        
        result = sentiment_analyzer(text[:512])[0]  # Limit to 512 tokens
        return {
            "label": result["label"],
            "score": round(result["score"] * 100, 2),
            "emoji": "😊" if result["label"] == "POSITIVE" else "😞"
        }
    except Exception as e:
        return {
            "label": "ERROR",
            "score": 0,
            "emoji": "❌",
            "error": str(e)
        }


# ==================== TASK 3: NAMED ENTITY RECOGNITION ====================

@st.cache_resource
def load_ner_model():
    """Load and cache the NER model."""
    print("📥 Loading NER model (one-time download)...")
    return pipeline(
        "ner",
        model="dslim/bert-base-NER",
        aggregation_strategy="simple"
    )

def extract_entities(text: str) -> list:
    """Extract named entities from the given text.
    
    Returns a list of entities with their labels.
    """
    try:
        # Use cached model
        ner_pipeline = load_ner_model()
        
        entities = ner_pipeline(text)
        
        # Format the results
        formatted_entities = []
        for entity in entities:
            formatted_entities.append({
                "text": entity["word"],
                "label": entity["entity_group"],
                "score": round(entity["score"] * 100, 2)
            })
        
        return formatted_entities
    except Exception as e:
        return [{"text": "Error", "label": "ERROR", "score": 0, "error": str(e)}]


if __name__ == "__main__":
    # Quick local example
    english_text = "Hello, how are you?"
    print("Available models:", supported_languages())
    
    print("\n=== TRANSLATION ===")
    try:
        print("French (Google):", translate(english_text, "fr", "google"))
    except Exception as e:
        print("French translation failed:", e)
    
    print("\n=== SENTIMENT ANALYSIS ===")
    sentiment_text = "I love this product! It's amazing and works perfectly."
    print(f"Text: {sentiment_text}")
    print("Result:", analyze_sentiment(sentiment_text))
    
    print("\n=== NAMED ENTITY RECOGNITION ===")
    ner_text = "Apple Inc. was founded by Steve Jobs in Cupertino, California."
    print(f"Text: {ner_text}")
    print("Entities:", extract_entities(ner_text))