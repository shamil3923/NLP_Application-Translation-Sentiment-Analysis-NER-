import streamlit as st
from main import (
    translate, 
    supported_languages, 
    analyze_sentiment, 
    extract_entities
)

st.set_page_config(page_title="NLP Multi-Task Application", page_icon="🤖", layout="wide")

# Sidebar for navigation
st.sidebar.title("🤖 NLP Tasks")
st.sidebar.markdown("**Group Assignment - 3 Members**")
task = st.sidebar.radio(
    "Select a Task:",
    ["🌐 Translation", "😊 Sentiment Analysis", "🏷️ Named Entity Recognition"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "This application demonstrates three NLP tasks:\n\n"
    "**Member 1**: Translation (Google + MarianMT)\n\n"
    "**Member 2**: Sentiment Analysis\n\n"
    "**Member 3**: Named Entity Recognition"
)

# ==================== TASK 1: TRANSLATION ====================
if task == "🌐 Translation":
    st.title("🌐 Language Translation")
    st.markdown("Translate English text to other languages using **Google Translate** or **Helsinki-NLP MarianMT** models.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        langs = supported_languages()
        target = st.selectbox(
            "Target Language:", 
            options=list(langs.keys()), 
            format_func=lambda k: f"{langs[k]} ({k})"
        )
    
    with col2:
        model_type = st.selectbox(
            "Translation Model:",
            options=["google", "marian"],
            format_func=lambda x: "Google Translate (API)" if x == "google" else "Transformer Model (Neural MT)"
        )
    
    input_mode = st.radio("Input Mode:", ("Text Input", "Upload .txt File"))
    
    text = ""
    if input_mode == "Text Input":
        text = st.text_area("Enter English text:", height=150, placeholder="Type or paste your English text here...")
    else:
        uploaded = st.file_uploader("Upload a .txt file", type=["txt"])
        if uploaded is not None:
            try:
                raw = uploaded.read()
                text = raw.decode("utf-8")
                st.text_area("File Content:", text, height=150, disabled=True)
            except Exception:
                st.error("Unable to read uploaded file. Make sure it's a UTF-8 encoded text file.")
    
    if st.button("🔄 Translate", type="primary", use_container_width=True):
        if not text or text.strip() == "":
            st.warning("⚠️ Please provide some English text to translate.")
        else:
            status_msg = f"Translating to {langs[target]} using {model_type.upper()}..."
            if model_type == "marian":
                status_msg += " (First time may take a few minutes to download the model)"
            
            with st.spinner(status_msg):
                try:
                    result = translate(text, target, model_type)
                    
                    # Check if result is an error message
                    if "failed" in result.lower() or "error" in result.lower():
                        st.error(f"❌ {result}")
                        if model_type == "marian":
                            st.info("💡 Tip: MarianMT models need to be downloaded on first use. Make sure you have a stable internet connection.")
                    else:
                        st.success("✅ Translation Complete!")
                        st.markdown(f"### Translation Result ({langs[target]}):")
                        st.info(result)
                        st.caption(f"🔧 Model: {model_type.upper()} | Target: {langs[target]}")
                except Exception as e:
                    st.error(f"❌ Translation failed: {e}")
                    if model_type == "marian":
                        st.info("💡 Try using Google Translate instead, or check your internet connection and try again.")
    
    with st.expander("📚 View Supported Languages & Models"):
        st.markdown("### Google Translate (API-based)")
        st.write("✅ **Supports all languages with high quality:**")
        for code, name in langs.items():
            st.write(f"• **{name}** (`{code}`) - Recommended for production")
        
        st.markdown("---")
        st.markdown("### Transformer Models (Neural Machine Translation)")
        st.markdown("**✅ Excellent Quality - Dedicated Models:**")
        st.write("• **French** (`fr`) - Helsinki-NLP/opus-mt-en-fr")
        st.write("• **Hindi** (`hi`) - Helsinki-NLP/opus-mt-en-hi")
        st.write("")
        st.markdown("**✅ Good Quality - Family-based Model:**")
        st.write("• **Tamil** (`ta`) - Helsinki-NLP/opus-mt-en-dra (Dravidian)")
        st.write("")
        st.markdown("**⚠️ Limited Quality - Multilingual Model:**")
        st.write("• **Sinhala** (`si`) - Helsinki-NLP/opus-mt-en-mul (1000+ languages)")
        st.write("  - Note: Quality may be lower, compare with Google Translate")
        st.write("")
        st.info("💡 **Tip**: Compare transformer models with Google Translate to see quality differences! Sinhala works best with Google Translate.")

# ==================== TASK 2: SENTIMENT ANALYSIS ====================
elif task == "😊 Sentiment Analysis":
    st.title("😊 Sentiment Analysis")
    st.markdown("Analyze the sentiment (positive/negative) of English text using **DistilBERT** model.")
    
    text = st.text_area(
        "Enter text to analyze:", 
        height=150, 
        placeholder="Type or paste text to analyze its sentiment..."
    )
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        analyze_btn = st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True)
    
    if analyze_btn:
        if not text or text.strip() == "":
            st.warning("⚠️ Please enter some text to analyze.")
        else:
            with st.spinner("Analyzing sentiment..."):
                result = analyze_sentiment(text)
                
                if "error" in result:
                    st.error(f"❌ Analysis failed: {result['error']}")
                else:
                    st.success("✅ Analysis Complete!")
                    
                    # Display results in a nice format
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Sentiment", result["label"])
                    with col2:
                        st.metric("Confidence", f"{result['score']}%")
                    with col3:
                        st.markdown(f"### {result['emoji']}")
                    
                    # Visual representation
                    if result["label"] == "POSITIVE":
                        st.progress(result["score"] / 100)
                        st.success(f"😊 The text expresses a **{result['label']}** sentiment with {result['score']}% confidence.")
                    else:
                        st.progress(result["score"] / 100)
                        st.error(f"😞 The text expresses a **{result['label']}** sentiment with {result['score']}% confidence.")
                    
                    st.caption("🔧 Model: DistilBERT (Fine-tuned on SST-2)")
    
    with st.expander("ℹ️ About Sentiment Analysis"):
        st.markdown("""
        **Sentiment Analysis** is an NLP task that determines the emotional tone of text.
        
        - **Model**: DistilBERT (distilbert-base-uncased-finetuned-sst-2-english)
        - **Output**: POSITIVE or NEGATIVE with confidence score
        - **Use Cases**: Customer reviews, social media monitoring, feedback analysis
        """)

# ==================== TASK 3: NAMED ENTITY RECOGNITION ====================
elif task == "🏷️ Named Entity Recognition":
    st.title("🏷️ Named Entity Recognition (NER)")
    st.markdown("Extract named entities (people, organizations, locations) from text using **BERT-based NER** model.")
    
    text = st.text_area(
        "Enter text to extract entities:", 
        height=150, 
        placeholder="Type or paste text to identify named entities (people, organizations, locations)..."
    )
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        extract_btn = st.button("🔍 Extract Entities", type="primary", use_container_width=True)
    
    if extract_btn:
        if not text or text.strip() == "":
            st.warning("⚠️ Please enter some text to analyze.")
        else:
            with st.spinner("Extracting entities..."):
                entities = extract_entities(text)
                
                if entities and "error" in entities[0]:
                    st.error(f"❌ Extraction failed: {entities[0]['error']}")
                else:
                    st.success(f"✅ Found {len(entities)} entities!")
                    
                    if len(entities) > 0:
                        # Group entities by type
                        entity_types = {}
                        for entity in entities:
                            label = entity["label"]
                            if label not in entity_types:
                                entity_types[label] = []
                            entity_types[label].append(entity)
                        
                        # Display statistics
                        cols = st.columns(len(entity_types))
                        for idx, (label, ents) in enumerate(entity_types.items()):
                            with cols[idx]:
                                st.metric(label, len(ents))
                        
                        st.markdown("---")
                        
                        # Display entities by type
                        for label, ents in entity_types.items():
                            st.markdown(f"### {label}")
                            for entity in ents:
                                col1, col2 = st.columns([3, 1])
                                with col1:
                                    st.markdown(f"**{entity['text']}**")
                                with col2:
                                    st.caption(f"Confidence: {entity['score']}%")
                        
                        st.caption("🔧 Model: BERT-base-NER (dslim/bert-base-NER)")
                    else:
                        st.info("No entities found in the text.")
    
    with st.expander("ℹ️ About Named Entity Recognition"):
        st.markdown("""
        **Named Entity Recognition (NER)** identifies and classifies named entities in text.
        
        - **Model**: BERT-base-NER (dslim/bert-base-NER)
        - **Entity Types**:
            - **PER**: Person names
            - **ORG**: Organizations, companies, agencies
            - **LOC**: Locations, cities, countries
            - **MISC**: Miscellaneous entities
        - **Use Cases**: Information extraction, content classification, knowledge graphs
        """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎓 Group Project")
st.sidebar.markdown("NLP Multi-Task Application")
st.sidebar.markdown("*3 Members - 3 Tasks*")
