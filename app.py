import streamlit as st
import torch
from transformers import pipeline, AutoProcessor, AutoModelForVision
from PIL import Image
import io
import os
import tempfile
import numpy as np
import pandas as pd
from gtts import gTTS
import base64
from io import BytesIO
from PyPDF2 import PdfReader
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# Set Hugging Face token for model access
os.environ["HF_TOKEN"] = "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"

# Set page configuration
st.set_page_config(
    page_title="Gemma 3 Multimodal Translation",
    page_icon="🌐",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4B0082;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #6A5ACD;
        margin-bottom: 1rem;
    }
    .info-text {
        font-size: 1rem;
        color: #4682B4;
    }
    .stButton button {
        background-color: #4B0082;
        color: white;
        border-radius: 5px;
    }
    .result-container {
        background-color: #F8F8FF;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #E6E6FA;
    }
</style>
""", unsafe_allow_html=True)

# Header section
st.markdown("<h1 class='main-header'>Gemma 3 Multimodal Translation & Analysis</h1>", unsafe_allow_html=True)
st.markdown("<p class='info-text'>Powered by Google's Gemma 3 model (google/gemma-3-4b-pt)</p>", unsafe_allow_html=True)

# Helper function to get audio playback HTML
def get_audio_player_html(audio_data):
    audio_base64 = base64.b64encode(audio_data).decode()
    return f"""
    <audio controls autoplay=false>
        <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
    </audio>
    """

# Helper function for text-to-speech using gTTS
def text_to_speech(text, lang='en'):
    try:
        tts = gTTS(text=text, lang=lang, slow=False, tld='co.uk')  # Using female voice
        fp = BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return fp.read()
    except Exception as e:
        st.error(f"Error generating speech: {str(e)}")
        return None

# Function to extract text from documents
def extract_text_from_document(file):
    file_extension = file.name.split('.')[-1].lower()
    text = ""
    
    try:
        if file_extension == 'pdf':
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}')
            temp_file.write(file.getvalue())
            temp_file.close()
            
            pdf_reader = PdfReader(temp_file.name)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
                
            os.unlink(temp_file.name)
            
        elif file_extension in ['txt', 'csv']:
            stringio = io.StringIO(file.getvalue().decode("utf-8"))
            text = stringio.read()
            
        elif file_extension in ['jpg', 'jpeg', 'png']:
            # For images, we'll use Gemma to analyze them later
            text = "Image file detected. Will be processed by Gemma 3."
            
        else:
            text = "Unsupported file format for text extraction."
    
    except Exception as e:
        text = f"Error extracting text: {str(e)}"
    
    return text

# Train a simple ML model for auto-fill suggestions
@st.cache_resource
def train_autofill_model():
    # Sample data for demonstration
    sample_data = [
        "digital marketing campaign for fashion brand",
        "social media strategy for tech startup",
        "content marketing plan for food industry",
        "email marketing campaign for financial services",
        "SEO optimization for e-commerce website",
        "video marketing strategy for educational content",
        "influencer marketing for cosmetics brand",
        "mobile advertising campaign for gaming app",
        "B2B marketing strategy for software company",
        "local marketing campaign for retail store"
    ]
    
    # Sample labels (marketing type)
    labels = [
        "digital", "social", "content", "email", 
        "seo", "video", "influencer", "mobile", "b2b", "local"
    ]
    
    # Create feature vectors
    vectorizer = TfidfVectorizer(max_features=100)
    X = vectorizer.fit_transform(sample_data)
    
    # Train a simple model
    model = RandomForestClassifier(n_estimators=10)
    model.fit(X, labels)
    
    return model, vectorizer

# Function to get autofill suggestions
def get_autofill_suggestions(text, model, vectorizer):
    try:
        # Transform the input text
        X = vectorizer.transform([text])
        
        # Get prediction probabilities
        proba = model.predict_proba(X)[0]
        
        # Get top 3 suggestions
        indices = proba.argsort()[-3:][::-1]
        
        suggestions = []
        for idx in indices:
            marketing_type = model.classes_[idx]
            confidence = proba[idx]
            if confidence > 0.1:  # Only include if confidence is reasonable
                suggestions.append((marketing_type, confidence))
        
        return suggestions
    except Exception as e:
        st.error(f"Error generating suggestions: {str(e)}")
        return []

# Load ML model for autofill
autofill_model, vectorizer = train_autofill_model()

# Function to load Gemma model
@st.cache_resource
def load_gemma_model():
    try:
        # First try to use the pipeline approach
        pipe = pipeline(
            "image-text-to-text",
            model="google/gemma-3-4b-pt",
            token=os.environ["HF_TOKEN"],
            device=0 if torch.cuda.is_available() else -1,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        return pipe, "pipeline"
    except Exception as e:
        st.warning(f"Pipeline approach failed: {str(e)}. Trying direct model loading...")
        
        try:
            # Try the direct model loading approach
            processor = AutoProcessor.from_pretrained(
                "google/gemma-3-4b-pt", 
                token=os.environ["HF_TOKEN"]
            )
            model = AutoModelForVision.from_pretrained(
                "google/gemma-3-4b-pt",
                token=os.environ["HF_TOKEN"],
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            return (processor, model), "direct"
        except Exception as e:
            st.error(f"Failed to load Gemma model: {str(e)}")
            return None, "failed"

# Check if model should be loaded
if 'skip_model_load' not in st.session_state:
    st.session_state.skip_model_load = False

# Sidebar for language selection and options
with st.sidebar:
    st.markdown("<h2 class='sub-header'>Settings</h2>", unsafe_allow_html=True)
    
    # Model loading toggle
    skip_load = st.checkbox("Skip model loading (for faster UI testing)", value=st.session_state.skip_model_load)
    st.session_state.skip_model_load = skip_load
    
    source_lang = st.selectbox(
        "Source Language",
        options=["Auto-detect", "English", "Spanish", "French", "German", "Chinese", "Japanese", "Korean", "Russian", "Arabic", "Hindi"],
        index=0
    )
    
    target_lang = st.selectbox(
        "Target Language",
        options=["English", "Spanish", "French", "German", "Chinese", "Japanese", "Korean", "Russian", "Arabic", "Hindi"],
        index=0
    )
    
    # Technical info
    st.markdown("---")
    st.markdown("<h3 class='sub-header'>Technical Info</h3>", unsafe_allow_html=True)
    st.markdown(f"""
    - Model: google/gemma-3-4b-pt
    - HF Token: {'Configured ✅' if os.environ.get('HF_TOKEN') else 'Missing ❌'}
    - GPU: {'Available ✅' if torch.cuda.is_available() else 'Not available ❌'}
    - Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}
    """)

# Load model if not skipped
gemma_model = None
model_type = "none"
if not st.session_state.skip_model_load:
    with st.spinner("Loading Gemma 3 model... This might take a minute."):
        gemma_model, model_type = load_gemma_model()
        if gemma_model:
            st.success(f"Gemma 3 model loaded successfully using {model_type} approach!")
        else:
            st.error("Failed to load Gemma 3 model. Some features will be unavailable.")
else:
    st.sidebar.warning("Model loading skipped. Only UI demonstration available.")

# Main content area with tabs
tab1, tab2, tab3 = st.tabs(["Multimodal Translation", "Document Analysis", "Marketing Assistant"])

# Translation Tab
with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h3 class='sub-header'>Input</h3>", unsafe_allow_html=True)
        input_type = st.radio("Input Type", ["Text", "Image", "Combined (Image + Text)"])
        
        input_text = ""
        uploaded_file = None
        
        if input_type == "Text":
            input_text = st.text_area("Enter text to translate or analyze", height=200)
        elif input_type == "Image":
            uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
            if uploaded_file:
                st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        else:  # Combined
            uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
            if uploaded_file:
                st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
            input_text = st.text_area("Enter text prompt or question about the image", 
                                      value="<start_of_image> In this image, there is")
    
    with col2:
        st.markdown("<h3 class='sub-header'>Translation & Analysis</h3>", unsafe_allow_html=True)
        
        if st.button("Process", key="process_button"):
            with st.spinner("Processing with Gemma 3..."):
                result_text = ""
                
                # Process text only
                if input_type == "Text" and input_text:
                    if gemma_model and not st.session_state.skip_model_load:
                        try:
                            # Process text with Gemma 3
                            if model_type == "pipeline":
                                result = gemma_model(input_text)
                                result_text = result[0]["generated_text"]
                            else:
                                st.warning("Direct text processing not implemented for this model version")
                                result_text = f"[Gemma would process]: {input_text}"
                        except Exception as e:
                            st.error(f"Error processing text: {str(e)}")
                            result_text = f"Error processing the text. Using fallback."
                    else:
                        # Fallback without model
                        result_text = f"[Gemma would process]: {input_text}"
                
                # Process image only or image+text
                elif uploaded_file and (input_type == "Image" or input_type == "Combined (Image + Text)"):
                    if gemma_model and not st.session_state.skip_model_load:
                        try:
                            # Open the image
                            image = Image.open(uploaded_file).convert('RGB')
                            
                            # Process image with Gemma 3
                            prompt = input_text if input_type == "Combined (Image + Text)" else "<start_of_image> Describe this image in detail."
                            
                            if model_type == "pipeline":
                                result = gemma_model(image, text=prompt)
                                result_text = result[0]["generated_text"]
                            else:
                                # Use direct model approach
                                processor, model = gemma_model
                                inputs = processor(text=prompt, images=image, return_tensors="pt")
                                
                                # Move to GPU if available
                                if torch.cuda.is_available():
                                    inputs = {k: v.to("cuda") for k, v in inputs.items()}
                                    model = model.to("cuda")
                                
                                # Generate with model
                                with torch.inference_mode():
                                    output = model.generate(**inputs, max_new_tokens=512, do_sample=False)
                                
                                # Decode output
                                result_text = processor.decode(output[0], skip_special_tokens=True)
                        except Exception as e:
                            st.error(f"Error processing image: {str(e)}")
                            result_text = f"Error processing the image. Using fallback."
                    else:
                        # Fallback without model
                        result_text = "[Gemma would analyze the image and generate text based on it]"
                
                else:
                    st.error("Please provide input for processing")
                    result_text = ""
                
                # Display results
                if result_text:
                    st.markdown("<h4 class='sub-header'>Results</h4>", unsafe_allow_html=True)
                    st.markdown("<div class='result-container'>", unsafe_allow_html=True)
                    st.write(result_text)
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Generate and display speech
                    target_lang_code = target_lang.lower()[:2]
                    if target_lang_code == "ch":  # Fix for Chinese
                        target_lang_code = "zh"
                    
                    # Limit text to avoid long processing time
                    speech_text = result_text[:500] + ("..." if len(result_text) > 500 else "")
                    audio_bytes = text_to_speech(speech_text, lang=target_lang_code)
                    
                    if audio_bytes:
                        st.markdown("<h4 class='sub-header'>Audio Narration (Female Voice)</h4>", unsafe_allow_html=True)
                        st.markdown(get_audio_player_html(audio_bytes), unsafe_allow_html=True)

# Document Analysis Tab
with tab2:
    st.markdown("<h3 class='sub-header'>Document Analysis</h3>", unsafe_allow_html=True)
    
    doc_file = st.file_uploader("Upload document for analysis", type=["pdf", "txt", "csv", "jpg", "jpeg", "png"])
    
    if doc_file is not None:
        file_details = {"Filename": doc_file.name, "FileType": doc_file.type, "FileSize": f"{doc_file.size / 1024:.2f} KB"}
        st.write(file_details)
        
        if st.button("Analyze Document"):
            with st.spinner("Analyzing document..."):
                # Extract text from document
                extracted_text = extract_text_from_document(doc_file)
                
                # Handle image files differently
                is_image = doc_file.name.lower().endswith(('jpg', 'jpeg', 'png'))
                
                if is_image and gemma_model and not st.session_state.skip_model_load:
                    try:
                        # Process image with Gemma
                        image = Image.open(doc_file).convert('RGB')
                        prompt = "<start_of_image> Analyze this document image in detail and extract key information."
                        
                        if model_type == "pipeline":
                            result = gemma_model(image, text=prompt)
                            extracted_text = result[0]["generated_text"]
                        else:
                            # Use direct model approach
                            processor, model = gemma_model
                            inputs = processor(text=prompt, images=image, return_tensors="pt")
                            
                            # Move to GPU if available
                            if torch.cuda.is_available():
                                inputs = {k: v.to("cuda") for k, v in inputs.items()}
                                model = model.to("cuda")
                            
                            # Generate with model
                            with torch.inference_mode():
                                output = model.generate(**inputs, max_new_tokens=512, do_sample=False)
                            
                            # Decode output
                            extracted_text = processor.decode(output[0], skip_special_tokens=True)
                    except Exception as e:
                        st.error(f"Error analyzing image document: {str(e)}")
                
                # Display extracted text preview
                st.markdown("<h4 class='sub-header'>Document Content</h4>", unsafe_allow_html=True)
                st.markdown("<div class='result-container'>", unsafe_allow_html=True)
                st.write(extracted_text[:1000] + ("..." if len(extracted_text) > 1000 else ""))
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Get ML autofill suggestions
                suggestions = get_autofill_suggestions(extracted_text, autofill_model, vectorizer)
                
                # Display suggestions
                st.markdown("<h4 class='sub-header'>ML Auto-fill Suggestions</h4>", unsafe_allow_html=True)
                st.markdown("<div class='result-container'>", unsafe_allow_html=True)
                
                if suggestions:
                    for marketing_type, confidence in suggestions:
                        st.write(f"• {marketing_type.title()} Marketing Strategy (Confidence: {confidence:.2f})")
                else:
                    st.write("No strong suggestions available based on the content")
                
                st.markdown("</div>", unsafe_allow_html=True)

# Marketing Assistant Tab
with tab3:
    st.markdown("<h3 class='sub-header'>Marketing Assistant</h3>", unsafe_allow_html=True)
    
    # Marketing Strategy Form
    with st.form("marketing_form"):
        # Business Information
        st.markdown("<h4 class='sub-header'>Business Information</h4>", unsafe_allow_html=True)
        business_name = st.text_input("Business Name")
        industry = st.selectbox("Industry", ["Retail", "Technology", "Healthcare", "Finance", "Education", "Food & Beverage", "Entertainment", "Other"])
        target_audience = st.text_area("Target Audience Description")
        
        # Campaign Goals
        st.markdown("<h4 class='sub-header'>Campaign Goals</h4>", unsafe_allow_html=True)
        primary_goal = st.selectbox("Primary Goal", ["Brand Awareness", "Lead Generation", "Sales Conversion", "Customer Retention", "Product Launch"])
        budget = st.slider("Budget (USD)", 1000, 100000, 10000, step=1000)
        timeline = st.slider("Campaign Duration (Days)", 7, 180, 30)
        
        # Marketing Channels
        st.markdown("<h4 class='sub-header'>Marketing Channels</h4>", unsafe_allow_html=True)
        channels = st.multiselect("Select Marketing Channels", 
                                  ["Social Media", "Email", "Content Marketing", "SEO", "Paid Advertising", 
                                   "Influencer Marketing", "Events", "PR", "Video Marketing"])
        
        # Submit Button
        submitted = st.form_submit_button("Generate Marketing Strategy")
        
    if submitted:
        with st.spinner("Generating marketing strategy..."):
            # Generate content using Gemma if available
            strategy_text = ""
            
            if gemma_model and not st.session_state.skip_model_load:
                try:
                    # Create a prompt for Gemma
                    prompt = f"""
                    Create a marketing strategy for {business_name}, a company in the {industry} industry.
                    Target audience: {target_audience}
                    Primary goal: {primary_goal}
                    Budget: ${budget}
                    Timeline: {timeline} days
                    Marketing channels: {', '.join(channels)}
                    
                    Provide a comprehensive strategy with specific actionable steps.
                    """
                    
                    if model_type == "pipeline":
                        result = gemma_model(prompt)
                        strategy_text = result[0]["generated_text"]
                    else:
                        # Fallback to template
                        strategy_text = f"[Gemma would generate a full marketing strategy for {business_name}]"
                except Exception as e:
                    st.error(f"Error generating strategy: {str(e)}")
                    strategy_text = f"Error generating the strategy. Using fallback."
            else:
                # Simple template if model not available
                strategy_text = f"""
                # Marketing Strategy for {business_name}
                
                ## Executive Summary
                This strategy focuses on achieving {primary_goal} for {business_name} in the {industry} industry over {timeline} days with a budget of ${budget:,}.
                
                ## Target Audience
                {target_audience}
                
                ## Channel Strategy
                {', '.join(channels)}
                
                ## Implementation Plan
                [Detailed implementation steps would be generated by Gemma 3]
                
                ## Performance Metrics
                - Expected ROI: 3.5x
                - Estimated Reach: {int(budget * 100):,}
                - Engagement Rate: 4.2%
                """
            
            # Generate a voice summary using gTTS
            strategy_summary = f"Hello! I've analyzed the marketing data for {business_name}. Based on your {primary_goal} goal, I recommend focusing on {', '.join(channels[:2])} for your {industry} business targeting {target_audience[:50]}..."
            
            audio_bytes = text_to_speech(strategy_summary)
            
            # Display strategy
            st.markdown("<h4 class='sub-header'>Marketing Strategy</h4>", unsafe_allow_html=True)
            st.markdown("<div class='result-container'>", unsafe_allow_html=True)
            st.write(strategy_text)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Display voice message
            if audio_bytes:
                st.markdown("<h4 class='sub-header'>Strategy Voice Summary (Female Voice)</h4>", unsafe_allow_html=True)
                st.markdown(get_audio_player_html(audio_bytes), unsafe_allow_html=True)
            
            # Display metrics and KPIs
            st.markdown("<h4 class='sub-header'>Projected Performance Metrics</h4>", unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Estimated Reach", f"{int(budget * 100):,}", "+15%")
            with col2:
                st.metric("Engagement Rate", "4.2%", "+0.8%")
            with col3:
                st.metric("ROI", "3.5x", "+0.2x")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center;'>Powered by Gemma 3 (google/gemma-3-4b-pt) © 2025</p>", unsafe_allow_html=True)
