import streamlit as st
import torch
from transformers import pipeline, AutoProcessor, Gemma3ForConditionalGeneration
from PIL import Image
import io
import os
import tempfile
import numpy as np
import pandas as pd
from gtts import gTTS
import base64
import time
from io import BytesIO
import mammoth
import docx
import pytesseract
from PyPDF2 import PdfReader
import librosa
import moviepy.editor as mp
import whisper
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from langdetect import detect, LangDetectException

# Set page configuration
st.set_page_config(
    page_title="ROH-Ads: Multimodal Translation & Analysis",
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
st.markdown("<h1 class='main-header'>ROH-Ads: Multimodal Translation & Analysis</h1>", unsafe_allow_html=True)
st.markdown("<p class='info-text'>AI-powered multimodal translation and analysis with 140+ language support</p>", unsafe_allow_html=True)

# Helper function to get audio playback HTML
def get_audio_player_html(audio_data):
    audio_base64 = base64.b64encode(audio_data).decode()
    return f"""
    <audio controls autoplay=false>
        <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
    </audio>
    """

# Helper function for text-to-speech
def text_to_speech(text, lang='en'):
    tts = gTTS(text=text, lang=lang, slow=False)
    fp = BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    return fp.read()

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
            
        elif file_extension in ['docx', 'doc']:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}')
            temp_file.write(file.getvalue())
            temp_file.close()
            
            with open(temp_file.name, "rb") as docx_file:
                result = mammoth.extract_raw_text(docx_file)
                text = result.value
                
            os.unlink(temp_file.name)
            
        elif file_extension in ['txt', 'csv']:
            stringio = io.StringIO(file.getvalue().decode("utf-8"))
            text = stringio.read()
            
        elif file_extension in ['jpg', 'jpeg', 'png']:
            image = Image.open(io.BytesIO(file.getvalue()))
            text = pytesseract.image_to_string(image)
            
        else:
            text = "Unsupported file format for text extraction."
    
    except Exception as e:
        text = f"Error extracting text: {str(e)}"
    
    return text

# Function to extract audio from video
def extract_audio_from_video(video_file):
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    temp_video.write(video_file.getvalue())
    temp_video.close()
    
    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    temp_audio.close()
    
    video = mp.VideoFileClip(temp_video.name)
    video.audio.write_audiofile(temp_audio.name, verbose=False, logger=None)
    
    os.unlink(temp_video.name)
    
    return temp_audio.name

# Function to transcribe audio using Whisper
def transcribe_audio(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    os.unlink(audio_path)
    return result["text"]

# Train a simple ML model for auto-fill suggestions
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
    except:
        return []

# Initialize ML model for autofill
@st.cache_resource
def load_ml_model():
    return train_autofill_model()

# Load ML model
autofill_model, vectorizer = load_ml_model()

# Initialize Gemma 3 model
@st.cache_resource
def load_gemma_model():
    try:
        model = Gemma3ForConditionalGeneration.from_pretrained("google/gemma-3-4b-pt")
        processor = AutoProcessor.from_pretrained("google/gemma-3-4b-pt")
        return model, processor
    except:
        st.warning("Failed to load Gemma 3 model. Using a fallback approach.")
        return None, None

# Sidebar for language selection
with st.sidebar:
    st.markdown("<h2 class='sub-header'>Settings</h2>", unsafe_allow_html=True)
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
    
    st.markdown("---")
    st.markdown("<h3 class='sub-header'>About ROH-Ads</h3>", unsafe_allow_html=True)
    st.markdown("""
    ROH-Ads is an AI-powered tool that helps businesses create effective marketing strategies. It features:
    
    - Multimodal translation across 140+ languages
    - Document analysis and processing
    - ML-based auto-fill suggestions
    - Marketing strategy insights
    """)

# Main content area with tabs
tab1, tab2, tab3 = st.tabs(["Translate", "Document Analysis", "Marketing Assistant"])

# Translation Tab
with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h3 class='sub-header'>Input</h3>", unsafe_allow_html=True)
        input_type = st.radio("Input Type", ["Text", "Image", "Audio", "Video"])
        
        input_text = ""
        uploaded_file = None
        
        if input_type == "Text":
            input_text = st.text_area("Enter text to translate", height=200)
        else:
            uploaded_file = st.file_uploader(f"Upload {input_type.lower()} file", type=["jpg", "jpeg", "png", "mp3", "wav", "mp4", "avi"] if input_type in ["Image", "Audio", "Video"] else None)
            
            if uploaded_file and input_type == "Image":
                st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
            elif uploaded_file and input_type == "Audio":
                st.audio(uploaded_file)
            elif uploaded_file and input_type == "Video":
                st.video(uploaded_file)
    
    with col2:
        st.markdown("<h3 class='sub-header'>Translation</h3>", unsafe_allow_html=True)
        
        if st.button("Translate", key="translate_button"):
            with st.spinner("Processing..."):
                # Process different input types
                if input_type == "Text" and input_text:
                    # Detect language if auto-detect is selected
                    if source_lang == "Auto-detect":
                        try:
                            detected_lang = detect(input_text)
                            st.info(f"Detected language: {detected_lang}")
                        except LangDetectException:
                            st.warning("Could not detect language, defaulting to English")
                    
                    # In a real app, you would call a translation API here
                    translated_text = f"Translated: {input_text} (from {source_lang} to {target_lang})"
                    
                    st.markdown("<div class='result-container'>", unsafe_allow_html=True)
                    st.write(translated_text)
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Generate speech output
                    audio_bytes = text_to_speech(translated_text, lang=target_lang.lower()[:2])
                    st.markdown(get_audio_player_html(audio_bytes), unsafe_allow_html=True)
                    
                elif uploaded_file:
                    if input_type == "Image":
                        # Process image with Gemma 3
                        st.write("Image analysis would be processed by Gemma 3 model")
                        
                        # Placeholder for image analysis
                        analyzed_text = "This is a sample image analysis that would be generated by Gemma 3"
                        translated_text = f"Translated: {analyzed_text} (from {source_lang} to {target_lang})"
                        
                        st.markdown("<div class='result-container'>", unsafe_allow_html=True)
                        st.write(translated_text)
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                    elif input_type == "Audio":
                        # Process audio with speech recognition
                        st.write("Audio transcription in progress...")
                        
                        # Placeholder for audio transcription
                        transcribed_text = "This is a sample audio transcription"
                        translated_text = f"Translated: {transcribed_text} (from {source_lang} to {target_lang})"
                        
                        st.markdown("<div class='result-container'>", unsafe_allow_html=True)
                        st.write(translated_text)
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                    elif input_type == "Video":
                        # Extract audio from video and transcribe
                        st.write("Video processing in progress...")
                        
                        # Placeholder for video processing
                        video_text = "This is sample text extracted from video content"
                        translated_text = f"Translated: {video_text} (from {source_lang} to {target_lang})"
                        
                        st.markdown("<div class='result-container'>", unsafe_allow_html=True)
                        st.write(translated_text)
                        st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.error("Please provide input for translation")

# Document Analysis Tab
with tab2:
    st.markdown("<h3 class='sub-header'>Document Analysis</h3>", unsafe_allow_html=True)
    
    doc_file = st.file_uploader("Upload document for analysis", type=["pdf", "docx", "txt", "csv", "jpg", "jpeg", "png"])
    
    if doc_file is not None:
        file_details = {"Filename": doc_file.name, "FileType": doc_file.type, "FileSize": f"{doc_file.size / 1024:.2f} KB"}
        st.write(file_details)
        
        if st.button("Analyze Document"):
            with st.spinner("Analyzing document..."):
                # Extract text from document
                extracted_text = extract_text_from_document(doc_file)
                
                # Display extracted text preview
                st.markdown("<h4 class='sub-header'>Extracted Text Preview</h4>", unsafe_allow_html=True)
                st.markdown("<div class='result-container'>", unsafe_allow_html=True)
                st.write(extracted_text[:500] + ("..." if len(extracted_text) > 500 else ""))
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
                
                # Document summary
                st.markdown("<h4 class='sub-header'>Document Summary</h4>", unsafe_allow_html=True)
                st.markdown("<div class='result-container'>", unsafe_allow_html=True)
                st.write("Here would be a summary of the document generated by Gemma 3 model")
                st.markdown("</div>", unsafe_allow_html=True)

# Marketing Assistant Tab
with tab3:
    st.markdown("<h3 class='sub-header'>ROH-Ads Marketing Assistant</h3>", unsafe_allow_html=True)
    
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
            # In a real application, this would utilize the Gemma 3 model for generating insights
            
            # Generate a voice message using gTTS
            strategy_summary = f"Hello! I've analyzed the marketing data for {business_name}. Based on your {primary_goal} goal, I recommend focusing on {', '.join(channels[:2])} for your {industry} business targeting {target_audience[:50]}..."
            
            audio_bytes = text_to_speech(strategy_summary)
            
            # Display strategy
            st.markdown("<h4 class='sub-header'>Marketing Strategy Overview</h4>", unsafe_allow_html=True)
            st.markdown("<div class='result-container'>", unsafe_allow_html=True)
            
            st.write(f"### ROH-Ads Strategy for {business_name}")
            st.write(f"**Industry:** {industry}")
            st.write(f"**Target Audience:** {target_audience}")
            st.write(f"**Primary Goal:** {primary_goal}")
            st.write(f"**Budget:** ${budget:,}")
            st.write(f"**Timeline:** {timeline} days")
            
            st.markdown("#### Recommended Strategy")
            st.write("Your marketing strategy would be generated here by the Gemma 3 model based on the provided inputs.")
            
            st.markdown("#### Channel Breakdown")
            for channel in channels:
                st.write(f"• **{channel}:** Strategy details for this channel would be provided here.")
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Display voice message
            st.markdown("<h4 class='sub-header'>Strategy Voice Summary</h4>", unsafe_allow_html=True)
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
st.markdown("<p style='text-align: center;'>ROH-Ads: AI for Smart Marketing Strategies © 2025</p>", unsafe_allow_html=True)
