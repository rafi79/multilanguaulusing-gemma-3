import streamlit as st
import os
import io
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="Gemma 3 Translator",
    page_icon="🌐",
    layout="wide"
)

# Set Hugging Face token
HF_TOKEN = "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"
os.environ["HF_TOKEN"] = HF_TOKEN

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

# Header
st.markdown("<h1 class='main-header'>Gemma 3 Multimodal Translation</h1>", unsafe_allow_html=True)
st.markdown("<p class='info-text'>AI-powered multimodal translation with Gemma 3</p>", unsafe_allow_html=True)

# Main app logic
def load_transformers():
    try:
        import torch
        from transformers import pipeline, AutoProcessor, AutoModelForCausalLM
        st.success("Successfully imported transformers and torch!")
        return True
    except ImportError as e:
        st.error(f"Import error: {str(e)}")
        return False

# Sidebar
with st.sidebar:
    st.markdown("<h2 class='sub-header'>Settings</h2>", unsafe_allow_html=True)
    
    # Model loading control
    check_imports = st.button("Check Transformers Imports")
    if check_imports:
        success = load_transformers()
        if success:
            st.info("Transformers is properly installed!")
        else:
            st.warning("Please check the logs for import errors")
    
    # Language selection
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
    
    # About
    st.markdown("---")
    st.markdown("<h3 class='sub-header'>About</h3>", unsafe_allow_html=True)
    st.markdown("""
    This app uses the Gemma 3 model from Google for multimodal translation.
    """)
    
    # Show HF token status
    st.markdown("---")
    st.markdown("<h3 class='sub-header'>HF Token Status</h3>", unsafe_allow_html=True)
    st.write("HF Token: " + ("✅ Configured" if HF_TOKEN else "❌ Missing"))
    
# Tabs
tab1, tab2 = st.tabs(["Basic Demo", "Import Status"])

# Basic Demo Tab
with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h3 class='sub-header'>Input</h3>", unsafe_allow_html=True)
        input_type = st.radio("Input Type", ["Text", "Image", "Combined (Image + Text)"])
        
        if input_type == "Text":
            input_text = st.text_area("Enter text to translate", height=200)
        elif input_type == "Image":
            uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
            if uploaded_file:
                try:
                    img = Image.open(uploaded_file)
                    st.image(img, caption="Uploaded Image", use_column_width=True)
                except Exception as e:
                    st.error(f"Error loading image: {str(e)}")
        else:  # Combined
            uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
            if uploaded_file:
                try:
                    img = Image.open(uploaded_file)
                    st.image(img, caption="Uploaded Image", use_column_width=True)
                except Exception as e:
                    st.error(f"Error loading image: {str(e)}")
            input_text = st.text_area("Enter text prompt", value="<start_of_image> In this image, there is")
    
    with col2:
        st.markdown("<h3 class='sub-header'>Output</h3>", unsafe_allow_html=True)
        
        if st.button("Process"):
            st.info("This is a placeholder for Gemma 3 model output.")
            st.markdown("<div class='result-container'>", unsafe_allow_html=True)
            
            if input_type == "Text":
                st.write("Text translation would appear here when the model is connected.")
            elif input_type == "Image":
                st.write("Image analysis would appear here when the model is connected.")
            else:  # Combined
                st.write("Image and text combined analysis would appear here when the model is connected.")
            
            st.markdown("</div>", unsafe_allow_html=True)

# Import Status Tab
with tab2:
    st.markdown("<h3 class='sub-header'>Import Diagnostic Information</h3>", unsafe_allow_html=True)
    
    st.markdown("### Base Imports")
    try:
        import sys
        st.success("✅ Successfully imported sys")
    except ImportError:
        st.error("❌ Failed to import sys")
    
    try:
        import PIL
        st.success(f"✅ Successfully imported PIL (version: {PIL.__version__})")
    except ImportError:
        st.error("❌ Failed to import PIL")
    except AttributeError:
        st.success("✅ Successfully imported PIL (version unknown)")
    
    st.markdown("### Machine Learning Imports")
    try:
        import torch
        st.success(f"✅ Successfully imported torch (version: {torch.__version__})")
        st.write(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            st.write(f"CUDA device: {torch.cuda.get_device_name(0)}")
    except ImportError:
        st.error("❌ Failed to import torch")
    
    try:
        import transformers
        st.success(f"✅ Successfully imported transformers (version: {transformers.__version__})")
    except ImportError:
        st.error("❌ Failed to import transformers")
    
    try:
        from transformers import pipeline
        st.success("✅ Successfully imported transformers.pipeline")
    except ImportError:
        st.error("❌ Failed to import transformers.pipeline")
    
    # Python environment info
    st.markdown("### Python Environment")
    st.write(f"Python version: {sys.version}")
    st.write(f"Platform: {sys.platform}")
    
    # Display paths
    st.markdown("### System Paths")
    st.code("\n".join(sys.path))

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center;'>Gemma 3 Multimodal Translation © 2025</p>", unsafe_allow_html=True)
