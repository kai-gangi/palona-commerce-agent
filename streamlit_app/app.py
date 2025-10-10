"""Streamlit frontend for AI Commerce Agent.

This module provides a user-friendly web interface for the AI Commerce Agent
using Streamlit. It offers both text-based and image-based product search
capabilities through an intuitive chat interface.

Features:
    - Chat-based interaction with the AI agent
    - Image upload for visual product search
    - Product recommendations display
    - Conversation history management
    - Real-time API communication

The interface supports three main use cases:
    1. General conversation with the AI agent
    2. Text-based product recommendations
    3. Image-based product search and similarity matching

Example:
    Run the Streamlit app:
        $ streamlit run streamlit_app/app.py
        
    Then navigate to the provided URL to interact with the agent.

Dependencies:
    - streamlit: Web app framework
    - requests: HTTP client for API communication
    - PIL: Image processing for uploads
    - base64: Image encoding for API transmission
"""

import streamlit as st
import requests
import base64
from PIL import Image
import io
import json

API_URL = "http://localhost:8000/api/chat"
STREAM_URL = "http://localhost:8000/api/chat/stream"

st.set_page_config(
    page_title="ShopBot",
    page_icon="🛍️",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": """Hello! I'm **ShopBot**, your AI shopping assistant! I can help you:

- **Chat** - Ask me anything or get to know me better
- **Find Products** - Describe what you're looking for (e.g., "comfortable running shoes")
- **Search by Image** - Upload a photo to find similar products

What would you like to explore today?"""
        }
    ]

if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None

if "processing_image_search" not in st.session_state:
    st.session_state.processing_image_search = False

def send_message_to_api(message: str, history: list, image_data: str = None):
    """Centralized function to handle streaming API calls."""
    try:
        # Create placeholder for loading/streaming response
        assistant_message = {
            "role": "assistant",
            "content": ""
        }
        st.session_state.messages.append(assistant_message)
        
        # Create container for loading and streaming content
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("🤔 Thinking...")
            
            response = requests.post(
                STREAM_URL,
                json={
                    "message": message,
                    "history": history,
                    "image": image_data
                },
                timeout=60,
                stream=True
            )
        
        if response.status_code == 200:
            full_response = ""
            
            # Process streaming response
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data_str = line[6:]  # Remove 'data: ' prefix
                        
                        if data_str == '[DONE]':
                            break
                            
                        try:
                            chunk = json.loads(data_str)
                            
                            if chunk["type"] == "content":
                                full_response += chunk["content"]
                                message_placeholder.markdown(full_response + "▌")
                            
                            elif chunk["type"] == "complete":
                                # Update the stored message
                                st.session_state.messages[-1]["content"] = full_response
                                message_placeholder.markdown(full_response)
                                
                                # ONLY store products in session state, don't display here
                                if chunk.get("products"):
                                    st.session_state.messages[-1]["products"] = chunk["products"]
                                
                                # Force rerun to display products in chat history
                                st.rerun()
                            
                            elif chunk["type"] == "error":
                                message_placeholder.error(chunk["content"])
                                break
                                
                        except json.JSONDecodeError:
                            continue
            
            # Clear the uploaded image
            if "uploaded_image" in st.session_state:
                st.session_state.uploaded_image = None
            
        else:
            message_placeholder.error(f"API Error: {response.status_code}")
            st.error(response.text)
    
    except requests.exceptions.RequestException as e:
        message_placeholder.error(f"Connection Error: {str(e)}")
        st.info("Make sure the backend server is running on http://localhost:8000")
    except Exception as e:
        message_placeholder.error(f"Error: {str(e)}")

# Header
st.title("🛍️ ShopBot")
st.markdown("*Your intelligent shopping assistant - Search by text or image!*")

# Sidebar
with st.sidebar:
    st.header("About")
    st.markdown("""
    This AI agent can help you:
    - 💬 Have general conversations
    - 🔍 Find products by description
    - 📸 Search by image
    
    **Try asking:**
    - "What's your name?"
    - "Show me sports t-shirts"
    - "I need running shoes"
    - Upload an image to find similar items
    """)
    
    st.divider()
    
    # Image upload section
    st.header("🖼️ Image Search")
    uploaded_file = st.file_uploader(
        "Upload an image to find similar products",
        type=["jpg", "jpeg", "png"],
        help="Upload a product image"
    )
    
    if uploaded_file:
        st.session_state.uploaded_image = uploaded_file
        st.image(uploaded_file, caption="Uploaded Image", width="stretch")
        st.success("✅ Image attached! Type a message or click 'Search by Image'")

        if st.button("🔍 Search by Image", use_container_width=True):
            # Convert image to base64
            image = Image.open(uploaded_file)
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()

            # Add user message with image
            user_message = {
                "role": "user",
                "content": "Find products similar to this image",
                "image": img_str,
                "image_file": uploaded_file  # Store for display
            }
            st.session_state.messages.append(user_message)
            st.session_state.processing_image_search = True
            
            # Trigger rerun to show the message in main chat and process streaming
            st.rerun()

    st.divider()
    
    # Clear chat button
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = [st.session_state.messages[0]]  # Keep welcome message
        if "uploaded_image" in st.session_state:
            st.session_state.uploaded_image = None
        st.rerun()

# Chat history display
chat_container = st.container()

with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display attached image if present
            if "image_file" in message:
                st.image(message["image_file"], width=200, caption="Attached image")
            
            # Display products if present
            if "products" in message and message["products"]:
                st.divider()
                st.subheader("🛒 Products Found:")
                
                # Display products in grid
                for i in range(0, len(message["products"]), 3):
                    cols = st.columns(3)
                    for idx, product in enumerate(message["products"][i:i+3]):
                        with cols[idx]:
                            with st.container(border=True):
                                try:
                                    img = Image.open(product["image_path"])
                                    st.image(img, width="stretch")
                                except:
                                    st.info("📦 Product Image")
                                
                                st.markdown(f"**{product['name']}**")
                                st.markdown(f"💰 ${product['price']}")
                                st.caption(f"Category: {product['category']}")
                                
                                with st.expander("View Details"):
                                    st.write(product['description'])
                                    st.write(f"**Tags:** {', '.join(product['tags'])}")

# Chat input
if prompt := st.chat_input("Ask me anything about products..."):
    # Prepare user message
    user_message = {
        "role": "user",
        "content": prompt
    }
    
    # Get image data if image is uploaded
    image_data = None
    if st.session_state.get("uploaded_image"):
        image = Image.open(st.session_state.uploaded_image)
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        image_data = base64.b64encode(buffered.getvalue()).decode()
        user_message["image"] = image_data
        user_message["image_file"] = st.session_state.uploaded_image  # Store for display
    
    st.session_state.messages.append(user_message)
    
    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(prompt)
        if st.session_state.get("uploaded_image"):
            st.image(st.session_state.uploaded_image, width="stretch", caption="Attached image")
    
    # Prepare history (exclude current message and products)
    history = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in st.session_state.messages[:-1]  
        if "products" not in msg  
    ]
    
    # Call streaming API
    send_message_to_api(prompt, history, image_data)

# Check if we need to process an image search from the sidebar button
if (st.session_state.get("processing_image_search", False) and 
    len(st.session_state.messages) > 0):
    
    last_message = st.session_state.messages[-1]
    if (last_message["role"] == "user" and 
        last_message["content"] == "Find products similar to this image" and 
        "image" in last_message):
        
        # Reset the flag to prevent duplicate processing
        st.session_state.processing_image_search = False
        
        # Prepare history and call streaming API
        history = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in st.session_state.messages[:-1]  
            if "products" not in msg  
        ]
        
        send_message_to_api(
            last_message["content"], 
            history, 
            last_message["image"]
        )

st.divider()
st.caption("Built by Kai Emilio Gangi with ❤️ to Palona")