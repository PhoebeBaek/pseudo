import os
import streamlit as st
import mcp_client as mc
import asyncio

# Intilize stremalit web UI
st.markdown("<h1 style='text-align: center;'>Demo Webpage 🍃</h1>", unsafe_allow_html=True)
st.markdown("<br><br>", unsafe_allow_html=True)


image_uploader = st.file_uploader(label="image",  type="png")
go_button = st.button("Analyze Image", type="primary")

# Call agent to use tools
if go_button:
    if image_uploader is not None:
        image_path = os.path.join("image", image_uploader.name)
        with open(image_path, "wb") as f:
            f.write(image_uploader.getbuffer())
            
        with st.spinner("Working..."):
            response = asyncio.run(mc.menu_analyze_agent(image_path))
            st.write(response)
    else:
        st.write("Upload an image!")