import pickle
import os
import cv2
import streamlit as st 
from app_files.utils import Processor

st.title("Sudoku Solver")

st.subheader("Upload a sudoku board")

uploaded_file = st.file_uploader("Choose a Sudoku board file", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    st.write(f"Name: {uploaded_file.name}")
    st.write(f"Tye: {uploaded_file.type}")
    st.write(f"Size: {uploaded_file.size / 1e6} MB")

    with open(os.path.join("assets", str(uploaded_file.name)), "wb") as f:
        f.write(uploaded_file.read())

    st.success("File uploaded successfully")
    save_path = "assets/" + str(uploaded_file.name)
    st.image(save_path, width=250)

    io_path = os.path.join("assets", str(uploaded_file.name))
    proc = Processor()

    res = proc.process(io_path)
    
    if res is not None:

        st.subheader(f"Your solution is ready:")

        display = res / 255
        st.image(display, width=300)
    
    else:
        st.subheader("Couldnot find a solution!")

    os.remove(os.path.join("assets", str(uploaded_file.name)))