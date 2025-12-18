
import streamlit as st
from streamlit.testing.v1 import AppTest

def app():
    st.file_uploader("Upload", key="uploader")
    if st.session_state.get("uploader"):
        st.write("Uploaded!")

def test_uploader():
    at = AppTest.from_function(app)
    at.run()
    print(f"Uploader: {at.get('file_uploader')[0]}")
    try:
        at.get("file_uploader")[0].upload([__file__]).run()
        print("Success!")
    except Exception as e:
        print(f"Fail: {e}")

if __name__ == "__main__":
    test_uploader()
