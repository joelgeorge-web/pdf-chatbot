import streamlit as st
import os
import time
import fitz  # PyMuPDF library
import torch    
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

current_dir = os.getcwd()
save_file_name = None  # Initialize the global variable

def streamlit_file_loader():
    global save_file_name  # Declare the global variable
    st.title("PDF Reader")
    st.write("Upload a PDF file to read its content.")
    uploaded_file = st.file_uploader("Choose a file to upload", type=['txt', 'pdf', 'docx'])
    
    if uploaded_file is not None:
        filename = uploaded_file.name
        safe_filename = os.path.splitext(filename)[0] + "_" + os.path.splitext(filename)[1]
        save_file_name = os.path.join(current_dir, safe_filename)

        try:
            with open(save_file_name, "wb") as buffer:
                buffer.write(uploaded_file.getbuffer())
            st.success(f"Successfully uploaded and saved {filename}")
        except Exception as e:
            st.error(f"Error saving file: {e}")

def read_pdf():
    if save_file_name and save_file_name.lower().endswith('.pdf') or save_file_name and save_file_name.lower().endswith('.txt'):
        try:
            pdf_document = fitz.open(save_file_name)
            num_pages = len(pdf_document)
            for page_num in range(num_pages):
                page = pdf_document[page_num]
                text = page.get_text()
                print(f"Page {page_num + 1}:\n{text}")
            pdf_document.close()
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
    else:
        st.warning("No file uploaded or incorrect file format.")



def chatbot():


    # Streamed response emulator
    def response_generator():
        # Load model directly
        set_seed(2024)  

        model_checkpoint = "microsoft/Phi-3-mini-4k-instruct"

        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint,trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_checkpoint,
                                                    trust_remote_code=True,
                                                    torch_dtype="auto")
        print(prompt)
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, do_sample=True, max_new_tokens=500)
        response= tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(response)
        for word in response.split():
            yield word + " "
            time.sleep(0.05)


    st.title("Ask your questions here!")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Message Chatbot..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            response = st.write_stream(response_generator())
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})


def main():
    streamlit_file_loader()
    read_pdf()
    chatbot()

if __name__ == "__main__":
    main()
