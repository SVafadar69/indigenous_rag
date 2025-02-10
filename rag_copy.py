import streamlit as st
import os
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders.image import UnstructuredImageLoader
import PyPDF2
import base64
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import Chroma
from groundx import Document, GroundX
from groq import Groq
from dotenv import load_dotenv
from PIL import Image
import tempfile
from IPython.display import clear_output
import html
load_dotenv()

client = GroundX(api_key=os.getenv("GROUNDX_API_KEY"))
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# Lookup document IDs
document_id_1 = client.documents.lookup(id=14537).documents[0].document_id
document_id_2 = client.documents.lookup(id=14538).documents[0].document_id

# flip image 
image_path = "assets/indigenous_background.jpg"
def flip_image(image_path): 

    image = Image.open(image_path)
    flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)
    return flipped_image

flipped_image = flip_image(image_path=image_path)

# save flipped image to temporary file 
def temporary_file(flipped_image): 
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file: 
        flipped_image.save(temp_file.name)  
        temp_flipped_image_path = temp_file.name 

    return temp_flipped_image_path

temporary_file_image_path = temporary_file(flipped_image=flipped_image)

# Function to encode the local image into base64
def get_base64_image(image_path):
    with open(image_path, "rb") as file:
        encoded = base64.b64encode(file.read()).decode()
    return encoded

# Path to your local image
background_image = get_base64_image(temporary_file_image_path)

#########################################################
# Helper Functions
#########################################################
def search_over_both_textbooks(user_query, document_id_1, document_id_2):
    dual_search = client.search.documents(
        query=user_query,
        document_ids=[document_id_1, document_id_2],
        verbosity=2,
        n=1
    )
    return dual_search.search.text

def llm_call_groq(documents_retrieved, user_query):
    response_final = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": f"Using the following text: {documents_retrieved}, "
                           f"answer the following question: {user_query}. "
                           "Keep your answer short and concise. Only use enough words that is necessary. Only use answers \
                           from the retrieved text."
            }
        ],
        temperature=1,
        max_tokens=300
    )
    return response_final.choices[0].message.content


st.set_page_config(page_title="Indigenous Class Q&A", page_icon="🌏", layout="wide")
# Inject CSS to set the background image
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Poppins:wght@400;600&display=swap');
    .stApp {{
        background-image: url("data:image/jpeg;base64,{background_image}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}

    .stButton > button {{
        background: linear-gradient(135deg, #3e8c20, #56ab2f) !important;
        color: white !important;
        font-weight: bold !important;
        border-radius: 8px !important;
        padding: 12px 24px !important;
        border: none !important;
        cursor: pointer;
        transition: background 0.3s ease;
        font-family: 'Poppins', sans-serif;
    }}
    .stButton > button:hover {{
        background: linear-gradient(135deg, #2e6c17, #3e8c20) !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Move query box more to the left using columns
col1, col2 = st.columns([4, 3.5])  # Adjust column ratios for alignment

with col2:
    # Place all elements in the wider left column
    st.markdown(
        f"<h1 style='color: #8B0000; font-family: Playfair Display, serif; font-size: 3rem;'>Textbook Question Answering </h1>",
        unsafe_allow_html=True
    )

    # Subheader with Forest Green
    st.markdown(
        f"<h3 style='color: #2E8B57; font-family: Poppins, sans-serif;'>Type/Paste a question below to search through the textbook. Try to make your question as specific as possible. </h3>",
        unsafe_allow_html=True
    )

    # Multiline Text Input
    user_query = st.text_area("", height=120)

    # Add a "Submit" button
    submit_button = st.button("Submit")

    if submit_button and user_query.strip():
        retrieved_answer_final = search_over_both_textbooks(
            user_query=user_query,
            document_id_1=document_id_1,
            document_id_2=document_id_2
        )

        final_answer = llm_call_groq(
            documents_retrieved=retrieved_answer_final,
            user_query=user_query
        )
        formatted_answer = html.escape(final_answer).replace("\r\n", "\n").replace("\n", "<br>").strip()
        st.markdown(
                "<h4 style='color: #8B0000; font-family: Poppins, sans-serif;'>Answer:</h4>",
                unsafe_allow_html=True
            )
        st.markdown(f"<p style='color: #2E8B57;'>{formatted_answer}</p>", unsafe_allow_html=True)  # Forest Green

 
    st.markdown("</div>", unsafe_allow_html=True)
