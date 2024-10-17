import validators, streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from langchain_huggingface import HuggingFaceEndpoint

# Streamlit App Configuration
st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader('Summarize URL')

# Sidebar for Hugging Face API Token
with st.sidebar:
    hf_api_key = st.text_input("Huggingface API Token", value="", type="password")

generic_url = st.text_input("Enter URL", label_visibility="collapsed")

# Check if the token exists and has no spaces
if hf_api_key.strip() == "":
    st.error("Hugging Face API Token is required.")
    st.stop()

# Initialize HuggingFace Model
try:
    repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
    llm = HuggingFaceEndpoint(
        repo_id=repo_id, 
        max_length=150, 
        temperature=0.7, 
        token=hf_api_key.strip()
    )
except Exception as e:
    st.error(f"Failed to initialize HuggingFace Endpoint: {e}")
    st.stop()

# Define Prompt Template
prompt_template = """
Provide a summary of the following content in 300 words:
Content: {text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

# Button to Trigger the Summarization Process
if st.button("Summarize the Content from YT or Website"):
    if not validators.url(generic_url):
        st.error("Invalid URL. Please enter a valid YouTube or website URL.")
    else:
        try:
            with st.spinner("Processing..."):
                # Load Data from the URL or YouTube
                if "youtube.com" in generic_url:
                    loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=True)
                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        headers={
                            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) "
                                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                                          "Chrome/116.0.0.0 Safari/537.36"
                        }
                    )
                docs = loader.load()

                # Run the Summarization Chain
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                output_summary = chain.run(docs)

                # Display the Summary
                st.success(output_summary)

        except Exception as e:
            # Handle Exceptions Gracefully
            st.error(f"Exception: {e}")
            st.stop()

