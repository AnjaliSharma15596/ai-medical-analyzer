import os
from dotenv import load_dotenv
import PyPDF2
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

load_dotenv()


def extract_text_from_pdf(pdf_file):
    try:
        import fitz
        import io
        
        pdf_bytes = pdf_file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        # Pehle normal text try karo
        text = ""
        for page in doc:
            text += page.get_text() + "\n"
        doc.close()
        
        # Agar text nahi mila toh OCR use karo
        if len(text.strip()) < 50:
            import pytesseract
            from pdf2image import convert_from_bytes
            
            pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
            
            print("OCR mode: Converting pages to images...")
            images = convert_from_bytes(pdf_bytes, dpi=150)
            
            text = ""
            for i, img in enumerate(images):
                print(f"Processing page {i+1}...")
                text += pytesseract.image_to_string(img) + "\n"
        
        if not text or len(text.strip()) < 50:
            return "Error: Could not extract text"
        
        return text
        
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

def create_vector_store(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_texts(chunks, embeddings)
    return vector_store


def create_conversation_chain(vector_store):
    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant",
        temperature=0.3,
        timeout=30,       # give up waiting after 30 seconds instead of hanging forever
        max_retries=2      # automatically retry up to 2 times on transient failures
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    prompt_template = """You are a helpful medical report assistant.
    Your job is to explain medical reports in simple easy-to-understand language.

    STRICT RULES:
    - Always explain medical terms in simple language
    - Highlight ABNORMAL values clearly with ⚠️ ABNORMAL
    - Mark normal values with ✅ NORMAL
    - Always recommend consulting a doctor for serious findings
    - Never diagnose — only explain what the report says
    - Be empathetic and calm in your responses
    - Structure your response clearly with headings
    - ONLY use information present in the Context below. If the Context does not contain
      information needed to answer the Question, clearly say "This information is not
      present in the uploaded report" instead of guessing or using outside knowledge

    Context from medical report:
    {context}

    Chat History:
    {chat_history}

    Question: {question}

    Answer in a clear, simple and structured way:"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "chat_history", "question"]
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": PROMPT},
        return_source_documents=False,
        verbose=False
    )

    return chain


def analyze_report(chain, question):
    try:
        response = chain({"question": question})
        return response["answer"]
    except Exception as e:
        error_text = str(e).lower()

        # Distinguish common failure types so the user gets a clear, honest message
        if "timeout" in error_text or "timed out" in error_text:
            return ("⚠️ The AI is taking too long to respond right now. "
                    "This usually means the service is under heavy load. Please try again in a moment.")
        elif "rate limit" in error_text or "429" in error_text:
            return ("⚠️ Too many requests right now — please wait a few seconds and try again.")
        elif "connection" in error_text or "network" in error_text:
            return ("⚠️ Couldn't reach the AI service. Please check your connection and try again.")
        else:
            # Fallback for anything unexpected — still user-friendly, no raw stack trace shown
            return ("⚠️ Something went wrong while analyzing your question. Please try again, "
                    "or try re-uploading the report if the issue continues.")


def get_initial_analysis(chain):
    initial_question = """Please analyze this medical report and provide:
    1. 📋 REPORT SUMMARY — What type of report is this?
    2. ✅ NORMAL VALUES — List all values that are normal
    3. ⚠️ ABNORMAL VALUES — List all values that are outside normal range
    4. 🔍 KEY FINDINGS — What are the most important findings?
    5. 💊 RECOMMENDATIONS — What should the patient discuss with their doctor?

    Please be clear, simple and use easy language."""

    return analyze_report(chain, initial_question)
