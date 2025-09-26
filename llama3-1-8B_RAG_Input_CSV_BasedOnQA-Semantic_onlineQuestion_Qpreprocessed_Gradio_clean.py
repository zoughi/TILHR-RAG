#------------------------------------------------------------------------------
# Step 0: Import required libraries and initialize constants and thresholds
#------------------------------------------------------------------------------
import time
from typing import Optional, List
from langchain.schema import Document
import glob
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import gradio as gr
from langchain.embeddings import HuggingFaceEmbeddings
import unicodedata
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Thresholds and model initialization
qa_similarity_threshold = 0.87 # آستانه شباهت برای لایه اول (have been tested)
retriever_similarity_threshold = 0.85 # آستانه شباهت برای لایه دوم (have been tested)
init_temperature = 0.3 # مقادیز بیشتر برای پاسخ‌های خلاقانه‌تر (بازه 0-2)
#embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")
embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
client_embedding_model = embedding_model.client  # استفاده از همان مدل موجود در HuggingFaceEmbeddings

#------------------------------------------------------------------------------
# Step 1: Preprocess questions (remove punctuation and normalize whitespace)
#------------------------------------------------------------------------------
def preprocess_question(text: str) -> str:
    text = re.sub(r'[\d\.\،\;\:\?\!\"\(\)\[\]«»]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

#------------------------------------------------------------------------------
# Step 2: Load and embed FAQ questions
#------------------------------------------------------------------------------
print("Loading FAQ data...")
str1 = "articles/QA_files/FAQ_Original_new29.csv"
print(str1)
qa_df = pd.read_csv(str1)
questionFAQ_embeddings = client_embedding_model.encode(qa_df['question'].tolist())

#------------------------------------------------------------------------------
# Step 3: Load and convert CSV chunks to Document objects
#------------------------------------------------------------------------------
csv_files = glob.glob("articles/csv_files/chunks_Original_new29.csv")
doc_splits = []

for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    for _, row in df.iterrows():
        doc_splits.append(Document(
            page_content=row['page_content'],
            metadata={'source': row['source']}
        ))

#------------------------------------------------------------------------------
# Step 4: Custom retriever with similarity threshold
#------------------------------------------------------------------------------
class ThresholdRetriever:
    def __init__(self, documents,client_embedding_model):
        self.documents = documents
        self.client_embedding_model = client_embedding_model
        self.doc_embeddings = client_embedding_model.encode([doc.page_content for doc in documents])

    def invoke(self, query: str, k: int = 5) -> List[tuple[Document, float]]:
        query_embedding = self.client_embedding_model.encode([query])
        similarities = cosine_similarity(query_embedding, self.doc_embeddings)[0]
        top_indices = np.argsort(similarities)[-k:][::-1]
        results = [(self.documents[i], similarities[i]) for i in top_indices if similarities[i] >= retriever_similarity_threshold]
        print(f"Top similarities: {similarities[top_indices]}")
        return results

retriever = ThresholdRetriever(documents=doc_splits,client_embedding_model=client_embedding_model)

#------------------------------------------------------------------------------
# Step 5: Prompt setup for each layer of RAG
#------------------------------------------------------------------------------
qa_prompt = PromptTemplate(
    template="""بر اساس مستندات زیر به سوال پاسخ دقیق بده. فقط از اطلاعات موجود استفاده کن.
        1. پاسخ را روان، طبیعی، رسمی و کاملا فارسی بیان کنید

    متن مرجع:
    {reference_text}
    
    سوال کاربر:
    {question}
    
پاسخ دقیق و رسمی:""",
    input_variables=["reference_text", "question"],
)

search_prompt = PromptTemplate(
    template="""بر اساس مستندات زیر به سوال پاسخ دقیق بده. فقط از اطلاعات موجود استفاده کن.
        1. پاسخ را روان، طبیعی، رسمی و کاملا فارسی بیان کنید

    **سوال کاربر**: 
    {question}
    
    **مستندات مرتبط**:
    {documents}
پاسخ دقیق و رسمی:""",
    input_variables=["question", "documents"],
)

general_prompt = PromptTemplate(
    template="""
شما یک دستیار گفت‌ و گوگر فارسی‌ زبان، مودب، حرفه‌ای و **با شخصیت انسانی (مفرد)** هستید که توانایی پاسخ به سوالات عمومی، اجتماعی و دانشی را دارید.

دستورالعمل‌ها:

1. شما باید فقط با ضمیر «من» صحبت کنید و از افعال اول‌ شخص مفرد استفاده کنید. از افعال جمع (مثل "می‌کنیم"، "هستیم") استفاده نکنید.
2. اگر سوال حالت احوال‌پرسی یا اجتماعی دارد (مثلاً «سلام»، «خوبی؟»، «چه خبر؟»، «تو کی هستی؟») پاسخ شما باید:
   - به زبان فارسی معیار، محترمانه، طبیعی و دوستانه باشد.
   - از واژگان بیگانه (مثل «مرحبا») استفاده نکنید.
   - جمله‌ بندی روان و ضمیر و فعل هم‌ راستا باشند.
   - از پرسیدن سؤال غیرضروری (مثل «چی خبر؟») خودداری شود.
3. اگر سوالات پربوط به پژوهشگاه ارتباطات و فناوری اطلاعات (یا پژوهشگاه مخابرات) نبود،  پاسخ: لطفا سوالات مرتبط با پژوهشگاه مخابرات بپرسید تا بتوانم به شما پاسخ دهم
4. پاسخ فقط باید به زبان فارسی باشد و از هیچ زبان دیگری استفاده نشود.

مثال‌ها:
- سوال: سلام → پاسخ: سلام! وقت بخیر، چگونه می‌توانم به شما کمک کنم؟
- سوال: خوبی؟ → پاسخ: ممنونم،لطفا سوالات مرتبط با پژوهشگاه مخابرات بپرسید تا بتوانم به شما پاسخ دهم.
- سوال: چطوری؟ یا چطوری → پاسخ: ممنونم، لطفا سوالات مرتبط با پژوهشگاه مخابرات بپرسید تا بتوانم به شما پاسخ دهم.

اکنون لطفاً به پرسش زیر پاسخ دهید:

سوال:
{question}

پاسخ:""",
    input_variables=["question"],
)


llm = ChatOllama(model="llama3.1", temperature=init_temperature)
qa_chain = qa_prompt | llm | StrOutputParser()
search_chain = search_prompt | llm | StrOutputParser()
general_chain = general_prompt | llm | StrOutputParser()

#------------------------------------------------------------------------------
# Step 6: Main RAG application with three fallback layers
#------------------------------------------------------------------------------
class IntelligentRAGApplication:
    def __init__(self, retriever, qa_chain, search_chain, general_chain, qa_df, questionFAQ_embeddings, client_embedding_model):
        self.retriever = retriever
        self.qa_chain = qa_chain
        self.search_chain = search_chain
        self.general_chain = general_chain
        self.qa_df = qa_df
        self.questionFAQ_embeddings = questionFAQ_embeddings
        self.client_embedding_model = client_embedding_model


    def find_most_similar_question(self, processed_question: str) -> tuple[Optional[str], float]:
        question_embedding = self.client_embedding_model.encode([processed_question])
        similarities = cosine_similarity(question_embedding, self.questionFAQ_embeddings)
        max_index = np.argmax(similarities)
        max_similarity = similarities[0][max_index]
        if max_similarity >= qa_similarity_threshold:
            return self.qa_df.iloc[max_index]['original_chunk'], max_similarity
        return None, max_similarity

    def run(self, question: str) -> tuple[str, str]:
        source = None
        processed_question = preprocess_question(question)

        # Layer 1: FAQ-based matching
        reference_text, similarity_score = self.find_most_similar_question(processed_question)
        print(f"QA similarity_score: {similarity_score}")
        if reference_text:
            source = self.qa_df.iloc[np.argmax(
                cosine_similarity(
                    self.client_embedding_model.encode([processed_question]),
                    self.questionFAQ_embeddings
                )[0]
            )]['source']
            print("[Layer 1 - FAQ match]")
            answer = self.qa_chain.invoke({"reference_text": reference_text, "question": processed_question})
            return answer, source

        # Layer 2: Semantic retrieval
        docs_and_scores = self.retriever.invoke(processed_question)
        if docs_and_scores:
            top_docs = docs_and_scores[:5]# فقط 5 سند برتر
            best_score = top_docs[0][1]
            if best_score >= retriever_similarity_threshold:
                sources = list({doc.metadata['source'] for doc, _ in top_docs})
                source = " | ".join(sources)
                print("[Layer 2 - Semantic search]")
                doc_texts = "\n\n---\n\n".join([doc.page_content for doc, _ in top_docs])
                answer = self.search_chain.invoke({"question": processed_question, "documents": doc_texts})
                return answer, source

        # Layer 3: General knowledge fallback
        print("[Layer 3 - General knowledge fallback]")
        return self.general_chain.invoke({"question": processed_question}), "دانش عمومی مدل"

rag_application = IntelligentRAGApplication(
    retriever=retriever,
    qa_chain=qa_chain,
    search_chain=search_chain,
    general_chain=general_chain,
    qa_df=qa_df,
    questionFAQ_embeddings=questionFAQ_embeddings,
    client_embedding_model=client_embedding_model
    
)

#------------------------------------------------------------------------------
# Step 7: Gradio interface
#------------------------------------------------------------------------------
def clean_output(text: str) -> str:
    try:
        return unicodedata.normalize("NFKC", text).encode("utf-8", errors="ignore").decode("utf-8")
    except Exception:
        return text

def chat_fn(message, history):
    if not message.strip():
        return "لطفاً سوال معتبری وارد کنید."
    try:
        start_time = time.time()
        answer, source = rag_application.run(message)
        answer = clean_output(answer)
        elapsed_time = time.time() - start_time
        return f"پاسخ:\n{answer}\n\nمنبع: {source}\n⏱️ زمان پاسخ‌دهی: {elapsed_time:.2f} ثانیه"
    except Exception as e:
        return f"⚠️ خطا در پردازش سوال: {str(e)}"

with gr.Blocks(css="""
#custom-textbox textarea {
    background-color: #f3f0ff !important;
    border: 1px solid #ccc !important;
    direction: rtl;
    text-align: right;
}
.footer {
    text-align: center;
    color: gray;
    margin-top: 1rem;
    direction: rtl;
}
""") as demo:
    gr.ChatInterface(
        fn=chat_fn,
        title="🤖 چت‌بات پژوهشگاه ارتباطات و فناوری اطلاعات 🤖",
        chatbot=gr.Chatbot(label="گفتگو با چت‌بات", rtl=True),
        textbox=gr.Textbox(
            placeholder="سوال خود را وارد کنید...",
            rtl=True,
            container=True,
            elem_id="custom-textbox"
        ),
        description="برای خروج، عبارت 'خروج' یا 'پایان' را وارد کنید."
    )
    gr.HTML("<div class='footer'>Developed by Toktam Zoughi</div>")

demo.launch()
