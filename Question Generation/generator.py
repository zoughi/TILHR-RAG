from langchain.llms import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import pandas as pd

def initialize_llm():
    """Initialize the language model for FAQ generation."""
    print("🤖 Initializing LLM...")
    llm = ChatOllama(
        model="llama3.1",
        temperature=0.3
    )

    prompt = PromptTemplate(
        template="""
You are an FAQ Generator.
Based on the following Farsi text, generate a list of 5 relevant, independent, and informative Frequently Asked Questions (FAQs) in Farsi, along with their answers.
Rules:

    Language and Clarity
        All questions and answers must be written in Farsi.
        Be clear and concise. Avoid unnecessary complexity.
        Do not include any introductory text, explanations, or summaries in your output. Only output the questions and answers.

    Self-Contained and Explicit
        Each question must be fully self-contained and understandable on its own.
        Do not use vague references or pronouns like “این خدمت”، “آن‌ها”، “آن”، “او” یا “آن”.
        Always state the full subject explicitly. For example: “خدمت حمایت از محققین پسادکتری”، “دانشجویان مقطع دکتری”، etc.

    Based Only on Provided Text
        Only use information explicitly stated in the text.
        Do not infer, guess, or use outside knowledge.
        Do not ask about missing information (e.g., costs, numbers, data that are not in the text).

    Content Diversity
        Each FAQ must cover a different aspect of the content.
        Avoid asking multiple questions about the same concept or topic.

    User-Centered
        Focus on common or practical questions that real users might have.
        Questions should relate directly to the services, rules, or features described in the text.
    Note:
Do not include introductory text or answers in your output. Only output the questions.(do find the answers just don't include them)

{chunk}

Questions (in Farsi):

""",
        input_variables=["chunk"],
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain

def generate_faqs(chunks_df, chain):
    """Generate FAQs for each chunk."""
    print("🧠 Generating questions for each chunk...")
    faq_results = []
    faq_id = 1

    for i, row in chunks_df.iterrows():
        print(f"\n--- Chunk {i+1} ---")
        try:
            result = chain.invoke({"chunk": row["page_content"]})
            questions = [q.strip() for q in result.split("\n") if q.strip()]
            for q in questions:
                if q[0].isdigit():
                    q = q[q.find('.')+1:].strip()
                faq_results.append({
                    "id": faq_id,
                    "chunk_id": row["chunk_id"],
                    "source": row["source"],
                    "question": q,
                    "original_chunk": row["page_content"]
                })
                faq_id += 1
        except Exception as e:
            print(f"❌ Error generating for Chunk {i+1}: {e}")

    # Save FAQs
    print(f"\n✅ Total questions generated: {len(faq_results)}")
    faq_df = pd.DataFrame(faq_results)
    faq_df.to_csv("outputs/questions.csv", index=False, encoding='utf-8-sig')
    print("💾 Saved FAQs to 'Questions.csv'.")
