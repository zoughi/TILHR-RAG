import os
import glob
import pandas as pd
from RecursiveCharacterTextSplitter import RecursiveCharacterTextSplitter

def load_md_files(md_folder="Raw_materials/md"):
    """Read all .md files from the specified folder and prepare them for processing."""
    print("📥 Loading .md files from 'Raw_materials/md'...")
    documents = []

    for filepath in glob.glob(os.path.join(md_folder, "*.md")):
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content:
                filename = os.path.splitext(os.path.basename(filepath))[0]
                
                # Clean repeated segments in filename (e.g., foo_foo_bar ➜ foo_bar)
                filename_parts = filename.split('_')
                cleaned_parts = []
                for part in filename_parts:
                    if not cleaned_parts or part != cleaned_parts[-1]:
                        cleaned_parts.append(part)
                cleaned_filename = '_'.join(cleaned_parts)

                documents.append({
                    "content": content,
                    "source": cleaned_filename
                })

    print("✅ Number of loaded documents:", len(documents))
    return documents

def split_documents_into_chunks(documents, chunk_size=800, chunk_overlap=200):
    """Split each document into overlapping text chunks for LLM processing."""
    print("🔪 Splitting into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    all_chunks = []
    global_id = 1

    for doc_index, doc_info in enumerate(documents):
        # Break content into chunks with overlap
        chunks = text_splitter.create_documents([doc_info["content"]])
        for j, chunk in enumerate(chunks):
            chunk_id = f"{doc_index + 1}_{j + 1}"
            chunk.metadata = {
                "source": doc_info["source"],
                "chunk_id": chunk_id
            }
            all_chunks.append({
                "id": global_id,
                "chunk_id": chunk_id,
                "source": doc_info["source"],
                "page_content": chunk.page_content
            })
            global_id += 1

    print(f"✅ Total chunks: {len(all_chunks)}")

    # Save chunks to CSV for later use
    chunks_df = pd.DataFrame(all_chunks)
    chunks_df.to_csv("outputs/chunks.csv", index=False, encoding='utf-8-sig')
    print("💾 Chunks saved to 'chunks.csv'.")
