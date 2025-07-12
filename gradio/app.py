import gradio as gr
import os
import sys
import pandas as pd

# Ensure src is in sys.path for imports
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if src_path not in sys.path:
    sys.path.append(src_path)
from rag import (
    retrieve_similar_complaints,
    load_faiss_index,
    load_metadata,
    load_embedding_model,
    prepare_chunks_and_metadata,
    generate_llama_llm_answer
)

# Load models and data once at startup

df = pd.read_csv('data/chunked_complaints.csv')
embedding_model = load_embedding_model('all-MiniLM-L6-v2')
index_path = 'notebooks/vector_store/complaint_chunks.index'
metadata_path = 'notebooks/vector_store/complaint_chunks_metadata.pkl'
index = load_faiss_index(index_path)
metadata_list = load_metadata(metadata_path)
all_chunks, _ = prepare_chunks_and_metadata(df)

# Prompt template
PROMPT_TEMPLATE = (
    "You are a financial analyst assistant for CrediTrust. "
    "Your task is to answer questions about customer complaints. "
    "Use only the following retrieved complaint excerpts to formulate your answer. "
    "If the context does not contain the answer, state that you don't have enough information.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n"
    "Answer:"
)

def rag_answer(query, history):
    # Add user message
    history.append({"role": "user", "content": query})

    # Retrieve relevant chunks
    results = retrieve_similar_complaints(query, embedding_model, index, metadata_list, all_chunks, k=5)
    retrieved_chunks = [chunk for chunk, meta, dist in results]
    sources = retrieved_chunks[:3]

    answer = generate_llama_llm_answer(
        query,
        retrieved_chunks,
        PROMPT_TEMPLATE,
        model_name="meta-llama/Meta-Llama-3.1-8B-Instruct"
    )

    final_answer = answer + "\n\n**Sources:**\n" + "\n".join(f"- {s}" for s in sources)
    history.append({"role": "assistant", "content": final_answer})
    yield "", history

with gr.Blocks() as demo:
    gr.Markdown("## 💬 RAG Chat Interface with Streaming")

    chatbot = gr.Chatbot(type="messages")
    with gr.Row():
        txt = gr.Textbox(show_label=False, placeholder="Ask your question...", scale=8)
        submit_btn = gr.Button("Ask", scale=1)
        clear_btn = gr.Button("Clear", variant="stop", scale=1)

    # Enable streaming
    submit_btn.click(
        fn=rag_answer,
        inputs=[txt, chatbot],
        outputs=[txt, chatbot]
    )

    clear_btn.click(lambda: ("", []), outputs=[txt, chatbot])

demo.launch(share=True)

