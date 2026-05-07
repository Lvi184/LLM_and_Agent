# Hugging Face Agents Course - Code Reproduction

This repository contains my reproduction of the code from the [Hugging Face Agents Course](https://github.com/huggingface/agents-course).

## Course Overview

The Hugging Face Agents Course teaches how to build AI agents using various frameworks including:
- **LangGraph** - A framework for building stateful multi-agent applications
- **LlamaIndex** - A data framework for LLM applications
- **LangChain** - A framework for developing applications powered by language models

## Repository Structure

```
├── Agentic_RAG/                # Basic RAG implementation
├── Agentic_RAG_Langgraph/      # RAG with LangGraph
├── Agentic_RAG_Llamaindex/     # RAG with LlamaIndex
├── LangGraph/                  # LangGraph agent examples
├── Llamaindex/                 # LlamaIndex agent examples
└── ...
```

## Modules

### 1. Agentic_RAG
Basic Retrieval-Augmented Generation implementation with:
- Custom retriever
- Tool integration
- Dataset handling

**Requirements:**
```bash
pip install -r requirements.txt
```

### 2. Agentic_RAG_Langgraph
RAG implementation using LangGraph framework:
- State management
- Agent orchestration
- Tool calling

**Requirements:**
```bash
pip install -U langgraph langchain langchain-openai langchain-community python-dotenv rank-bm25 duckduckgo-search
```

### 3. Agentic_RAG_Llamaindex
RAG implementation using LlamaIndex framework:
- Document indexing
- Query engines
- Workflow management

**Requirements:**
```bash
pip install -U llama-index llama-index-llms-openai-like python-dotenv duckduckgo-search rank-bm25
```

### 4. LangGraph
LangGraph agent examples including:
- Mail sorting agents
- Multi-agent workflows

### 5. Llamaindex
LlamaIndex components and workflows:
- Tool usage
- Agent workflows
- RAG implementations

## Getting Started

1. Clone the repository:
```bash
git clone https://github.com/Lvi184/LLM_and_Agent.git
cd LLM_and_Agent/03-Hugging_Face_Agents_Course
```

2. Install dependencies for each module as specified above.

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

4. Run the examples:
```bash
# For LangGraph examples
python Agentic_RAG_Langgraph/run.py

# For LlamaIndex examples  
python Agentic_RAG_Llamaindex/run.py
```

## Original Course

This code is based on the official [Hugging Face Agents Course](https://github.com/huggingface/agents-course).

## License

MIT License
