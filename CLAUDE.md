# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an educational project demonstrating different AI grounding strategies for user search and retrieval systems. It uses EPAM's DIAL API (Azure OpenAI compatible) with a mock user service.

## Setup Commands

```bash
# Start mock user service (required - generates 1000 test users)
docker-compose up -d

# Install dependencies
pip install -r requirements.txt

# Set DIAL API key (required for LLM calls)
export DIAL_API_KEY=your_key_here
```

## Running the Tasks

Each task is a standalone script demonstrating a different grounding approach:

```bash
# Task 1: No grounding (batch processing)
python -m task.t1.no_grounding

# Task 2.1: Vector-based grounding (FAISS similarity search)
python -m task.t2.Input_vector_based

# Task 2.2: API-based grounding (structured parameter extraction)
python -m task.t2.input_api_based

# Task 3: Input-output grounding (Chroma + output verification)
python -m task.t3.in_out_grounding
```

## Architecture

### Three Grounding Approaches

1. **No Grounding (t1/)**: Loads all users into LLM context via batches. High token usage but simple implementation. Uses async parallel batch processing.

2. **Input Grounding (t2/)**:
   - **Vector-based**: Creates FAISS vectorstore from user documents, performs similarity search with embeddings (`text-embedding-3-small-1`), then generates answer from retrieved context.
   - **API-based**: Uses LLM to extract search parameters (name/surname/email) with Pydantic structured output, then queries user service API directly.

3. **Input-Output Grounding (t3/)**: Combines vector search (Chroma) with output verification. Embeds only `id` + `about_me` fields to reduce costs. Extracts entity IDs from LLM output, then fetches full user data to verify existence and prevent hallucinations.

### Key Components

- `task/_constants.py`: DIAL API URL, API key, user service endpoint
- `task/user_client.py`: HTTP client for mock user service (get all, get by ID, search)
- Mock user service: `localhost:8041` with Swagger at `/docs`. Users are added/deleted every 5 minutes.

### LLM Integration Pattern

All tasks use `AzureChatOpenAI` from langchain-openai with model `gpt-4o`. Embeddings use `AzureOpenAIEmbeddings` with `text-embedding-3-small-1` (384 dimensions).

### Structured Output

API-based grounding uses `PydanticOutputParser` with LCEL chains for extracting typed search parameters from natural language queries.
