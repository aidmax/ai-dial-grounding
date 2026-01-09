# Hobbies Searching Wizard - Flow Description

**A simple AI-powered application that searches user profiles in the User service by hobbies.**

**Example:**
- Input: `"I need people who love to go to mountains"`
- Output:
  ```json
  {
    "rock climbing": [{"full user info JSON"},...],
    "hiking": [{"full user info JSON"},...],
    "camping": [{"full user info JSON"},...]
  }
  ```

---

## Flow Architecture

### Phase 1: Load Context (Cold Start)

**Step 1.1: Get All Users**
- On application start, fetch all users from User Service
- Users batch processing (100 users per batch)

**Step 1.2: Create VectorDB**
- Embed only **User ID** and **about_me** fields (reduces context window)
- Store embedded batches in VectorDB (Chroma/PyVector)
- This creates the initial knowledge base

---

### Phase 2: Enhanced Input Vector Based Grounding

When a user submits a query, the retrieval process begins:

**Step R (Retrieval with Adaptive Sync):**

1. **Sync VectorDB with User Service**
   - Get all current users from UserService
   - Get all User IDs from Vector Store
   - Identify new users (added in last 5 min)
   - Identify deleted users (removed in last 5 min)
   - Update Vector store:
     - Delete old user embeddings
     - Add new user embeddings

2. **Convert Request to Embeddings**
   - User input → Embedding model → Vector representation

3. **Retrieve Relevant Context**
   - Perform similarity search in VectorDB
   - Retrieve top k (e.g., 50) most relevant user profiles
   - Return only `User ID` + `about_me` sections

---

### Phase 3: Augmented Prompt & Generation

**Step A (Augment):**
- Combine user query with retrieved context
- Format: System prompt + Retrieved user profiles + User query

**Step G (Generate):**
- Send augmented prompt to LLM (gpt-4o)
- LLM performs **Named Entity Extraction (NEE)**
- Output format (structured):
  ```json
  {
    "hobby": [user_id1, user_id2, user_id3, ...]
  }
  ```
- Only user IDs are returned (no PII in generation phase)

---

### Phase 4: Output API Based Grounding

**Purpose:** Verify user IDs exist and fetch full user data (prevents hallucinations)

**For each user_id in LLM response:**

1. **Verify & Fetch**
   - Call `GET user/{id}` on User Service
   - If user exists → Retrieve full JSON data with all fields:
     - `name`, `email`, `surname`, `bio`, etc.
   - If user doesn't exist → Filter out (hallucination detected)

2. **Group Results**
   - Organize verified users by hobby
   - Return final JSON response:
     ```json
     {
       "hobby1": [full_user_json1, full_user_json2, ...],
       "hobby2": [full_user_json3, ...],
       ...
     }
     ```

**Step O (Output Grounding Complete):**
- Return response to user with verified, complete user profiles

---

## Key Benefits

| Feature | Benefit |
|---------|---------|
| **Minimal Embeddings** | Only `id` + `about_me` embedded → Reduced costs |
| **Adaptive VectorDB** | Syncs every query → Handles dynamic user changes |
| **Structured Output** | LLM returns only IDs → Faster generation, less tokens |
| **Output Grounding** | Verifies IDs + fetches full data → No hallucinations |
| **API Integration** | User Service provides source of truth → Consistency |

---

## Decision Points (Diamond Nodes)

1. **Input Validation** - Is user query valid?
2. **Embedding Conversion** - Convert query to vector
3. **Context Retrieval** - Are there relevant users?
4. **ID Verification** - Does user ID exist in service?

---

## Data Flow Summary

```
User Query
  → VectorDB Sync (add/delete users)
  → Similarity Search (retrieve relevant profiles)
  → Augmented Prompt (query + context)
  → LLM Generation (extract hobby → [user_ids])
  → Output Grounding (verify IDs + fetch full data)
  → Final Response (hobby → [full user objects])
```
