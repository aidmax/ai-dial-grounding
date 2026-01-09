import asyncio
from typing import Any, Optional

from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import SystemMessagePromptTemplate, ChatPromptTemplate
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from pydantic import SecretStr, BaseModel, Field
from task._constants import DIAL_URL, API_KEY
from task.user_client import UserClient

#TODO: Info about app:
# HOBBIES SEARCHING WIZARD
# Searches users by hobbies and provides their full info in JSON format:
#   Input: `I need people who love to go to mountains`
#   Output:
#     ```json
#       "rock climbing": [{full user info JSON},...],
#       "hiking": [{full user info JSON},...],
#       "camping": [{full user info JSON},...]
#     ```
# ---
# 1. Since we are searching hobbies that persist in `about_me` section - we need to embed only user `id` and `about_me`!
#    It will allow us to reduce context window significantly.
# 2. Pay attention that every 5 minutes in User Service will be added new users and some will be deleted. We will at the
#    'cold start' add all users for current moment to vectorstor and with each user request we will update vectorstor on
#    the retrieval step, we will remove deleted users and add new - it will also resolve the issue with consistency
#    within this 2 services and will reduce costs (we don't need on each user request load vectorstor from scratch and pay for it).
# 3. We ask LLM make NEE (Named Entity Extraction) https://cloud.google.com/discover/what-is-entity-extraction?hl=en
#    and provide response in format:
#    {
#       "{hobby}": [{user_id}, 2, 4, 100...]
#    }
#    It allows us to save significant money on generation, reduce time on generation and eliminate possible
#    hallucinations (corrupted personal info or removed some parts of PII (Personal Identifiable Information)). After
#    generation we also need to make output grounding (fetch full info about user and in the same time check that all
#    presented IDs are correct).
# 4. In response we expect JSON with grouped users by their hobbies.
# ---
# This sample is based on the real solution where one Service provides our Wizard with user request, we fetch all
# required data and then returned back to 1st Service response in JSON format.
# ---
# Useful links:
# Chroma DB: https://docs.langchain.com/oss/python/integrations/vectorstores/index#chroma
# Document#id: https://docs.langchain.com/oss/python/langchain/knowledge-base#1-documents-and-document-loaders
# Chroma DB, async add documents: https://api.python.langchain.com/en/latest/vectorstores/langchain_chroma.vectorstores.Chroma.html#langchain_chroma.vectorstores.Chroma.aadd_documents
# Chroma DB, get all records: https://api.python.langchain.com/en/latest/vectorstores/langchain_chroma.vectorstores.Chroma.html#langchain_chroma.vectorstores.Chroma.get
# Chroma DB, delete records: https://api.python.langchain.com/en/latest/vectorstores/langchain_chroma.vectorstores.Chroma.html#langchain_chroma.vectorstores.Chroma.delete
# ---
# TASK:
# Implement such application as described on the `flow.png` with adaptive vector based grounding and 'lite' version of
# output grounding (verification that such user exist and fetch full user info)


# Pydantic models for structured LLM output
class HobbyUsers(BaseModel):
    """Mapping of a hobby to list of user IDs who have that hobby."""
    hobby: str = Field(description="The hobby name extracted from user about_me sections")
    user_ids: list[int] = Field(description="List of user IDs who have this hobby")


class HobbySearchResult(BaseModel):
    """Result of hobby search containing grouped users by hobby."""
    hobbies: list[HobbyUsers] = Field(
        default=[],
        description="List of hobbies with their associated user IDs"
    )


# System prompt for Named Entity Extraction
SYSTEM_PROMPT = """You are a hobby extraction assistant. Your task is to analyze user profiles and extract hobbies that match the user's search query.

## Instructions:
1. Analyze the user's question to understand what hobbies/interests they are looking for
2. Search through the provided user profiles (each contains an ID and about_me section)
3. Identify users whose about_me section mentions hobbies related to the search query
4. Group users by the specific hobby mentioned in their profile
5. Return ONLY user IDs - do not include any personal information

## Important:
- Only extract hobbies that are ACTUALLY mentioned in the about_me sections
- Group similar hobbies appropriately (e.g., "hiking", "mountain hiking" could be grouped)
- If no users match the query, return an empty list
- Be inclusive - if a hobby is related to the search query, include it

## Response Format:
{format_instructions}
"""

USER_PROMPT = """## USER PROFILES:
{context}

## SEARCH QUERY:
{query}"""


def format_user_for_embedding(user: dict[str, Any]) -> str:
    """Format user with only id and about_me for embedding."""
    return f"User ID: {user['id']}\nAbout me: {user.get('about_me', 'No information')}"


class HobbiesWizard:
    def __init__(self, embeddings: AzureOpenAIEmbeddings, llm_client: AzureChatOpenAI):
        self.embeddings = embeddings
        self.llm_client = llm_client
        self.vectorstore: Optional[Chroma] = None
        self.user_client = UserClient()
        self.parser = PydanticOutputParser(pydantic_object=HobbySearchResult)

    async def __aenter__(self):
        """Cold start: load all users and create vectorstore."""
        print("🚀 Cold start: Loading all users...")
        all_users = self.user_client.get_all_users()

        # Create documents with only id and about_me
        documents = []
        for user in all_users:
            doc = Document(
                page_content=format_user_for_embedding(user),
                metadata={"user_id": user["id"]},
                id=str(user["id"])
            )
            documents.append(doc)

        # Create Chroma vectorstore
        print(f"📦 Creating vectorstore with {len(documents)} users...")
        self.vectorstore = await Chroma.afrom_documents(
            documents=documents,
            embedding=self.embeddings,
            collection_name="users_hobbies"
        )
        print("✅ Vectorstore ready!")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup vectorstore on exit."""
        if self.vectorstore:
            # Delete the collection to clean up
            self.vectorstore.delete_collection()

    async def _sync_vectorstore(self):
        """Sync vectorstore with current user service data (handle additions/deletions)."""
        print("🔄 Syncing vectorstore with user service...")

        # 1. Get all current users from service
        current_users = self.user_client.get_all_users()
        current_user_ids = {str(user["id"]) for user in current_users}

        # 2. Get all user IDs from vectorstore
        stored_data = self.vectorstore.get()
        stored_ids = set(stored_data["ids"]) if stored_data["ids"] else set()

        # 3. Identify new and deleted users
        new_user_ids = current_user_ids - stored_ids
        deleted_user_ids = stored_ids - current_user_ids

        # 4. Delete removed users from vectorstore
        if deleted_user_ids:
            print(f"🗑️ Removing {len(deleted_user_ids)} deleted users...")
            self.vectorstore.delete(ids=list(deleted_user_ids))

        # 5. Add new users to vectorstore
        if new_user_ids:
            print(f"➕ Adding {len(new_user_ids)} new users...")
            new_users = [u for u in current_users if str(u["id"]) in new_user_ids]
            new_documents = [
                Document(
                    page_content=format_user_for_embedding(user),
                    metadata={"user_id": user["id"]},
                    id=str(user["id"])
                )
                for user in new_users
            ]
            await self.vectorstore.aadd_documents(new_documents)

        if not new_user_ids and not deleted_user_ids:
            print("✅ Vectorstore is up to date")
        else:
            print(f"✅ Sync complete: +{len(new_user_ids)} / -{len(deleted_user_ids)}")

    async def retrieve_context(self, query: str, k: int = 50) -> str:
        """Retrieve relevant user profiles based on query."""
        # First sync vectorstore
        await self._sync_vectorstore()

        # Perform similarity search
        print(f"🔍 Searching for relevant users (top {k})...")
        results = await self.vectorstore.asimilarity_search(query, k=k)

        print(f"📊 Found {len(results)} relevant users")

        # Join all retrieved documents as context
        context_parts = [doc.page_content for doc in results]
        return "\n\n".join(context_parts)

    def augment_prompt(self, query: str, context: str) -> str:
        """Create augmented prompt with context and query."""
        return USER_PROMPT.format(context=context, query=query)

    async def generate_hobby_ids(self, augmented_prompt: str) -> HobbySearchResult:
        """Generate structured output with hobby -> user IDs mapping."""
        print("🤖 Extracting hobbies and user IDs...")

        # Create prompt template with format instructions
        messages = [
            SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
            HumanMessage(content=augmented_prompt)
        ]

        prompt = ChatPromptTemplate.from_messages(messages).partial(
            format_instructions=self.parser.get_format_instructions()
        )

        # Use LCEL chain
        result: HobbySearchResult = await (prompt | self.llm_client | self.parser).ainvoke({})
        return result

    async def ground_output(self, hobby_result: HobbySearchResult) -> dict[str, list[dict[str, Any]]]:
        """Output grounding: fetch full user data and verify IDs exist."""
        print("🔒 Output grounding: fetching full user data...")

        grounded_result: dict[str, list[dict[str, Any]]] = {}

        for hobby_users in hobby_result.hobbies:
            hobby_name = hobby_users.hobby
            grounded_result[hobby_name] = []

            # Fetch full user data for each ID
            for user_id in hobby_users.user_ids:
                try:
                    user_data = await self.user_client.get_user(user_id)
                    grounded_result[hobby_name].append(user_data)
                except Exception as e:
                    print(f"⚠️ User ID {user_id} not found (hallucination filtered): {e}")

            print(f"  ✓ {hobby_name}: {len(grounded_result[hobby_name])} verified users")

        return grounded_result


async def main():
    # Initialize embeddings
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=DIAL_URL,
        api_key=SecretStr(API_KEY),
        model="text-embedding-3-small-1",
        dimensions=384
    )

    # Initialize LLM client
    llm_client = AzureChatOpenAI(
        azure_endpoint=DIAL_URL,
        api_key=SecretStr(API_KEY),
        api_version="",
        model="gpt-4o"
    )

    async with HobbiesWizard(embeddings, llm_client) as wizard:
        print("\n🧙 Hobbies Searching Wizard")
        print("Query samples:")
        print(" - I need people who love to go to mountains")
        print(" - Find users interested in photography")
        print(" - Who likes cooking and reading?")
        print("Type 'quit' or 'exit' to stop\n")

        while True:
            user_question = input("> ").strip()
            if user_question.lower() in ['quit', 'exit']:
                print("Goodbye! 👋")
                break

            if not user_question:
                continue

            print("\n--- Processing query ---")

            # 1. Retrieve context (with vectorstore sync)
            context = await wizard.retrieve_context(user_question)

            # 2. Augment prompt
            augmented_prompt = wizard.augment_prompt(user_question, context)

            # 3. Generate structured output (hobby -> user IDs)
            hobby_result = await wizard.generate_hobby_ids(augmented_prompt)

            # 4. Output grounding (fetch full user data, verify IDs)
            if hobby_result.hobbies:
                grounded_result = await wizard.ground_output(hobby_result)

                # 5. Print final JSON result
                import json
                print(f"\n=== RESULT ===")
                print(json.dumps(grounded_result, indent=2))
            else:
                print("\n❌ No users found matching your hobby search")

            print()


if __name__ == "__main__":
    asyncio.run(main())

