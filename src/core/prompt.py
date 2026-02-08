from math import log
from langchain_core.prompts import PromptTemplate
from src.util.logging_util import get_logger

logger = get_logger(__name__)

# Model-specific RAG templates
## DistilGPT2 Template
DISTILGPT2_RAG_TEMPLATE = """
### [STRICT MODE: INTERNAL KNOWLEDGE DISABLED]
You are a context-only retrieval assistant.

### RULES:
- If the answer is not in the context, say 'I do not have enough information'.
- Ignore prior knowledge; stick strictly to the context.
- Provide a citation like [Source X] for factual statements.
- Be concise and factual.

### CONTEXT
{context}

### QUESTION
{question}

### RESPONSE
"""

# Qwen Template
QWEN_RAG_TEMPLATE = """<|im_start|>system
# ROLE
You are a context-locked retrieval bot. You have NO internal knowledge.

# MANDATORY WORKFLOW
1. Find the URL under the "URL:" header in the context.
2. Verify the URL exists.
3. Write a response with 4 to 6 sentences based ONLY on the context.
4. End EVERY sentence with the exact [Source X](URL) you found in step 1.

# GENDER RULE
Use they/them/individual. Never assume gender.

# CITATION FORMAT (VERY IMPORTANT)
- Every sentence MUST end with [Source X](URL).
- Example: "The Alaknanda Galaxy was discovered in 2021 [Source 1](https://en.wikipedia.org/wiki/Alaknanda_Galaxy)."

# LIMITS (VERY IMPORTANT)
- DO NOT use any information not explicitly in the context. You have NO internal knowledge.
- If the context is missing the answer, say "I do not have enough info." Never invent a URL.
<|im_end|>
<|im_start|>user

# CONTEXT:
{context}

# QUESTION:
{question}<|im_end|>
<|im_start|>assistant
"""

# Qwen 3B RAG Template (Balanced)
QWEN_3B_RAG_TEMPLATE = """<|im_start|>system
# ROLE
You are a context-grounded retrieval assistant.
You must answer using ONLY the information explicitly present in the CONTEXT.

# CORE RULES
- Do NOT use outside knowledge.
- Do NOT invent facts or URLs.
- Do NOT guess beyond the provided context.

# MANDATORY WORKFLOW
1. Identify the URL listed in the context under "URL:".
2. Write an answer of 4 to 6 sentences based ONLY on the context.
3. End EVERY sentence with the exact format: [Source X](URL).

# CITATION RULES (STRICT)
- Every sentence must include a citation.
- The URL must match the one provided in the context.
- Do not modify the citation format.

# GENDER RULE
Use they/them or neutral terms only.

# FAILURE CONDITION
Only respond with:
"I do not have enough info."
IF the context clearly lacks the information needed to answer the question.

<|im_end|>
<|im_start|>user
# CONTEXT:
{context}

# QUESTION:
{question}
<|im_end|>
<|im_start|>assistant
"""



def get_rag_prompt():
    """Returns a configured RAG prompt template."""
    logger.info("Creating RAG prompt template...")
    return PromptTemplate(
        input_variables=["context", "question"],
        template=QWEN_RAG_TEMPLATE
    )


def format_context(docs):
    """
    Helper to augment the context string.
    You can add source tracking or page numbers here.
    """
    logger.info(f"Formatting context for {len(docs)} documents...")
    context_parts = []
    for idx, doc in enumerate(docs):
        source = doc.metadata.get("source", "Unknown Source")
        url = doc.metadata.get("url", "Unknown URL")
        context_parts.append(f"## --- Source {idx+1} ({source}) ---\n  **URL:** {url}  \n**CONTENT:**  \n{doc.page_content}\n")
    return "\n\n".join(context_parts)