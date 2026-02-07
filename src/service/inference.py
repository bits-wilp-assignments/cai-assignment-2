from src.util.logging_util import get_logger
from src.config.app_config import LLM_MODEL_TASK, LLM_MODEL, LLM_CONFIG
from src.core.prompt import format_context, get_rag_prompt, format_context
from src.core.llm import LLMFactory
from .retrieval import retrieve_contexts

from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.output_parsers import StrOutputParser
from transformers.utils import logging

# Set transformers logging to error only
logging.set_verbosity_error()

# Get logger instance
logger = get_logger(__name__)

# Get RAG prompt template
rag_prompt = get_rag_prompt()

# Create LLM instance
llm_factory = LLMFactory()
llm_instance = llm_factory.get_instance(
    model_name=LLM_MODEL,
    task=LLM_MODEL_TASK,
    **LLM_CONFIG
)


def perform_rag_inference():
    # Setup the input data
    map_input = RunnableParallel(
        {
            "context": retrieve_contexts,
            "question": RunnablePassthrough(),
        }
    )

    # Build the chain
    rag_chain = map_input | RunnableParallel(
        {
            # Pass the formatted string to the prompt
            "answer": (
                RunnableLambda(
                    lambda x: {
                        "context": format_context(x["context"]),
                        "question": x["question"],
                    }
                )
                | rag_prompt
                | llm_instance
                | StrOutputParser()
            ),
            # Carry the raw context forward to the final output
            "sources": RunnableLambda(lambda x: x["context"]),
        }
    )
    return rag_chain

def rag_inference(question: str, is_streaming: bool = True):
    rag_chain = perform_rag_inference()
    if is_streaming:
        for chunk in rag_chain.stream({"question": question}):
            if "answer" in chunk:
                token = chunk["answer"]
                if token:
                    print(token, end="", flush=True)
        print()  # Print a newline after streaming is done
    else:
        response = rag_chain.invoke({"question": question})
        if "answer" in response:
            print(response["answer"])
        else:
            logger.error("RAG inference failed to produce an answer.")