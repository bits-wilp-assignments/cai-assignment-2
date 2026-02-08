import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, TextStreamer
from langchain_huggingface import HuggingFacePipeline
from src.util.logging_util import get_logger


class LLMFactory:
    """Singleton factory that manages HuggingFace LLM instances."""

    _instance = None
    _llms = {}  # Dictionary to store LLM instances by configuration

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LLMFactory, cls).__new__(cls)
            cls._instance.logger = get_logger(__name__)
            cls._instance.logger.info("LLMFactory singleton created.")
        return cls._instance

    def get_instance(
        self,
        model_name: str,
        task: str = "text-generation",
        local_files_only: bool = True,
        max_new_tokens: int = 500,
        temperature: float = 0.001,
        repetition_penalty=1.0,
        top_p: float = 0.95,
        torch_dtype: str = "auto",
        device_map: str = "auto",
        do_sample: bool = True,
        is_streaming: bool = False,
        **kwargs,
    ) -> HuggingFacePipeline:
        self.logger.debug(f"All LLM parameters: model_name={model_name}, task={task}, max_new_tokens={max_new_tokens}, temperature={temperature}, repetition_penalty={repetition_penalty}, top_p={top_p}, torch_dtype={torch_dtype}, device_map={device_map}, do_sample={do_sample}, is_streaming={is_streaming}, kwargs={kwargs}")
        # Create a unique key for this configuration
        config_key = f"{model_name}_{device_map}_{max_new_tokens}_{temperature}_{top_p}_{torch_dtype}"
        # Return existing instance if available
        if config_key in self._llms:
            self.logger.info(
                f"Returning existing LLM instance for: {model_name} on {device_map}"
            )
            return self._llms[config_key]

        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
            # Set pad token if not set
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Load model using AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch_dtype, device_map=device_map, local_files_only=local_files_only
            )

            # Initialize the streamer for text generation
            streamer = TextStreamer(
                tokenizer, skip_prompt=True, skip_special_tokens=True
            ) if is_streaming else None

            # Create pipeline
            pipe = pipeline(
                task=task,
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                top_p=top_p,
                do_sample=do_sample,
                streamer=streamer,
                **kwargs,
            )

            # Wrap in LangChain
            llm_instance = HuggingFacePipeline(pipeline=pipe)

            # Store and return
            self._llms[config_key] = llm_instance
            self.logger.info(f"Successfully loaded {model_name} on {model.hf_device_map}")
            return llm_instance

        except Exception as e:
            self.logger.error(f"Error loading model {model_name}: {e}")
            raise
