from langchain.agents import create_agent
from langchain.tools import tool
from langchain_deepseek import ChatDeepSeek
import re
import spacy


class ExerciseAgent:
    """
    AI-powered agent for creating 'fill-in-the-gap' exercises from audio input.

    The ExerciseAgent class integrates speech-to-text transcription, NLP-based vocabulary extraction, 
    and LLM-powered exercise generation. It is designed to assist teachers in producing language learning 
    exercises automatically, with vocabulary tailored to a specific proficiency level.

    Attributes:
        llm: Language model instance used for reasoning and text generation.
        whisper: Speech-to-text transcription model instance.
        nlp: spaCy NLP model for text processing and vocabulary extraction.
        tools: List of callable tools used by the agent for vocabulary selection and exercise creation.
        agent: High-level agent combining the LLM and tools with a system prompt.

    Args:
        whisper: An instance of a speech-to-text model (e.g., OpenAI Whisper) for audio transcription.
        level (str): Desired vocabulary level for the exercises (e.g., "B2", "C1").
        model_name (str, optional): Name of the LLM model to use. Defaults to "deepseek-reasoner".
        temperature (float, optional): Temperature parameter for the LLM. Defaults to 0.5.

    Methods:
        transcribe(file_path):
            Transcribes an audio file to text using the provided whisper model.
        
        text_splitter(text):
            Splits long text into manageable chunks or sentences for easier processing.
        
        vocabulary_extractor(text):
            Extracts key content words (nouns, verbs, adjectives, adverbs) from the text, ignoring stopwords and punctuation.
        
        intelligent_vocab_selector(full_vocabulary, text):
            LLM-powered tool for selecting the most valuable vocabulary for exercises.
        
        exercise_creator(extracted_vocabulary, chuncked):
            LLM-powered tool for generating 'fill-in-the-gap' exercises based on selected vocabulary and text chunks.
        
        run(file_name):
            Main pipeline: transcribes an audio file, splits text, extracts vocabulary, selects words, 
            and generates a final exercise in a structured format.
    """
    def __init__(self, whisper, level, model_name="deepseek-reasoner", temperature=0.5):

        self.llm = ChatDeepSeek(model=model_name, temperature=temperature)
        self.whisper = whisper
        self.nlp = spacy.load("en_core_web_sm") 
        system_prompt = (
            "You are an AI agent that creates 'fill-in-the-gap' exercises for teachers.\n"
            "You will: select the vocabulary, then produce exercises.\n"
            "Gaps must be in the format [answer]. Keep sentences clear and grammatical."
            "You can use the tools provided."
            f"Level of the vocabulary must be: {level}"
            "Do NOT write any commentrary on the parts of speech or anything else. Put out put in the form:" \
            "1. Sentence." 
            "2. Sentence"
        )

        # Tools init
        self.tools = [
            self.exercise_creator,
            self.intelligent_vocab_selector
        ]

        # Agent init
        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=system_prompt
        )

    def transcribe(self, file_path):
        result = self.whisper.transcribe(file_path)
        return result["text"]
    
    def text_splitter(self, text: str):
        """Split long text into shorter chunks."""
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        if not paragraphs or any(len(p) > 500 for p in paragraphs):
            sentences = re.split(r'[.!?]+', text)
            paragraphs = [s.strip() for s in sentences if len(s.strip()) > 10]
        return paragraphs
    
    def vocabulary_extractor(self, text: str):
        """Extract content words."""
        doc = self.nlp(text)
        vocabulary = []
        with open("stopwords.txt", "r") as f:
            stop_words = f.read().split("\n")
        for token in doc:
            if token.text.lower() in stop_words or token.is_punct or len(token.text)<=2:
                continue
            if token.pos_ in ["NOUN","VERB","ADJ","ADV"]:
                vocabulary.append(token.text)
        return vocabulary
    
    # ---TOOLS BELOW---

    @tool
    def intelligent_vocab_selector(full_vocabulary: list, text: str):
        """
        LLM-powered vocabulary selection. 
        Returns the selected words for the exercise with reasoning.
        """
        return {
            "available_words": full_vocabulary,
            "text_context": text,
            "selection_criteria": "Choose words that are most valuable for language learning exercises for С1 level. Pick ones that are suitable for daily speech or can appear in high literature."
        }

    @tool
    def exercise_creator(extracted_vocabulary: list, chuncked: list):
        """Create a fill-in-the-gap exercise using the given words."""
        return f"Create the tasks using {extracted_vocabulary} and {chuncked}. USE ONLY ONE OR TWO GAPS IN ONE SENTENCE. DO NOT PUT TWO MISSING GAPS ONE RIGHT AFTER ANOTHER. The missing gaps must not be vague and multioptional among all of the sentences in the task. Use the text to understand the context, but make the sentences for the task by yourself."

    # Overall run
    def run(self, file_name: str):
        """Transcribe and create task"""
        transcribed_text = self.transcribe(file_name)
        chuncked = self.text_splitter(transcribed_text)
        vocab = self.vocabulary_extractor(str(chuncked))
        input = f"""Original text: {chuncked[:20]}, Extracted vocabulary: {vocab}"""

        result = self.agent.invoke({"messages": [{"role": "user", "content": str(input)}]})
        
        last_message = result['messages'][-1]

        if hasattr(last_message, 'content') and last_message.content:
            final_answer = last_message.content
            return final_answer 
