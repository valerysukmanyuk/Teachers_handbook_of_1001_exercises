from langchain.agents import create_agent
from langchain.tools import tool
from langchain_deepseek import ChatDeepSeek
import re
import spacy


class ExerciseAgent:
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
        input = f"""Original text: {chuncked}, Extracted vocabulary: {vocab}"""

        result = self.agent.invoke({"messages": [{"role": "user", "content": str(input)}]})
        
        last_message = result['messages'][-1]

        if hasattr(last_message, 'content') and last_message.content:
            final_answer = last_message.content
            return final_answer 
