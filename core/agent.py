from langchain.agents import Tool, initialize_agent, AgentType
from langchain.memory import ConversationBufferWindowMemory
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage
from typing import Dict, List
import openai
from config.settings import settings
from .memory import MemoryManager
from .audio import AudioProcessor
from .security import SafeToolExecutor

class VoiceAgent:
    def __init__(self):
        self.audio_processor = AudioProcessor()
        self.memory_manager = MemoryManager()
        openai.api_key = settings.openai_api_key
        
        self.llm = ChatOpenAI(
            model_name=settings.model_name,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens
        )
        
        self.tools = [
            Tool(
                name="shell_command",
                func=SafeToolExecutor.execute_shell_command,
                description="Execute safe shell commands"
            ),
            Tool(
                name="open_application", 
                func=SafeToolExecutor.open_application,
                description="Open applications like VS Code, Chrome, etc."
            ),
            Tool(
                name="write_file",
                func=SafeToolExecutor.write_file,
                description="Write content to a file"
            )
        ]
        
        self.memory = ConversationBufferWindowMemory(
            k=settings.memory_window_size,
            return_messages=True
        )
        
        self.agent = initialize_agent(
            self.tools,
            self.llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True
        )
        
    async def process_voice_input(self, session_id: str) -> Dict:
        try:
            audio_data = await self.audio_processor.audio_queue.get()
            transcription = await self.audio_processor.transcribe_audio(audio_data)
            if not transcription:
                return {"error": "No speech detected"}
                
            return await self.process_text_input(transcription, session_id)
            
        except Exception as e:
            return {"error": str(e)}
    
    async def process_text_input(self, text: str, session_id: str) -> Dict:
        try:
            history = self.memory_manager.get_session_memory(session_id)
            
            voice_prompt = f"""
            You are a Voice AI Assistant. Respond concisely and naturally for speech output.
            User said: "{text}"
            
            Keep responses under {settings.max_tokens} characters and conversational.
            """
            
            result = self.agent.run(voice_prompt)
            audio_file = self.audio_processor.synthesize_speech(result)
            
            self.memory_manager.store_session_memory(session_id, text, result)
            
            return {
                "text": result,
                "audio_file": audio_file,
                "actions_taken": self._extract_actions_from_result(result)
            }
            
        except Exception as e:
            error_response = "I encountered an error processing your request."
            return {
                "text": error_response,
                "audio_file": self.audio_processor.synthesize_speech(error_response)
            }
    
    def _extract_actions_from_result(self, result: str) -> List[str]:
        actions = []
        if "opened" in result.lower():
            actions.append("application_opened")
        if "executed" in result.lower():
            actions.append("command_executed") 
        if "written" in result.lower():
            actions.append("file_written")
        return actions