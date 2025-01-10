from typing import List, Dict
from openai import AsyncOpenAI
from ..core.config import settings
from ..schemas.chat import ChatMessage

class OpenAIClient:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    async def get_response(
        self,
        message: str,
        lead_info: Dict,
        chat_history: List[ChatMessage]
    ) -> str:
        try:
            # Convert chat history to OpenAI format
            messages = [
                {"role": "system", "content": self._get_system_prompt(lead_info)}
            ]
            
            # Add chat history
            for msg in chat_history:
                messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
            
            # Add current message
            messages.append({
                "role": "user",
                "content": message
            })

            # Get completion from OpenAI
            response = await self.client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=messages,
                temperature=0.7,
                max_tokens=150
            )

            return response.choices[0].message.content

        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")

    def _get_system_prompt(self, lead_info: Dict) -> str:
        return f"""You are an AI assistant helping with real estate leads.
Current lead information:
- Name: {lead_info.get('name')}
- Email: {lead_info.get('email')}
- Details: {lead_info.get('detail')}

Please maintain a professional and helpful tone. If you need specific information,
ask for it politely. Your responses should be concise and relevant to real estate inquiries.""" 