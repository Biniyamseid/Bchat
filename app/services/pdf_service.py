from fastapi import UploadFile
from pypdf import PdfReader
from io import BytesIO

class PDFService:
    async def extract_text(self, file: UploadFile) -> str:
        content = await file.read()
        pdf = PdfReader(BytesIO(content))
        
        text = ""
        for page in pdf.pages:
            text += page.extract_text() + "\n"
        
        return text.strip()