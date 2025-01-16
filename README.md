# SMS Server

A FastAPI-based SMS server that provides chat functionality using OpenAI's GPT models and DynamoDB for data persistence.

## 🚨 Important Notice for Developers

Before setting up the project, please read our [Developer Guide](DEVELOPER_GUIDE.md) to avoid common setup issues and learn best practices.

## Prerequisites

- Python 3.8+
- AWS Account with DynamoDB access
- OpenAI API key
- Git

## Installation

1. Clone the repository

```bash
git clone https://github.com/Cr-ai-mer/sms-server.git
cd sms-server
```

2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

## Configuration

1. Create a `.env` file in the root directory with the following variables:

```env
AWS_REGION=us-east-2
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
OPENAI_API_KEY=your_openai_api_key

TEXTGRID_ACCOUNT_SID=
TEXTGRID_AUTH_TOKEN=
TEXTGRID_PHONE_NUMBER_SID=
WEBHOOK_URL=https://ip-address/api/v1/sms/receive-sms

```

2. Make sure your AWS credentials have appropriate permissions to create and manage DynamoDB tables.

## Running the Application

1. Start the FastAPI server:

```bash
uvicorn app.main:app --reload
```

The server will start at `http://localhost:8000`

## API Documentation

Once the server is running, you can access:

- Interactive API documentation: `http://localhost:8000/docs`
- Alternative API documentation: `http://localhost:8000/redoc`

## Project Structure

```
app/
├── api/
│   ├── api_v1/
│   │   └── api.py
│   └── endpoints/
│       ├── chat.py
│       ├── script.py
│       └── ...
├── core/
│   └── config.py
├── models/
│   ├── db.py
│   └── dynamodb.py
├── schemas/
│   ├── chat.py
│   └── ...
├── services/
│   ├── chat_service.py
│   └── ...
└── main.py
```

## Features

- Real-time chat functionality using OpenAI's GPT models
- PDF script upload and processing
- DynamoDB integration for persistent storage
- Campaign management
- Lead information handling
- Chatbot script management

## API Endpoints

- `/api/v1/chat`: Chat-related endpoints
- `/api/v1/script`: Script management endpoints
- `/api/v1/knowledge`: Knowledge base endpoints
- `/api/v1/customer`: Customer management endpoints
- `/api/v1/sms`: SMS chat endpoints

## Development

To contribute to the project:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

## Support

## 📞 Support

If you encounter any issues not covered in the [Developer Guide](DEVELOPER_GUIDE.md), please:

1. Check existing GitHub issues
2. Create a new issue with detailed information
3. Reference relevant sections of the Developer Guide

For urgent matters, contact: [<the developers>](mailto:je.biniyam@gmail.com)

[Biniyamseid3@gmail.com] [je.biniyam@gmail.com]
