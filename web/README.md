# UIT Chatbot Web

Minimal frontend to talk to the FastAPI `/chat` endpoint.

## Setup
```bash
cd web
npm install
```

## Run dev server
```bash
npm run dev
# defaults to http://localhost:5173
```

## Build
```bash
npm run build
```

## Config
- `VITE_API_BASE_URL` (default: `http://localhost:8000`)
  - Example: `VITE_API_BASE_URL=https://your-backend-url` npm run dev

