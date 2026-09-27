"""
WebSocket Server for Real-Time AI Interview Coaching.
Provides bidirectional, low-latency communication between candidate clients and LangGraph agents.
"""

import os
import json
import logging
from typing import Dict, Any, List, Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import uvicorn

from graph import arun_question, arun_evaluation, run_final_report

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ws_interview_server")

app = FastAPI(
    title="AI Interview Coach - Real-Time WebSocket API",
    description="WebSocket & Real-Time Streaming Server powered by LangGraph, LangSmith & Groq",
    version="1.0.0"
)

# Enable CORS for external frontends (React, mobile, Vue, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ConnectionManager:
    """Manages active WebSocket client connections."""

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"Client connected. Active connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f"Client disconnected. Active connections: {len(self.active_connections)}")

    async def send_json(self, websocket: WebSocket, data: Dict[str, Any]):
        await websocket.send_json(data)


manager = ConnectionManager()


@app.get("/")
async def root():
    return {
        "service": "AI Interview Coach WebSocket Service",
        "status": "online",
        "websocket_endpoint": "/ws/interview",
        "langsmith_tracing": os.getenv("LANGCHAIN_TRACING_V2", "false") == "true"
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "active_ws_connections": len(manager.active_connections)
    }


@app.websocket("/ws/interview")
async def websocket_interview_endpoint(websocket: WebSocket):
    """
    Real-Time WebSocket endpoint for interactive mock interviews.

    Supported Client Payloads:
    -------------------------
    1. Ping:
       {"action": "ping"}

    2. Generate Question:
       {"action": "ask", "topic": "Projects", "job_description": "...", "question_number": 1, "history": []}

    3. Evaluate Answer:
       {"action": "evaluate", "question": "...", "answer": "...", "topic": "...", "job_description": "...", "history": []}

    4. Final Report:
       {"action": "final_report", "history": [...]}
    """
    await manager.connect(websocket)
    try:
        # Send initial connection handshake
        await manager.send_json(websocket, {
            "event": "connected",
            "message": "Connected to AI Interview Coach Real-Time WebSocket",
            "session_status": "ready"
        })

        while True:
            raw_data = await websocket.receive_text()
            try:
                payload = json.loads(raw_data)
            except json.JSONDecodeError:
                await manager.send_json(websocket, {
                    "event": "error",
                    "message": "Invalid JSON format."
                })
                continue

            action = payload.get("action", "")

            # 1. Ping / Health check
            if action == "ping":
                await manager.send_json(websocket, {
                    "event": "pong",
                    "timestamp": payload.get("timestamp")
                })

            # 2. Real-Time Question Generation
            elif action == "ask":
                topic = payload.get("topic", "Projects")
                job_description = payload.get("job_description", "")
                question_number = payload.get("question_number", 1)
                history = payload.get("history", [])

                await manager.send_json(websocket, {
                    "event": "agent_thinking",
                    "agent": "question_agent",
                    "message": f"🤖 Question Agent is generating question #{question_number} on {topic}..."
                })

                try:
                    question = await arun_question(
                        topic=topic,
                        job_description=job_description,
                        question_number=question_number,
                        history=history
                    )

                    await manager.send_json(websocket, {
                        "event": "question_generated",
                        "question_number": question_number,
                        "topic": topic,
                        "question": question
                    })
                except Exception as e:
                    logger.error(f"Error in ws question generation: {e}")
                    await manager.send_json(websocket, {
                        "event": "error",
                        "agent": "question_agent",
                        "message": str(e)
                    })

            # 3. Real-Time Answer Evaluation
            elif action == "evaluate":
                question = payload.get("question", "")
                answer = payload.get("answer", "")
                topic = payload.get("topic", "Projects")
                job_description = payload.get("job_description", "")
                history = payload.get("history", [])

                if not answer.strip():
                    await manager.send_json(websocket, {
                        "event": "error",
                        "message": "Answer cannot be empty."
                    })
                    continue

                await manager.send_json(websocket, {
                    "event": "agent_thinking",
                    "agent": "evaluation_agent",
                    "message": "🤖 Evaluation & Report Agents are analyzing your answer..."
                })

                try:
                    result = await arun_evaluation(
                        question=question,
                        answer=answer,
                        topic=topic,
                        job_description=job_description,
                        history=history
                    )

                    await manager.send_json(websocket, {
                        "event": "evaluation_completed",
                        "feedback": result["feedback"],
                        "report": result["report"]
                    })
                except Exception as e:
                    logger.error(f"Error in ws answer evaluation: {e}")
                    await manager.send_json(websocket, {
                        "event": "error",
                        "agent": "evaluation_agent",
                        "message": str(e)
                    })

            # 4. Final Comprehensive Report
            elif action == "final_report":
                history = payload.get("history", [])

                await manager.send_json(websocket, {
                    "event": "agent_thinking",
                    "agent": "report_agent",
                    "message": "🤖 Report Agent is generating comprehensive final interview assessment..."
                })

                try:
                    report = run_final_report(history)
                    await manager.send_json(websocket, {
                        "event": "final_report_completed",
                        "final_report": report
                    })
                except Exception as e:
                    logger.error(f"Error in ws final report: {e}")
                    await manager.send_json(websocket, {
                        "event": "error",
                        "agent": "report_agent",
                        "message": str(e)
                    })

            else:
                await manager.send_json(websocket, {
                    "event": "unknown_action",
                    "message": f"Action '{action}' is not supported. Use 'ask', 'evaluate', 'final_report', or 'ping'."
                })

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"Unexpected WebSocket error: {e}")
        manager.disconnect(websocket)


if __name__ == "__main__":
    host = os.getenv("WS_HOST", "0.0.0.0")
    port = int(os.getenv("WS_PORT", 8000))
    logger.info(f"Starting Real-Time Interview WebSocket Server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
