import json
from dataclasses import dataclass, field
from typing import List

import chess
import chess.engine
import instructor
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()
app = FastAPI()

app.mount("/static", StaticFiles(directory="app/static"), name="static")
app.mount("/img", StaticFiles(directory="app/img"), name="img")

# GAME STATE
board = chess.Board()
engine = chess.engine.SimpleEngine.popen_uci("/opt/homebrew/bin/stockfish")
moves = []
chat_history = []

llm = instructor.from_openai(OpenAI())

connected_clients = set()


class ChatMessage(BaseModel):
    sender: str
    content: str


class NextMove(BaseModel):
    move: str = Field(
        ..., description="The best move to win the chess game. It should be in standard algebraic notation."
    )
    reasoning: str = Field(..., description="Reasoning explaining why the move is the best one.")
    shit_talk: str = Field(..., description="Shit talk about the player, the game, and tease the player.")


@dataclass
class ChessAgent:
    board_state: str
    legal_moves: str
    history: str
    feedback: str = None  # type: ignore
    next_move: NextMove = None  # type: ignore
    conversation_history: List[ChatMessage] = field(default_factory=list)

    def __post_init__(self):
        self.next_move = llm.chat.completions.create(
            model="gpt-4o-mini",
            response_model=NextMove,
            messages=[
                {"role": "system", "content": "You are a chess grand master with a penchant for trash talk."},
                {
                    "role": "system",
                    "content": f"Given the current state of the chess board: {self.board_state}, legal moves: {self.legal_moves}, history of moves so far: {self.history}, and feedback on the previous move generated: {self.feedback}, generate the next move. The next move should be in standard algebraic notation like e2e4, e7e5, c6d4 etc. Also, provide some witty trash talk to tease the player.",
                },
            ],
        )


class Move(BaseModel):
    from_square: str
    to_square: str


@app.get("/", response_class=FileResponse)
async def read_root():
    return FileResponse("app/static/index.html")


@app.post("/move")
async def make_move(move: Move):
    global board, moves, chat_history

    try:
        # Make the player's move
        player_move = chess.Move.from_uci(move.from_square + move.to_square)
        if player_move in board.legal_moves:
            board.push(player_move)
            moves.append(player_move.uci())

            # Generate AI move
            ai_agent = ChessAgent(
                board_state=str(board), legal_moves=str(board.legal_moves), history=str(moves), feedback=""
            )
            ai_move_str = ai_agent.next_move.move
            ai_move = board.parse_san(ai_move_str)

            # Make the AI's move
            board.push(ai_move)
            moves.append(ai_move.uci())

            # Add AI's move and shit talk to chat history
            chat_history.append(ChatMessage(sender="ai", content=ai_agent.next_move.shit_talk))

            # Broadcast chat update to all connected clients
            await broadcast_chat_update()

            return {
                "player_move": player_move.uci(),
                "ai_move": ai_move.uci(),
                "board_fen": board.fen(),
                "game_over": board.is_game_over(),
                "reasoning": ai_agent.next_move.reasoning,
                "chat_history": [msg.dict() for msg in chat_history],
            }
        else:
            return {"error": "Invalid move", "board_fen": board.fen()}
    except chess.InvalidMoveError:
        return {"error": "Invalid move", "board_fen": board.fen()}


@app.get("/reset")
async def reset_game():
    global board, moves, chat_history
    board = chess.Board()
    moves = []
    chat_history = []
    return {"message": "Game reset", "board_fen": board.fen()}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            if data["type"] == "chat":
                chat_history.append(ChatMessage(sender="user", content=data["message"]))
                await broadcast_chat_update()
    except WebSocketDisconnect:
        connected_clients.remove(websocket)


async def broadcast_chat_update():
    for client in connected_clients:
        await client.send_json({"type": "chat_update", "chat_history": [msg.dict() for msg in chat_history]})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
