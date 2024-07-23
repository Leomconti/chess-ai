import json
from dataclasses import dataclass
from typing import List

import chess
import chess.engine
import instructor
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket
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


llm = instructor.from_openai(OpenAI())


class NextMove(BaseModel):
    move: str = Field(
        ..., description="The best move to win the chess game. It should be in standard algebraic notation."
    )
    reasoning: str = Field(..., description="Reasoning explaining why the move is the best one.")
    shit_talk: str = Field(..., description="Shit talk about the player, the game, and tease the player.")

    # TODO: add a validator here that will check if the move is in the correct algebraic notation for chess,
    # if it's not instructor will send it back and retry


@dataclass
class ChessAgent:
    board_state: str
    legal_moves: str
    history: str
    feedback: str = None  # type: ignore
    # next move is the output field it will be null initially
    next_move: NextMove = None  # type: ignore
    conversation_history: List[str] = []

    # Post init is a function that runs after the dataclass has been initialized
    def __post_init__(self):
        # get the next move from the model
        self.next_move = llm.chat.completions.create(
            model="gpt-4o-mini",
            response_model=NextMove,
            messages=[
                {"role": "system", "content": "You are a chess grand master."},
                {
                    "role": "system",
                    "content": f"Given the current state of the chess board: {self.board_state}, legal moves: {self.legal_moves}, history of moves so far: {self.history}, and feedback on the previous move generated: {self.feedback}, generate the next move. The next move should be in standard algebraic notation like e2e4, e7e5, c6d4 etc.",
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
    global board, moves

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

        return {
            "player_move": player_move.uci(),
            "ai_move": ai_move.uci(),
            "board_fen": board.fen(),
            "game_over": board.is_game_over(),
            "reasoning": ai_agent.next_move.reasoning,
            "shit_talk": ai_agent.next_move.shit_talk,
        }
    else:
        return {"error": "Invalid move"}


@app.get("/reset")
async def reset_game():
    global board, moves
    board = chess.Board()
    moves = []
    return {"message": "Game reset", "board_fen": board.fen()}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        response = f"Received: {data}"
        await websocket.send_text(response)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
