# AI Chess + talk

This is not polished at all, did as quick thing but turned out pretty cool

AI Chess is a cool and interactive web-based chess game that combines the classic game of chess with cutting-edge AI technology. Play against an AI opponent that not only makes strategic moves but also engages in witty banter!

![image](https://github.com/user-attachments/assets/28bb1595-2c2a-47b0-80e9-fd745bd723d4)
![image](https://github.com/user-attachments/assets/567665c6-bcd6-405b-96d0-97dd0453e0cf)

## Features

- Interactive chessboard interface
- Real-time AI opponent powered by GPT-4
- Chat functionality for player-AI interaction
- Dynamic move analysis and reasoning
- Entertaining "trash talk" from the AI

## How It Works

1. The frontend is built using HTML, CSS, and JavaScript, featuring a responsive chessboard and chat interface. - claude helped, based
2. The backend is powered by FastAPI, did some websocket stuff for the chat, http for the moves, I don't care, just wanted to test this idea out
3. Chess moves are processed using the `python-chess` library, stockfish, downloaded with brew to use
4. The AI opponent makes use OpenAI's GPT-4o-mini model through the `instructor` library, which makes use of stockfish options, adds some commentary, and some shit talk for fun.

## Cool Factor

What makes this project stand out, OMG!!:

- **AI-Powered Opponent**: Play against an AI that thinks and communicates like a chess grandmaster.
- **Dynamic Analysis**: Get real-time reasoning behind each AI move.
- **Entertaining Interactions**: Enjoy witty "trash talk" from the AI, adding a fun twist to the game.
- **Seamless Integration**: Combines classic chess gameplay with modern AI technology.

## Technologies Used

- Frontend: HTML, CSS, JavaScript, chessboard.js
- Backend: Python, FastAPI, Pydantic.
- AI: OpenAI's GPT-4o-mini, `instructor` library
- Chess Logic: `python-chess` library

## Getting Started

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt` `brew install stockfish` or whatever you use for package management, then change the path in the code plz
3. Set up your OpenAI API key in a `.env` file
4. Run the server: `uvicorn app.api:app --port 8000`
5. Open `http://localhost:8000` in your browser

Start playing and experience the future of chess with AI Chess Master!

OBS: If you did not realize, 90% of this readme was claude sonnet writing, just to have anything, but anyways, cool project, love to use instructor, love to leverate LLMs and AI's in general to develop solutions / product. Crazy cool to work with
