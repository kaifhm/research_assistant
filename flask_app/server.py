from flask import (
    Flask,
    render_template,
    url_for,
    request,
    redirect,
    Response,
    session,
    stream_with_context,
)
from flask_app.db import db, Base, Conversation, Message
from sqlalchemy import create_engine

from rag_agent import RAGAgent, TOOLS

from dotenv import load_dotenv
import os

load_dotenv()

engine = create_engine(os.environ['SQLALCHEMY_DATABASE_URI'], echo=True)
Base.metadata.create_all(engine)

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ['FLASK_SECRET_KEY']
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ['SQLALCHEMY_DATABASE_URI']

agent = RAGAgent(TOOLS)
SYSTEM_PROMPT = os.getenv('SYSTEM_PROMPT', 'You are an assistant. Talk to and help the user')

db.init_app(app)

CURRENT_CONVO_ID = 'current_convo_id'

app.jinja_env.filters['enumerate'] = enumerate

@app.get('/')
def home():
    chats = db.session.execute(db.select(Conversation)).scalars().fetchall()
    return render_template("index.html", chats=chats, title='ChatOllama')

@app.post('/')
def home_post():
    user_message = request.form['user_message']
    chats = db.session.execute(db.select(Conversation)).scalars()
    convo = Conversation()
    message = Message(content=user_message, role="human")
    system_msg = Message(content=SYSTEM_PROMPT, role="system")
    convo.messages.extend([system_msg, message])

    title: str = "".join(agent.ask(
        [ # type: ignore
            {'role': 'system', 'content': (f'<user_message> {user_message} </user_message>. '
                                           'This is the first message in a new conversation. '
                                           'Generate a title for this conversation. Do not use '
                                           'any tools. Keep the title short and simple. '
                                           'Do not say Title')},
        ]
    ))
    convo.title = title.removeprefix('Title:').strip()

    db.session.add(convo)
    db.session.commit()
    return redirect(url_for("chat_get", convo_id=convo.id),
                    Response=Response(render_template("chat.html", title=title, # type: ignore
                                                      chats=chats,
                                                      messages=[user_message])))

@app.get('/chat/<string:convo_id>/')
def chat_get(convo_id):
    chats = db.session.execute(db.select(Conversation)).scalars()
    convo = db.get_or_404(Conversation, convo_id, description=f"No conversation with id {convo_id}")
    session[CURRENT_CONVO_ID] = convo_id
    messages = db.session.execute(db.select(Message)
                .filter(Message.conversation_id == convo_id, Message.role.in_(['human', 'user', 'ai', 'assistant'])) \
                    .order_by(Message.created_at.asc())).scalars()
    return render_template("chat.html", messages=messages, chats=chats, convo_id=convo_id, title=convo.title)


@stream_with_context # type: ignore
def stream_ai_message(messages: list[dict], convo_id: str, last_message: Message):

    captured_data = []

    try:
        for message_chunk in agent.ask(messages): # pyright: ignore[reportArgumentType]
            captured_data.append(message_chunk)
            yield message_chunk
    except (BrokenPipeError, IOError, GeneratorExit):
        db.session.delete(last_message)
        db.session.commit()
        # log message delete
        return ""

    full_message = "".join(captured_data)
    ai_message = Message(content=full_message, role="ai", conversation_id=convo_id) # type: ignore
    db.session.add(ai_message)
    db.session.commit()


def format_flask_messages(messages: list[Message]) -> list[dict]:
    return [{"content": msg.content, "role": msg.role} for msg in messages]

@app.post('/chat/<string:convo_id>/')
def chat_post(convo_id):
    user_message = request.form['user_message']
    message = Message(content=user_message, role="human", conversation_id=convo_id)
    db.session.add(message)
    db.session.commit()
    chat_messages = db.session.execute(db.select(Message).filter(Message.conversation_id == convo_id)).scalars().all()
    return Response(stream_ai_message(format_flask_messages(chat_messages), # pyright: ignore[reportArgumentType]
                                      convo_id=convo_id, last_message=message))

@app.get('/chat/<string:convo_id>/delete')
def chat_delete(convo_id):
    # return to current chat if deleted chat is not current chat else return to home
    convo = db.get_or_404(Conversation, convo_id, description=f"No conversation with id {convo_id}")
    for message in convo.messages:
        db.session.delete(message)
    db.session.delete(convo)
    db.session.commit()
    if str(session.get(CURRENT_CONVO_ID)) == convo_id:
        session[CURRENT_CONVO_ID] = None
        return redirect(url_for('home'))
    return redirect(url_for('chat_get', convo_id=convo_id))

@app.post('/chat/<string:convo_id>/rename')
def chat_rename(convo_id):
    convo = db.get_or_404(Conversation, convo_id, description=f"No conversation with id {convo_id}")
    new_chat_name = request.form['chat_name']
    convo.title = new_chat_name
    db.session.commit()
    current_convo_id = str(session.get(CURRENT_CONVO_ID))
    return redirect(url_for('chat_get', convo_id=current_convo_id))
