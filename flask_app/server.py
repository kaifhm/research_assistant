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
SYSTEM_PROMPT = os.getenv(
    'SYSTEM_PROMPT', 'You are an assistant. Talk to and help the user')

db.init_app(app)

CURRENT_CONVO = 'current_convo'

app.jinja_env.filters['enumerate'] = enumerate

class OllamaModels:
    import urllib.request, json

    models = None

    @classmethod
    def refresh_models(cls):
        resp = cls.urllib.request.urlopen('http://localhost:11434/api/tags')
        cls.models = cls.json.loads(resp.read().decode())['models']

OllamaModels.refresh_models()

@app.get('/')
def home():
    chats = db.session.execute(db.select(Conversation).order_by(Conversation.created_at.desc())).scalars()
    return render_template("index.html", chats=chats, title='ChatLLM')


@app.post('/')
def home_post():
    user_message = request.form['user_message']
    chats = db.session.execute(db.select(Conversation).order_by(Conversation.created_at.desc())).scalars()
    convo = Conversation()
    message = Message(content=user_message, role="human")
    convo.messages.append(message)

    title: str = "".join(agent.ask(
        [  # type: ignore
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
                    Response=Response(render_template("chat.html", title=title,  # type: ignore
                                                      chats=chats,
                                                      messages=[user_message])))


@app.get('/chat/<string:convo_id>/')
def chat_get(convo_id):
    convo = db.get_or_404(Conversation, convo_id,
                          description=f"No conversation with id {convo_id}")
    chats = db.session.execute(db.select(Conversation).order_by(Conversation.created_at.desc())).scalars()
    session[CURRENT_CONVO] = convo
    messages = convo.messages
    return render_template("chat.html", messages=messages, chats=chats, convo_id=convo.id, title=convo.title)


@stream_with_context  # type: ignore
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
    ai_message = Message(content=full_message, role="ai",
                         conversation_id=convo_id)  # type: ignore
    db.session.add(ai_message)
    db.session.commit()


def format_flask_messages(messages: list[Message]) -> list[dict]:
    return [{"content": msg.content, "role": msg.role} for msg in messages]


@app.post('/chat/<string:convo_id>/')
def chat_post(convo_id):
    user_message = request.form['user_message']
    message = Message(content=user_message, role="human",
                      conversation_id=convo_id)
    db.session.add(message)
    db.session.commit()
    chat_messages = db.session.execute(db.select(Message).filter(
        Message.conversation_id == convo_id)).scalars()
    return Response(stream_ai_message(format_flask_messages(chat_messages),  # pyright: ignore[reportArgumentType]
                                      convo_id=convo_id, last_message=message))


@app.get('/upload')
def upload():
    # Cache chats and use here. Don't keep querying the same thing.
    chats = db.session.execute(db.select(Conversation).order_by(Conversation.created_at.desc())).scalars()
    return render_template("upload.html", chats=chats, title="ChatLLM | Upload files")


@app.post('/file-upload')
def file_upload():
    file = request.files['file']
    with open(str(file.filename), 'ab') as wfile:
        if wfile.tell() > 256 * 1048576:
            return "File too big", 403
        for chunk in file.stream:
            try:
                wfile.write(chunk)
            except (OSError, Exception) as e:
                print(e)
                return "An exception occured. Could not write file", 500

    return "Uploaded"


@app.get('/chat/<string:convo_id>/delete')
def chat_delete(convo_id):
    convo = db.get_or_404(Conversation, convo_id,
                          description=f"No conversation with id {convo_id}")
    db.session.delete(convo)
    db.session.commit()
    if session.get(CURRENT_CONVO)['id'] == str(convo.id):  # type: ignore
        session[CURRENT_CONVO] = None
        return redirect(url_for('home'))
    return redirect(request.referrer)


@app.post('/chat/<string:convo_id>/rename')
def chat_rename(convo_id):
    convo = db.get_or_404(Conversation, convo_id,
                          description=f"No conversation with id {convo_id}")
    new_chat_name = request.form['chat_name']
    convo.title = new_chat_name
    db.session.commit()
    return redirect(request.referrer)  # type: ignore


@app.get('/change-model/<string:model_name>')
def change_model(model_name):
    if model_name not in [model['name'] for model in OllamaModels.models]: # type: ignore
        return "No such model", 404
    agent.change_model(model_name)
    return f"Model changed to {model_name}"
