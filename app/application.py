from flask import Flask,render_template,request,session,redirect,url_for
from app.components.retriever import retrieve_context

from dotenv import load_dotenv
import os
import pathlib

import os
# Robustly load .env from project root

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
## OpenAI only; HuggingFace deprecated

app = Flask(__name__)
app.secret_key = os.urandom(24)

from markupsafe import Markup
def nl2br(value):
    return Markup(value.replace("\n" , "<br>\n"))

app.jinja_env.filters['nl2br'] = nl2br

@app.route("/" , methods=["GET","POST"])
def index():
    if "messages" not in session:
        session["messages"]=[]

    if request.method=="POST":
        user_input = request.form.get("prompt")
        provider = "openai"  # Only OpenAI supported

        if user_input:
            messages = session["messages"]
            messages.append({"role" : "user" , "content":user_input})
            session["messages"] = messages

            # Aggregate chat history for context
            chat_history = ""
            for msg in messages:
                if msg["role"] == "user":
                    chat_history += f"User: {msg['content']}\n"
                elif msg["role"] == "assistant":
                    chat_history += f"Assistant: {msg['content']}\n"

            import traceback
            try:
                # Pass user_input and chat_history only, temperature fixed to 1
                result = retrieve_context.invoke({"query": user_input, "chat_history": chat_history})
                # Always use the cleaned string response
                if isinstance(result, tuple):
                    result = result[0]
                messages.append({"role" : "assistant" , "content" : result})
                session["messages"] = messages
            except Exception as e:
                tb = traceback.format_exc()
                error_type = type(e).__name__
                error_msg = f"Error: {error_type}: {str(e)}\nTraceback:\n{tb}"
                from app.common.logger import get_logger
                logger = get_logger(__name__)
                logger.error(f"Frontend error: {error_msg}")
                return render_template("index.html" , messages = session["messages"] , error = error_msg)
        return redirect(url_for("index"))
    return render_template("index.html" , messages=session.get("messages" , []))

@app.route("/clear")
def clear():
    session.pop("messages" , None)
    return redirect(url_for("index"))

if __name__=="__main__":
    app.run(host="0.0.0.0" , port=5000 , debug=False , use_reloader = False)




