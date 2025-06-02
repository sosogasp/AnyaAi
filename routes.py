from flask import Blueprint, render_template, request, session
from datetime import datetime
import socketio
import os
from gemini.Anya import responder_pergunta

bp = Blueprint("chat", __name__)

def registrar_log(origem, mensagem, chat_id):
    os.makedirs("logs", exist_ok=True)
    caminho = f"logs/chat_{chat_id}.log"
    mensagem = mensagem.strip()
    if mensagem:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(caminho, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] [{origem}] {mensagem}\n")
        html = f"[{timestamp}] [{origem}] {mensagem}"
        socketio.emit("nova_mensagem", {"html": html})

def carregar_historico():
    chat_id = session.get("chat_id")
    caminho = f"logs/chat_{chat_id}.log"
    linhas_coloridas = []
    if os.path.exists(caminho):
        with open(caminho, "r", encoding="utf-8") as f:
            linhas = list(f.readlines())
            for linha in linhas:
                if "[Yor Forger]" in linha:
                    cor = "red"
                elif "[Loid Forger]" in linha:
                    cor = "blue"
                elif "[Anya Forger]" in linha:
                    cor = "green"
                else:
                    cor = "black"
                linhas_coloridas.append(f'<font color="{cor}">{linha.strip()}</font>')
    return linhas_coloridas

@bp.route("/")
def home():
    return render_template("index.html", title="Página Inicial")

def processar_pergunta(nome_usuario, template_html):
    if "chat_id" not in session:
        session["chat_id"] = datetime.now().strftime("%Y%m%d-%H%M%S")
        registrar_log("SISTEMA", f"=== Início da Sessão {session['chat_id']} ===", session["chat_id"])

    if request.method == "POST":
        if "enviar" in request.form:
            msg = request.form["mensagem"]
            registrar_log(nome_usuario, msg, session["chat_id"])

            prompt = f"{nome_usuario} {msg}"

            if msg.strip().endswith("?"):
                resposta = responder_pergunta(prompt)
                registrar_log("[Anya Forger]", resposta, session["chat_id"])

        elif "encerrar" in request.form:
            registrar_log("SISTEMA", f"=== Fim da Sessão {session['chat_id']} ===", session["chat_id"])
            session.pop("chat_id", None)

    historico = carregar_historico()
    return render_template(template_html, historico=historico, title=f"Chat - {nome_usuario.strip('[]')}")

@bp.route("/Yor Forger", methods=["GET", "POST"])
def yor_forger():
    return processar_pergunta("[Yor Forger]", "Yor Forger.html")

@bp.route("/Loid Forger", methods=["GET", "POST"])
def loid_forger():
    return processar_pergunta("[Loid Forger]", "Loid Forger.html")
