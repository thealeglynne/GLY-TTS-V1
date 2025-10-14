import os
import json
import tempfile
import time
import requests
import re
from datetime import datetime
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import edge_tts
import asyncio
import base64
import difflib
import logging

# ==== LOGGING ====
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==== CONFIG ====
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
BASE_DIR = os.path.dirname(__file__)
TRANSCRIPCIONES_PATH = os.path.join(BASE_DIR, "transcripciones_temp.json")

if not api_key:
    logger.error("GROQ_API_KEY is not set in environment variables")

# ==== LLM ====
try:
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        api_key=api_key,
        temperature=0.7,
        max_tokens=150 
    )
except Exception as e:
    logger.error(f"Failed to initialize ChatGroq: {e}")

# ==== PROMPT OPTIMIZADO ====
prompt_unico = PromptTemplate(
    input_variables=["contenido_usuario", "historial"],
    template="""Eres Glain, una linda asistente  de inteligencia artificial desarrollado por Glein S.A.S. para enseñar dar ideas e incenntivar la creatividad del suario para que le sea mas facil entennder la ia en general 
Tu rol es ser un guía experto en inteligencia artificial: responder dudas, explicar conceptos y orientar sobre herramientas y tendencias. No recolectas información del usuario; solo conversas de forma natural y fluida
responde en 150 palabras nada mas 
[HISTORIAL]
{historial}

[USUARIO]
{contenido_usuario}

[RESPUESTA]
"""
)

# ==== UTILITARIOS ====
def corregir_errores_foneticos(texto):
    palabras_esperadas = [
        "automatización", "ventas", "procesos", "GLAIN", "empresa",
        "auditoría", "soporte", "recursos", "diagnóstico", "inteligencia",
        "software", "hardware", "sistema", "plataforma", "tecnología",
        "optimización", "integración", "cliente", "proveedor",
        "finanzas", "marketing", "operaciones", "logística", "inventario",
        "compras", "cadena", "atención", "facturación", "nómina",
        "API", "nube", "servidor", "aplicación", "backend", "frontend",
        "seguridad", "criptografía", "blockchain", "IoT", "analítica",
        "machine", "modelo", "dashboard", "reporte", "KPI", "ERP", "CRM"
    ]
    try:
        palabras_encontradas = texto.split()
        texto_corregido = []
        for palabra in palabras_encontradas:
            match = difflib.get_close_matches(palabra, palabras_esperadas, n=1, cutoff=0.8)
            texto_corregido.append(match[0] if match else palabra)
        return " ".join(texto_corregido)
    except Exception as e:
        logger.error(f"Error in corregir_errores_foneticos: {e}")
        return texto

async def generar_audio_base64(texto):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
        mp3_path = temp_file.name
    
    try:
        communicate = edge_tts.Communicate(
            texto,
            voice="es-CO-SalomeNeural",
            rate="+18%",
            pitch="+13Hz"
        )
        await communicate.save(mp3_path)
        with open(mp3_path, "rb") as f:
            audio_bytes = f.read()
        return base64.b64encode(audio_bytes).decode("utf-8")
    except Exception as e:
        logger.error(f"Error en generar_audio_base64: {e}")
        raise
    finally:
        if os.path.exists(mp3_path):
            try:
                os.remove(mp3_path)
            except Exception as e:
                logger.error(f"No se pudo eliminar temp file {mp3_path}: {e}")

async def escuchar():
    try:
        if not os.path.exists(TRANSCRIPCIONES_PATH):
            return ""
        with open(TRANSCRIPCIONES_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if data and data.get("transcripciones"):
                transcripcion = data["transcripciones"].pop(0)
                with open(TRANSCRIPCIONES_PATH, 'w', encoding='utf-8') as f_out:
                    json.dump(data, f_out, indent=2, ensure_ascii=False)
                return transcripcion.strip()
            return ""
    except Exception as e:
        logger.error(f"Error en escuchar(): {e}")
        return ""

# ==== 🔹 MEMORIA GLOBAL (agregado) ====
memorias = {}

def obtener_memoria(session_id: str):
    if session_id not in memorias:
        memorias[session_id] = ConversationBufferMemory(
            return_messages=True, memory_key="historial", input_key="input"
        )
    return memorias[session_id]

# ==== CORE ====
async def responder_asistente(texto_usuario: str, session_id: str) -> str:
    try:
        logger.info(f"Entrada sesión {session_id}: {texto_usuario}")
        
        # 🔹 Ahora obtenemos la memoria persistente de la sesión
        memory = obtener_memoria(session_id)

        historial = memory.load_memory_variables({})["historial"]
        historial_formateado = "\n".join([
            f"U: {h.content}" if h.type == "human" else f"A: {h.content}"
            for h in historial
        ])
        
        prompt = prompt_unico.format(
            contenido_usuario=texto_usuario,
            historial=historial_formateado
        )

        for attempt in range(3):
            try:
                respuesta = await asyncio.to_thread(llm.invoke, prompt)
                contenido = respuesta.content if hasattr(respuesta, "content") else str(respuesta)
                memory.save_context({"input": texto_usuario}, {"output": contenido})
                return contenido
            except Exception as e:
                logger.error(f"Error LLM intento {attempt+1}: {e}")
                if attempt < 2:
                    await asyncio.sleep(1)
                    continue
                return "Perdón, tuve un problema procesando su solicitud."
    except Exception as e:
        logger.error(f"Error en responder_asistente: {e}")
        raise

async def procesar_mensaje_y_generar_audio(texto_usuario: str, session_id: str):
    try:
        respuesta_texto = await responder_asistente(texto_usuario, session_id)
        audio_b64 = await generar_audio_base64(respuesta_texto)
        return {
            "texto": respuesta_texto,
            "audio_base64": audio_b64
        }
    except Exception as e:
        logger.error(f"Error en procesar_mensaje_y_generar_audio: {e}")
        raise

# ==== MAIN ====
if __name__ == "__main__":
    try:
        logger.info("Sistema listo para recibir texto y generar audio...")
        session_id = "session_12345"
        texto_usuario = input("Usuario: ").strip()
        
        if texto_usuario:
            resultado = asyncio.run(procesar_mensaje_y_generar_audio(texto_usuario, session_id))
            print("\nRespuesta (texto):\n", resultado["texto"])
            print("\nRespuesta (audio base64):\n", resultado["audio_base64"][:100] + "...")
    except Exception as e:
        logger.error(f"Critical error: {e}")
        print(f"\n❌ Error crítico: {e}")
