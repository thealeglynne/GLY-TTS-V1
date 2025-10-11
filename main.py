from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import asyncio
import base64
import tempfile
import os
import edge_tts
import time

# Importa la función 'responder_asistente' desde el módulo 'sst'
# ✅ Puede devolver solo la respuesta o una tupla: (respuesta, tokens_info)
from agent.chat import responder_asistente

app = FastAPI()

activo = True

origins = [
    "https://glynne-sst-ai-hsiy.vercel.app",  # tu frontend en Vercel
    "http://localhost:3000",                  # opcional para pruebas locales
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === NUEVA RUTA PARA EL ESTADO DEL SERVIDOR ===
@app.get("/")
async def get_root():
    """
    Endpoint para que el frontend pueda comprobar si el backend está activo.
    Responde con un estado 'ok' y un mensaje.
    """
    return {"status": "ok", "message": "Backend is live"}

# === FUNCIÓN PARA GUARDAR AUDIO TEMPORAL ===
async def hablar_async_to_file(texto, filepath):
    communicate = edge_tts.Communicate(
        texto,
        voice="es-CO-SalomeNeural",
        rate="+18%",
        pitch="+13Hz"
    )
    await communicate.save(filepath)

@app.post("/conversar")
async def conversar(request: Request):
    global activo
    if not activo:
        return JSONResponse(content={"error": "Servicio desactivado temporalmente"}, status_code=503)
    try:
        data = await request.json()
        texto_usuario = data.get("texto")
        print(f"Texto recibido: '{texto_usuario}'")
        if not texto_usuario or not isinstance(texto_usuario, str) or not texto_usuario.strip():
            return JSONResponse(content={"error": "No se recibió texto válido"}, status_code=400)
        
        session_id = "default_session"

        # ✅ Manejar ambos casos: que responda con tupla o solo con respuesta
        resultado = await responder_asistente(texto_usuario.strip(), session_id)
        if isinstance(resultado, tuple):
            respuesta, tokens_info = resultado
        else:
            respuesta = resultado
            tokens_info = {
                "usuario": len(texto_usuario.split()),
                "llm": len(respuesta.split()),
                "total": len(texto_usuario.split()) + len(respuesta.split())
            }
        
        temp_path = os.path.join(tempfile.gettempdir(), f"respuesta_{int(time.time())}.mp3")
        await hablar_async_to_file(respuesta, temp_path)
        with open(temp_path, "rb") as f:
            audio_base64 = base64.b64encode(f.read()).decode("utf-8")
        try:
            os.remove(temp_path)
        except Exception as e:
            print(f"Error al eliminar archivo temporal: {e}")
        
        return {
            "transcripcion_usuario": texto_usuario,
            "respuesta_asistente": respuesta,
            "audio_base64": audio_base64,
            "tokens": tokens_info  # 👈 siempre devolver tokens
        }
    except Exception as e:
        print(f"Error en /conversar: {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse(content={"error": f"Error interno: {str(e)}"}, status_code=500)
