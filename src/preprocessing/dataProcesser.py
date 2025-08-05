import cv2
from pytube import YouTube
import tempfile
import os
import logging
from pathlib import Path
import uuid

def download_youtube_video(url):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Caminho absoluto para o cookies.txt
    cookies_path = Path(__file__).parent.parent.parent / "cookies.txt"
    cookies_str = str(cookies_path.resolve())

    try:
        import importlib.util
        if importlib.util.find_spec("yt_dlp"):
            logger.info("Tentando baixar com yt-dlp com cookies...")
            from yt_dlp import YoutubeDL

            # Cria um nome de arquivo único no diretório temporário
            temp_filename = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.mp4")

            ydl_opts = {
                'format': 'best[ext=mp4]/best',
                'outtmpl': temp_filename,
                'quiet': True,
                'cookiefile': cookies_str,
            }

            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                title = info.get('title', 'Video')

            if os.path.exists(temp_filename) and os.path.getsize(temp_filename) > 100 * 1024:
                logger.info(f"Vídeo baixado com sucesso via yt-dlp: {title}")
                return temp_filename, title
            else:
                logger.error("Arquivo de vídeo corrompido ou incompleto")

    except Exception as e:
        logger.error(f"Erro ao baixar com yt-dlp: {str(e)}")
    
    return None, None

def extrair_frames(filepath, f=5):
    """
    ATIVIDADE: Extrair frames de um vídeo
    
    Parâmetros:
    - filepath: Caminho do arquivo de vídeo
    - f: Intervalo entre frames (extrai 1 a cada f frames)
    
    Retorno:
    - frames: Lista de frames em formato RGB
    """
    if filepath:
        # TODO: Abrir vídeo com cv2.VideoCapture
        # TODO: Extrair frames em intervalos regulares
        # TODO: Converter frames de BGR para RGB
        
        pass

def pre_processar_frame(frame, tamanho=(640, 480), equalizar=True, remover_ruido=True, tracar_contorno = True):
    """
    ATIVIDADE: Pré-processar frames para análise
    
    Parâmetros:
    - frame: Frame original
    - tamanho: Tuple com dimensões de saída
    - equalizar: Aplicar equalização de histograma
    - remover_ruido: Aplicar filtro de redução de ruído
    - tracar_contorno: Aplicar detecção de bordas
    
    Retorno:
    - frame_processado: Frame processado em escala de cinza
    """
    
    if frame is None:
        return None

    # TODO: Redimensionar frame para tamanho padrão
    
    # TODO: Converter para escala de cinza

    # TODO: Aplicar contorno usando Canny Edge
    
    # TODO: Aplicar equalização de histograma
    
    # TODO: Aplicar filtro Gaussiano para redução de ruído
    
    pass