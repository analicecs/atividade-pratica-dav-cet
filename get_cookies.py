#!/usr/bin/env python3
"""
YouTube Cookie Extractor para yt-dlp
"""

import logging
import subprocess
import sys
import os
from pathlib import Path
from typing import Optional


class YouTubeCookieExtractor:
    """Extrator de cookies do YouTube para yt-dlp."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _install_browser_cookie3(self) -> bool:
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "browser_cookie3"
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except Exception as e:
            self.logger.error(f"Erro ao instalar browser_cookie3: {e}")
            return False
    
    def _extract_cookies_from_browser(self, browser_func) -> Optional[str]:
        try:
            cookies = browser_func(domain_name='youtube.com')
            
            lines = [
                "# Netscape HTTP Cookie File",
                "# This is a generated file! Do not edit.",
                ""
            ]
            
            for cookie in cookies:
                domain = cookie.domain or 'youtube.com'
                domain_specified = 'TRUE' if domain.startswith('.') else 'FALSE'
                path = cookie.path or '/'
                secure = 'TRUE' if cookie.secure else 'FALSE'
                expires = str(int(cookie.expires)) if cookie.expires else '0'
                
                line = f"{domain}\t{domain_specified}\t{path}\t{secure}\t{expires}\t{cookie.name}\t{cookie.value}"
                lines.append(line)
            
            return '\n'.join(lines)
            
        except Exception as e:
            self.logger.warning(f"Erro ao extrair cookies: {e}")
            return None
    
    def extract_cookies(self) -> Optional[str]:
        try:
            import browser_cookie3
        except ImportError:
            if not self._install_browser_cookie3():
                return None
            import browser_cookie3
        
        # Tenta Chrome primeiro, depois Firefox
        for browser_name, browser_func in [('Chrome', browser_cookie3.chrome), 
                                         ('Firefox', browser_cookie3.firefox)]:
            self.logger.info(f"Extraindo cookies do {browser_name}...")
            cookies = self._extract_cookies_from_browser(browser_func)
            if cookies:
                return cookies
        
        return None
    
    def save_cookies(self, cookies_content: str, file_path: Path) -> bool:
        try:
            # Criar o diretório pai se não existir
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(cookies_content)
            self.logger.info(f"Cookies salvos em: {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"Erro ao salvar cookies: {e}")
            return False
    
    def extract_and_save(self, project_dir: Optional[str | Path] = None) -> bool:
        # Se não for especificado um diretório, usar o diretório atual
        if project_dir is None:
            project_dir = os.getcwd()
        
        cookies_path = Path(project_dir) / "cookies.txt"
        
        cookies_content = self.extract_cookies()
        if cookies_content:
            return self.save_cookies(cookies_content, cookies_path)
        
        # Template manual como fallback
        template = """# Netscape HTTP Cookie File
# INSTRUÇÕES: Substitua os valores pelos seus cookies do YouTube
.youtube.com	TRUE	/	TRUE	1735689600	VISITOR_INFO1_LIVE	SEU_VISITOR_INFO
.youtube.com	TRUE	/	TRUE	1735689600	YSC	SEU_YSC_VALUE
"""
        self.logger.info("Criando template manual...")
        return self.save_cookies(template, cookies_path)


def extract_youtube_cookies(project_dir: Optional[str | Path] = None, 
                          logger: Optional[logging.Logger] = None) -> bool:
    """Extrai cookies do YouTube para o diretório especificado."""
    extractor = YouTubeCookieExtractor(logger)
    return extractor.extract_and_save(project_dir)