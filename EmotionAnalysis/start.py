#!/usr/bin/env python3
"""
AIRI EmotionAnalysis Service - Launcher
"""

import sys
import os
from pathlib import Path
import subprocess
import time

# Подавляем предупреждения AIRI сервисов
sys.path.append(str(Path(__file__).parent.parent / "ServiceManager"))
try:
    from suppress_warnings import suppress_all_warnings
    suppress_all_warnings()
except ImportError:
    pass

def main():
    """Основная функция запуска"""
    print("[INFO] Запуск AIRI EmotionAnalysis Services...")
    
    # Находим Python runtime в локальном venv
    current_dir = Path(__file__).resolve().parent
    venv_python = current_dir / "venv" / "Scripts" / "python.exe"
    
    if venv_python.exists():
        print(f"[OK] Найден Python runtime в venv: {venv_python}")
        python_executable = str(venv_python)
    else:
        print("[WARN] venv не найден, используем системный Python")
        python_executable = sys.executable
    
    # Путь к сервису
    service_dir = Path(__file__).resolve().parent
    print(f"[INFO] Рабочая директория: {service_dir}")
    print(f"[INFO] Python: {python_executable}")
    
    # Запускаем Aniemore сервис
    print("[INFO] Запуск Aniemore SER Service...")
    aniemore_cmd = [
        python_executable,
        "-m", "uvicorn",
        "aniemore_ser_service:app",
        "--host", "0.0.0.0",
        "--port", "8006",
        "--reload"
    ]
    
    print(f"[INFO] Aniemore команда: {' '.join(aniemore_cmd)}")
    
    try:
        aniemore_process = subprocess.Popen(
            aniemore_cmd,
            cwd=str(service_dir / "src"),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        print("[OK] Aniemore процесс запущен")
        
        # Показываем вывод процесса в реальном времени
        import threading
        def show_aniemore_output():
            for line in iter(aniemore_process.stdout.readline, ''):
                if line:
                    print(f"[aniemore] {line.strip()}")
        
        aniemore_thread = threading.Thread(target=show_aniemore_output, daemon=True)
        aniemore_thread.start()
        
    except Exception as e:
        print(f"[ERROR] Ошибка запуска Aniemore: {e}")
        aniemore_process = None
    
    time.sleep(2)
    
    # Запускаем Dostoevsky сервис
    print("[INFO] Запуск Dostoevsky Service...")
    dostoevsky_cmd = [
        python_executable,
        "-m", "uvicorn",
        "dostoevsky_service:app",
        "--host", "0.0.0.0",
        "--port", "8007",
        "--reload"
    ]
    
    print(f"[INFO] Dostoevsky команда: {' '.join(dostoevsky_cmd)}")
    
    try:
        dostoevsky_process = subprocess.Popen(
            dostoevsky_cmd,
            cwd=str(service_dir / "src"),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        print("[OK] Dostoevsky процесс запущен")
        
        # Показываем вывод процесса в реальном времени
        def show_dostoevsky_output():
            for line in iter(dostoevsky_process.stdout.readline, ''):
                if line:
                    print(f"[dostoevsky] {line.strip()}")
        
        dostoevsky_thread = threading.Thread(target=show_dostoevsky_output, daemon=True)
        dostoevsky_thread.start()
        
    except Exception as e:
        print(f"[ERROR] Ошибка запуска Dostoevsky: {e}")
        dostoevsky_process = None
    
    if not aniemore_process and not dostoevsky_process:
        print("[ERROR] Не удалось запустить ни один сервис!")
        return 1
    
    print("=" * 50)
    print("[INFO] Сервисы запущены:")
    if aniemore_process:
        print("[INFO]    Aniemore SER: http://localhost:8006")
    if dostoevsky_process:
        print("[INFO]    Dostoevsky: http://localhost:8007")
    print("=" * 50)
    print("[INFO] Сервисы готовы к работе!")
    
    try:
        # Ждем завершения процессов
        while True:
            time.sleep(1)
            
            # Проверяем статус процессов
            if aniemore_process and aniemore_process.poll() is not None:
                print("[INFO] Aniemore процесс завершился")
                aniemore_process = None
            
            if dostoevsky_process and dostoevsky_process.poll() is not None:
                print("[INFO] Dostoevsky процесс завершился")
                dostoevsky_process = None
            
            # Если все процессы завершились
            if not aniemore_process and not dostoevsky_process:
                print("[INFO] Все сервисы завершились")
                break
                
    except KeyboardInterrupt:
        print("\n[INFO] Остановка сервисов...")
        
        # Останавливаем процессы
        if aniemore_process:
            aniemore_process.terminate()
            aniemore_process.wait()
        
        if dostoevsky_process:
            dostoevsky_process.terminate()
            dostoevsky_process.wait()
        
        print("[INFO] Сервисы остановлены")
        return 0
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
