#!/usr/bin/env python3
"""
AIRI MemorySystem Service - Launcher
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
    print("[INFO] Запуск AIRI MemorySystem Service...")
    
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
    
    # Запускаем MemorySystem сервис
    print("[INFO] Запуск: MemorySystem Service")
    memory_cmd = [
        python_executable,
        "-m", "uvicorn",
        "src.api.memory_api:app",
        "--host", "0.0.0.0",
        "--port", "8005",
        "--reload"
    ]
    
    print(f"[INFO] Команда: {' '.join(memory_cmd)}")
    
    try:
        memory_process = subprocess.Popen(
            memory_cmd,
            cwd=str(service_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        print("[OK] MemorySystem процесс запущен")
        
        # Показываем вывод процесса в реальном времени
        import threading
        def show_memory_output():
            for line in iter(memory_process.stdout.readline, ''):
                if line:
                    print(f"[memory] {line.strip()}")
        
        memory_thread = threading.Thread(target=show_memory_output, daemon=True)
        memory_thread.start()
    except Exception as e:
        print(f"[ERROR] Ошибка запуска MemorySystem: {e}")
        return 1
    
    print("=" * 50)
    print("[INFO] Сервис запущен:")
    print("[INFO]    MemorySystem: http://localhost:8005")
    print("=" * 50)
    print("[INFO] Сервис готов к работе!")
    
    try:
        # Ждем завершения процесса
        while True:
            time.sleep(1)
            
            # Проверяем статус процесса
            if memory_process.poll() is not None:
                print("[INFO] MemorySystem процесс завершился")
                break
                
    except KeyboardInterrupt:
        print("\n[INFO] Остановка сервиса...")
        
        # Останавливаем процесс
        if memory_process:
            memory_process.terminate()
            memory_process.wait()
        
        print("[INFO] Сервис остановлен")
        return 0
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
