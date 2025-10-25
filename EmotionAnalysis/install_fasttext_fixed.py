#!/usr/bin/env python3
"""
Скрипт для установки исправленного fasttext на Windows
Исправляет проблему с ssize_t: необъявленный идентификатор
"""

import os
import sys
import subprocess
import tempfile
import shutil
from pathlib import Path

def install_fixed_fasttext():
    """Установка исправленного fasttext для Windows"""
    
    print("🔧 Установка исправленного fasttext для Windows...")
    
    # Создаем временную директорию
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Скачиваем fasttext
        print("📥 Скачивание fasttext-0.9.2...")
        subprocess.run([
            sys.executable, "-m", "pip", "download", "--no-deps", 
            "fasttext==0.9.2", "--dest", str(temp_path)
        ], check=True)
        
        # Распаковываем
        print("📦 Распаковка архива...")
        subprocess.run([
            "tar", "-xzf", str(temp_path / "fasttext-0.9.2.tar.gz"), 
            "-C", str(temp_path)
        ], check=True)
        
        # Исправляем исходный код
        print("🔧 Исправление исходного кода...")
        fix_file = temp_path / "fasttext-0.9.2" / "python" / "fasttext_module" / "fasttext" / "pybind" / "fasttext_pybind.cc"
        
        if fix_file.exists():
            # Читаем файл
            with open(fix_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Добавляем исправление для Windows
            fix_code = """// Fix for Windows compatibility - define ssize_t if not available
#ifdef _WIN32
#include <cstddef>
#ifndef _SSIZE_T_DEFINED
typedef long ssize_t;
#define _SSIZE_T_DEFINED
#endif
#endif

"""
            
            # Вставляем исправление после комментария
            if "Copyright (c) 2017-present, Facebook, Inc." in content:
                content = content.replace(
                    "Copyright (c) 2017-present, Facebook, Inc.",
                    "Copyright (c) 2017-present, Facebook, Inc." + "\n\n" + fix_code
                )
                
                # Записываем исправленный файл
                with open(fix_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                print("✅ Исходный код исправлен")
            else:
                print("⚠️ Не удалось найти место для вставки исправления")
        else:
            print("❌ Файл fasttext_pybind.cc не найден")
            return False
        
        # Устанавливаем исправленный fasttext
        print("📦 Установка исправленного fasttext...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", str(temp_path / "fasttext-0.9.2")
        ], check=True)
        
        print("✅ Исправленный fasttext успешно установлен!")
        return True

if __name__ == "__main__":
    try:
        success = install_fixed_fasttext()
        if success:
            print("🎉 Установка завершена успешно!")
        else:
            print("❌ Установка завершилась с ошибкой")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        sys.exit(1)
