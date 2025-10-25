# Инструкция по установке EmotionAnalysis с исправленным fasttext

## Проблема
Библиотека `fasttext` не компилируется на Windows из-за ошибки `ssize_t: необъявленный идентификатор`.

## Решение
Исправлен исходный код fasttext для совместимости с Windows.

## Установка

### Автоматическая установка
```bash
python install_fasttext_fixed.py
```

### Ручная установка
1. Скачать fasttext-0.9.2:
   ```bash
   pip download --no-deps fasttext==0.9.2
   ```

2. Распаковать архив:
   ```bash
   tar -xzf fasttext-0.9.2.tar.gz
   ```

3. Исправить исходный код:
   - Открыть файл: `fasttext-0.9.2/python/fasttext_module/fasttext/pybind/fasttext_pybind.cc`
   - Добавить в начало файла (после комментария с копирайтом):
   ```cpp
   // Fix for Windows compatibility - define ssize_t if not available
   #ifdef _WIN32
   #include <cstddef>
   #ifndef _SSIZE_T_DEFINED
   typedef long ssize_t;
   #define _SSIZE_T_DEFINED
   #endif
   #endif
   ```

4. Установить исправленный fasttext:
   ```bash
   pip install ./fasttext-0.9.2
   ```

5. Установить dostoevsky:
   ```bash
   pip install dostoevsky==0.6.0
   ```

## Проверка
```python
import fasttext
import dostoevsky
print("✅ Все библиотеки работают!")
```

## Примечания
- Исправление добавляет определение `ssize_t` для Windows
- Совместимо с Python 3.7+ и Windows
- Не влияет на функциональность библиотеки
