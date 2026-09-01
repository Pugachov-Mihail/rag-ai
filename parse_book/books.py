import os
import pdfplumber
import fitz
from ollama import Client
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import re
from reportlab.lib.pagesizes import A4, letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from pathlib import Path
import hashlib
import json


import pytesseract
from PIL import Image
import pdfplumber

pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'

# Проверяем
if os.path.exists(pytesseract.pytesseract.tesseract_cmd):
    print("✓ Tesseract найден на Mac!")
else:
    print("❌ Tesseract не найден")



class TechnicalBookTranslator:
    def __init__(self, pdf_path, qdrant_url="http://localhost:6333",
                 ollama_host="http://localhost:11434", embedding_model="nomic-embed-text"):
        self.pdf_path = pdf_path
        self.ollama_client = Client(host=ollama_host)
        self.qdrant_client = QdrantClient(qdrant_url)
        self.embedding_model = embedding_model
        self.collection_name = "technical_book_context"
        self.extracted_images = {}
        self.code_blocks = {}

        # Инициализируем коллекцию в Qdrant
        self._init_qdrant_collection()

    def _init_qdrant_collection(self):
        """Инициализация коллекции в Qdrant"""
        try:
            self.qdrant_client.delete_collection(self.collection_name)
        except:
            pass

        self.qdrant_client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
        print(f"✓ Qdrant коллекция инициализирована")

    def _detect_code_block(self, text):
        """Детектирует блоки кода и возвращает индексы"""
        # Паттерны для распознавания кода
        code_patterns = [
            r'```[\s\S]*?```',  # Markdown code blocks
            r'(?:^|\n)([ ]{4,}|\t)(?!.*[.,:;!?]$).*(?:\n(?:[ ]{4,}|\t).*)*',  # Indented code
            r'<code>[\s\S]*?</code>',  # HTML code tags
            r'def\s+\w+\(|class\s+\w+|import\s+\w+|function\s+\w+',  # Code keywords
        ]

        code_regions = []
        for pattern in code_patterns:
            for match in re.finditer(pattern, text, re.MULTILINE):
                code_regions.append((match.start(), match.end()))

        return code_regions

    def _split_text_and_code(self, text):
        """Разделяет текст на части: код и обычный текст"""
        code_regions = self._detect_code_block(text)

        parts = []
        last_end = 0

        for start, end in sorted(code_regions):
            # Добавляем текст до кода
            if start > last_end:
                parts.append({
                    'type': 'text',
                    'content': text[last_end:start]
                })

            # Добавляем код
            parts.append({
                'type': 'code',
                'content': text[start:end]
            })

            last_end = end

        # Добавляем оставшийся текст
        if last_end < len(text):
            parts.append({
                'type': 'text',
                'content': text[last_end:]
            })

        return parts

    def _get_embedding(self, text):
        """Получает эмбеддинг текста из Ollama"""
        try:
            response = self.ollama_client.embeddings(
                model=self.embedding_model,
                prompt=text
            )
            return response['embedding']
        except Exception as e:
            print(f"⚠ Ошибка при получении эмбеддинга: {e}")
            return None

    def _translate_text(self, text, context=""):
        """Переводит текст через Ollama с контекстом"""
        prompt = f"""Ты профессиональный переводчик технических документов на русский язык.

ПРАВИЛА:
1. Переводи только текст, не трогай код, имена функций, переменных, классов
2. Переводи комментарии в коде на русский
3. Сохраняй все форматирование: списки, нумерацию, структуру
4. Технические термины переводи точно и консистентно
5. Сохраняй стиль оригинала

{f'КОНТЕКСТ ДОКУМЕНТА:{chr(10)}{context}' if context else ''}

ТЕКСТ ДЛЯ ПЕРЕВОДА:
{text}
[24.08.2026 00:50] Михаил Пугачев: ПЕРЕВЕДЁННЫЙ ТЕКСТ:"""

        try:
            response = self.ollama_client.generate(
                model='mistral',  # или ваша модель перевода
                prompt=prompt,
                stream=False,
                options={
                    'temperature': 0.3,  # Ниже температура = точнее перевод
                    'top_p': 0.9
                }
            )
            return response['response'].strip()
        except Exception as e:
            print(f"⚠ Ошибка перевода: {e}")
            return text

    def _get_context_from_qdrant(self, text, limit=2):
        """Получает релевантный контекст из Qdrant"""
        embedding = self._get_embedding(text)
        if not embedding:
            return ""

        try:
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=embedding,
                limit=limit
            )

            context_parts = []
            for result in search_results:
                if result.score > 0.5:  # Только релевантные результаты
                    original = result.payload.get('original_text', '')
                    translated = result.payload.get('translated_text', '')
                    context_parts.append(f"Оригинал: {original[:100]}...\nПеревод: {translated[:100]}...")

            return "\n".join(context_parts)
        except:
            return ""

    def _save_to_qdrant(self, original_text, translated_text, chunk_id):
        """Сохраняет переведённый текст в Qdrant для контекста"""
        embedding = self._get_embedding(original_text)
        if not embedding:
            return

        try:
            point_id = int(hashlib.md5(chunk_id.encode()).hexdigest(), 16) % (10 ** 8)

            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload={
                            'original_text': original_text[:500],
                            'translated_text': translated_text[:500],
                            'chunk_id': chunk_id
                        }
                    )
                ]
            )
        except Exception as e:
            print(f"⚠ Ошибка сохранения в Qdrant: {e}")

    def extract_images_from_pdf(self):
        """Извлекает все изображения из PDF используя pdfplumber"""
        import fitz  # PyMuPDF для извлечения изображений
        import os

        # Создаём директорию, если её нет
        os.makedirs("extracted_images", exist_ok=True)

        print("📸 Извлечение изображений из PDF...")

        pdf = fitz.open(self.pdf_path)
        print(f"Document('{self.pdf_path}')")

        for page_num, page in enumerate(pdf):
            image_list = page.get_images()
            print(f"Страница {page_num}: {len(image_list)} изображений")

            for img_index, img in enumerate(image_list):
                xref = img[0]
                pix = fitz.Pixmap(pdf, xref)

                if pix.n - pix.alpha < 4:  # GRAY or RGB
                    filename = f"extracted_images/page_{page_num}_img_{img_index}.png"
                    pix.save(filename)
                    print(f"✓ Извлечено изображение: {filename}")
                else:  # CMYK
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                    filename = f"extracted_images/page_{page_num}_img_{img_index}.png"
                    pix.save(filename)
                    print(f"✓ Извлечено изображение: {filename}")

        pdf.close()  # ВАЖНО: закройте PDF после использования


    def translate_pdf(self):
        """Основной метод перевода PDF"""
        translated_content = {}


        with pdfplumber.open(self.pdf_path) as pdf:
            total_pages = len(pdf.pages)

            for page_num, page in enumerate(pdf.pages):
                print(f"\n📄 Обработка страницы {page_num + 1}/{total_pages}")

                # Извлекаем текст
                text = page.extract_text()

                # Проверяем, что текст не None и не пуст
                if text is None or text.strip() == "":
                    print(f"  ℹ Страница пуста")
                    translated_content[page_num] = {'text': '', 'original_text': '', 'parts': []}
                    continue

                # Разделяем текст на части (код и обычный текст)
                parts = self._split_text_and_code(text)

                translated_parts = []
                for part_idx, part in enumerate(parts):
                    if part['type'] == 'code':
                        # Код остаётся на английском
                        translated_parts.append(part)
                        print(f"  💻 Код обнаружен, оставляется без изменений")
                    else:
                        # Переводим текст
                        text_content = part['content'].strip()
                        if not text_content:
                            continue

                        # Получаем контекст из Qdrant
                        context = self._get_context_from_qdrant(text_content)

                        # Переводим с контекстом
                        translated = self._translate_text(text_content, context)

                        # Сохраняем в Qdrant
                        chunk_id = f"page_{page_num}_part_{part_idx}"
                        self._save_to_qdrant(text_content, translated, chunk_id)

                        translated_parts.append({
                            'type': 'text',
                            'content': translated
                        })

                        print(f"  ✓ Переведено {len(translated.split())} слов")

                # Собираем переведённый текст страницы
                translated_text = ''.join([p['content'] for p in translated_parts])

                translated_content[page_num] = {
                    'text': translated_text,
                    'original_text': text,
                    'parts': translated_parts
                }

        return translated_content

    def translate_pdf_v2(self):
        """Основной метод перевода PDF с поддержкой OCR и Ollama"""
        import pytesseract
        import fitz

        pytesseract.pytesseract.pytesseract_cmd = '/opt/homebrew/bin/tesseract'

        translated_content = {}

        print(f"\n🔄 Перевод PDF: {self.pdf_path}")

        try:
            pdf = fitz.open(self.pdf_path)
            total_pages = len(pdf)
            print(f"📊 Всего страниц: {total_pages}")

            if total_pages == 0:
                print("❌ PDF не содержит страниц!")
                pdf.close()
                return translated_content

            for page_num in range(total_pages):
                page = pdf[page_num]
                print(f"\n📄 Страница {page_num + 1}/{total_pages}:")

                # Пробуем обычное извлечение текста
                text = page.get_text()
                print(f"   📝 Обычное извлечение: {len(text) if text else 0} символов")

                # Если нет текста, используем OCR
                if not text or text.strip() == "":
                    print(f"   📸 Применяю Tesseract OCR...")
                    try:
                        pix = page.get_pixmap(matrix=fitz.Matrix(150 / 72, 150 / 72))
                        img_data = pix.tobytes("png")

                        temp_img_path = f"/tmp/page_{page_num}.png"
                        with open(temp_img_path, "wb") as f:
                            f.write(img_data)

                        text = pytesseract.image_to_string(temp_img_path, lang='rus+eng')

                        import os
                        os.remove(temp_img_path)

                    except Exception as e:
                        print(f"   ❌ Ошибка OCR: {type(e).__name__}: {str(e)[:100]}")
                        text = ""

                # Если текст пуст, пропускаем
                if not text or text.strip() == "":
                    print(f"   ⚠ Страница пуста, пропускаю")
                    translated_content[page_num] = {
                        'text': '',
                        'original_text': '',
                        'parts': []
                    }
                    continue

                print(f"   ✅ Текст найден: {text[:50]}...")

                # ✨ ЗДЕСЬ ВЫЗЫВАЕМ OLLAMA ДЛЯ ПЕРЕВОДА ✨
                print(f"   🔄 Отправляю в Ollama для перевода...")
                translated_text = self._translate_with_ollama(text)

                translated_content[page_num] = {
                    'text': translated_text,  # ← Переведённый текст
                    'original_text': text,  # ← Оригинальный текст
                    'parts': []
                }

                print(f"   ✅ Переведено: {translated_text[:50]}...")

            pdf.close()

        except Exception as e:
            print(f"❌ Критическая ошибка: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()

        print(f"\n📊 Итого обработано страниц: {len(translated_content)}")
        return translated_content

    def _translate_with_ollama(self, text: str, source_lang: str = "English", target_lang: str = "Russian") -> str:
        """Переводит текст с помощью Ollama"""
        try:
            # Используйте нужную вам модель (llama2, mistral, neural-chat и т.д.)
            prompt = f"""You are a professional translator. Translate the following {source_lang} text to {target_lang}.
    Keep the technical terminology, formatting, and structure intact.
    Preserve code snippets and special characters.

    {source_lang} text:
    {text}

    {target_lang} translation:"""

            response = self.ollama_client.generate(
                model="qwen2.5-coder:7b",  # Быстрее и для кода лучше
                prompt=prompt,
                stream=False,
                options={"temperature": 0.3, "num_predict": 1500}
            )

            translated_text = response.get('response', '').strip()

            if not translated_text:
                print(f"   ⚠ Ollama вернула пустой ответ")
                return text

            return translated_text

        except Exception as e:
            print(f"   ❌ Ошибка Ollama: {type(e).__name__}: {str(e)}")
            return text  # Возвращаем оригинальный текст при ошибке

    def create_translated_pdf(self, translated_content, output_path):
        """Создаёт новый PDF с переведённым текстом и оригинальными картинками"""
        import os
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_JUSTIFY

        print(f"\n📝 Создание переведённого PDF...")
        print(f"Всего страниц в контенте: {len(translated_content)}")

        # Создаём стили
        styles = getSampleStyleSheet()
        style = ParagraphStyle(
            'CustomStyle',
            parent=styles['Normal'],
            fontSize=11,
            leading=14,
            alignment=TA_JUSTIFY,
            fontName='Helvetica',
            encoding='utf-8'
        )

        code_style = ParagraphStyle(
            'CodeStyle',
            parent=styles['Normal'],
            fontSize=9,
            leading=11,
            fontName='Courier',
            textColor='#333333',
            backColor='#f5f5f5',
            spaceAfter=12
        )

        story = []
        total_items = 0

        # Обрабатываем каждую страницу
        for page_num in sorted(translated_content.keys()):
            content = translated_content[page_num]

            print(f"\n➜ Страница {page_num}:")

            if not content:
                print(f"  ⚠ Пустой контент")
                continue

            # Получаем текст
            text = content.get('text', '').strip()
            parts = content.get('parts', [])

            print(f"  - Текст: {len(text)} символов")
            print(f"  - Частей: {len(parts)}")

            # Обрабатываем текст
            if text:
                if not parts:
                    # Нет разделения на части — добавляем весь текст
                    for para_text in text.split('\n'):
                        para_text = para_text.strip()
                        if para_text and len(para_text) > 2:
                            try:
                                safe_text = self._escape_html(para_text)
                                story.append(Paragraph(safe_text, style))
                                story.append(Spacer(1, 0.05 * inch))
                                total_items += 2
                            except Exception as e:
                                print(f"    ⚠ Ошибка текста: {str(e)[:80]}")
                else:
                    # Обрабатываем части
                    for idx, part in enumerate(parts):
                        part_type = part.get('type', 'text')
                        part_content = part.get('content', '').strip()

                        if not part_content or len(part_content) < 2:
                            continue

                        try:
                            if part_type == 'code':
                                safe_code = self._escape_html(part_content)
                                safe_code = safe_code.replace('\n', '<br/>')
                                story.append(Paragraph(safe_code, code_style))
                                story.append(Spacer(1, 0.1 * inch))
                                total_items += 2
                                print(f"    ✓ Код (часть {idx}): {len(part_content)} символов")
                            else:
                                safe_text = self._escape_html(part_content)
                                story.append(Paragraph(safe_text, style))
                                story.append(Spacer(1, 0.05 * inch))
                                total_items += 2
                                print(f"    ✓ Текст (часть {idx}): {len(part_content)} символов")
                        except Exception as e:
                            print(f"    ⚠ Ошибка части {idx}: {str(e)[:80]}")

            # Добавляем картинки
            img_keys = sorted([k for k in self.extracted_images.keys()
                               if k.startswith(f"page_{page_num}_")])

            print(f"  - Изображений: {len(img_keys)}")

            for img_key in img_keys:
                img_path = self.extracted_images[img_key]
                try:
                    if os.path.exists(img_path):
                        img = Image(img_path, width=5.5 * inch, height=5.5 * inch)
                        story.append(img)
                        story.append(Spacer(1, 0.2 * inch))
                        total_items += 2
                        print(f"    ✓ Изображение добавлено")
                    else:
                        print(f"    ⚠ Файл не найден: {img_path}")
                except Exception as e:
                    print(f"    ⚠ Ошибка изображения: {str(e)[:80]}")

            story.append(PageBreak())
            total_items += 1

        print(f"\n📊 Всего элементов в PDF: {total_items}")

        if total_items < 5:
            print("❌ Ошибка: PDF почти пуст, проверьте исходные данные!")
            return False

        # Создаём PDF
        try:
            doc = SimpleDocTemplate(
                output_path,
                pagesize=letter,
                rightMargin=0.5 * inch,
                leftMargin=0.5 * inch,
                topMargin=0.75 * inch,
                bottomMargin=0.75 * inch
            )

            print(f"\n⏳ Строится PDF ({total_items} элементов)...")
            doc.build(story)

            # Проверяем результат
            import os
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                print(f"✓ PDF создан успешно!")
                print(f"  Путь: {output_path}")
                print(f"  Размер: {file_size / 1024:.1f} КБ")

                if file_size < 5000:
                    print("⚠ Внимание: файл очень маленький")

                return True
            else:
                print(f"❌ Файл не найден по пути: {output_path}")
                return False

        except Exception as e:
            print(f"❌ Критическая ошибка при создании PDF:")
            print(f"   {e}")
            import traceback
            traceback.print_exc()
            return False

    def _escape_html(self, text):
        """Экранирует специальные HTML символы"""
        text = str(text)
        text = text.replace('&', '&amp;')
        text = text.replace('<', '&lt;')
        text = text.replace('>', '&gt;')
        text = text.replace('"', '&quot;')
        return text


    def diagnose_pdf(self):
        """Проверяет структуру PDF"""
        with pdfplumber.open(self.pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages[:3]):  # Первые 3 страницы
                print(f"\n📄 Страница {page_num + 1}:")

                text = page.extract_text()
                print(f"  Текст: {len(text) if text else 0} символов")
                if text:
                    print(f"  Первые 100 символов: {text[:100]}")

                tables = page.extract_tables()
                print(f"  Таблицы: {len(tables) if tables else 0}")

                images = page.images
                print(f"  Изображения: {len(images)}")

                # Проверяем, есть ли вообще какой-то контент
                if not text and not tables and not images:
                    print(f"  ⚠ ВНИМАНИЕ: Страница полностью пуста!")
        print("this")

# === ИСПОЛЬЗОВАНИЕ ===

if __name__ == "__main__":
    # Инициализируем переводчик
    translator = TechnicalBookTranslator(
        pdf_path="Concurrency in Go.pdf",
        qdrant_url="http://localhost:6333",
        ollama_host="http://localhost:11434",
        embedding_model="nomic-embed-text"  # или другая модель эмбеддингов
    )

    # Извлекаем изображения
    print("📸 Извлечение изображений из PDF...")
    translator.extract_images_from_pdf()

    # Переводим PDF
    print("\n🔄 Перевод PDF...")
    translated_content = translator.translate_pdf()
    if len(translated_content) == 0:
        translated_content = translator.translate_pdf_v2()

    # Создаём итоговый PDF
    print("\n🎨 Создание итогового PDF...")
    translator.create_translated_pdf(
        translated_content,
        output_path="translated_book.pdf"
    )


    print("\n✅ Готово!")