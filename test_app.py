"""
AI Instagram Pipeline Test Interface
Тестовый интерфейс для проверки всех AI-модулей

Запуск: streamlit run test_app.py
"""
import streamlit as st
import os
import tempfile
from pathlib import Path
from datetime import datetime
import time

from PIL import Image

from ai_preset_selector import ai_select_preset, AVAILABLE_PRESETS
from instagram_processor import InstagramProcessor
from comfyui_enhancer import ComfyUIEnhancer, check_comfyui_enhancer
from product_desc_generator import ProductDescriptionGenerator
from ai_pipeline import InstaAutoPipeline


# Configuration
st.set_page_config(
    page_title="AI Instagram Pipeline Test",
    page_icon="🤖",
    layout="wide"
)


def load_image_for_preview(image_path: str):
    """Загрузить изображение для превью."""
    path = Path(image_path)
    ext = path.suffix.upper()

    if ext == ".NEF":
        try:
            import rawpy
            import numpy as np
            with rawpy.imread(image_path) as raw:
                rgb = raw.postprocess(
                    use_camera_wb=True,
                    no_auto_bright=True,
                    output_bps=16,
                )
            rgb_8 = (rgb / 256).astype(np.uint8)
            return Image.fromarray(rgb_8)
        except Exception as e:
            st.error(f"Ошибка загрузки NEF: {e}")
            return None
    else:
        img = Image.open(image_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img


def save_uploaded_file(uploaded_file):
    """Сохранение загруженного файла."""
    temp_dir = Path(tempfile.gettempdir()) / "ai_pipeline_test"
    temp_dir.mkdir(exist_ok=True)

    temp_path = temp_dir / uploaded_file.name
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return str(temp_path)


def main():
    st.title("🤖 AI Instagram Pipeline - Тестирование")
    st.markdown("### Полный автоматический пайплайн для обработки фото")

    # Проверка статусов
    with st.sidebar:
        st.header("📊 Статус систем")

        # Проверка Ollama
        try:
            from cli_wrappers import OllamaWrapper
            ollama_ok = OllamaWrapper.check_connection()
            st.success(f"✅ Ollama: OK" if ollama_ok else "❌ Ollama: Не подключён")
        except:
            st.error("❌ Ollama: Ошибка")

        # Проверка ComfyUI
        comfy_status = check_comfyui_enhancer()
        st.success(f"✅ ComfyUI: OK" if comfy_status.get("comfyui_connected") else "⚠️ ComfyUI: Не подключён")

        # Проверка Claude
        try:
            from cli_wrappers import ClaudeWrapper
            claude_ok = ClaudeWrapper.check_connection()
            st.success(f"✅ Claude CLI: OK" if claude_ok else "⚠️ Claude CLI: Не подключён (fallback: Ollama)")
        except:
            st.info("ℹ️ Claude: Fallback на Ollama")

        st.divider()

        # Настройки
        st.header("⚙️ Настройки")

        use_ai_preset = st.checkbox("AI-подбор пресета", value=True)
        use_comfyui = st.checkbox("ComfyUI улучшение", value=True)
        use_ai_desc = st.checkbox("AI-описание", value=True)

        target_size = st.selectbox(
            "Размер",
            [(1080, 1350, "Instagram (4:5)"),
             (2160, 2700, "Высокое (4:5)"),
             (1080, 1920, "Story (9:16)")],
            index=1,
            format_func=lambda x: x[2]
        )

        jpeg_quality = st.slider("Качество JPEG", 50, 100, 90)

    # Main interface
    tab1, tab2, tab3, tab4 = st.tabs([
        "📤 Загрузка",
        "🔬 AI-Анализ",
        "⚡ Обработка",
        "📝 Результат"
    ])

    # Инициализация состояния
    if 'processed_image' not in st.session_state:
        st.session_state.processed_image = None
    if 'original_image' not in st.session_state:
        st.session_state.original_image = None
    if 'analysis_result' not in st.session_state:
        st.session_state.analysis_result = None
    if 'description_result' not in st.session_state:
        st.session_state.description_result = None

    with tab1:
        st.subheader("Загрузка изображения")

        # Выбор источника
        source = st.radio("Источник", ["Загрузить файл", "Выбрать из папки"], horizontal=True)

        image_path = None

        if source == "Загрузить файл":
            uploaded_file = st.file_uploader(
                "Выберите изображение",
                type=["jpg", "jpeg", "png", "tif", "tiff", "nef"]
            )

            if uploaded_file:
                image_path = save_uploaded_file(uploaded_file)
                st.session_state.original_image = image_path

        else:
            input_dir = st.text_input("Папка с фото", value="D:/input")

            if os.path.isdir(input_dir):
                files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))]
                selected_file = st.selectbox("Выберите файл", files)

                if selected_file:
                    image_path = os.path.join(input_dir, selected_file)
                    st.session_state.original_image = image_path

        # Показ оригинала
        if st.session_state.original_image:
            st.divider()
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Оригинал**")
                orig_img = load_image_for_preview(st.session_state.original_image)
                if orig_img:
                    st.image(orig_img, caption=f"Размер: {orig_img.size[0]}x{orig_img.size[1]}", use_container_width=True)

            with col2:
                st.markdown("**Информация**")
                st.write(f"Файл: {Path(st.session_state.original_image).name}")
                if orig_img:
                    st.write(f"Размер: {orig_img.size[0]} x {orig_img.size[1]}")
                    st.write(f"Формат: {orig_img.format}")

    with tab2:
        st.subheader("🔬 AI-Анализ изображения")

        if not st.session_state.original_image:
            st.info("Сначала загрузите изображение на вкладке 'Загрузка'")
        else:
            # Кнопка анализа
            if st.button("🔍 Запустить AI-анализ", type="primary", use_container_width=True):
                with st.spinner("Анализ..."):
                    # AI-подбор пресета
                    if use_ai_preset:
                        st.session_state.analysis_result = ai_select_preset(
                            st.session_state.original_image,
                            use_ai=True
                        )
                    else:
                        # Ручной выбор
                        preset_name = st.selectbox("Выберите пресет", list(AVAILABLE_PRESETS.keys()))
                        preset = AVAILABLE_PRESETS[preset_name]
                        st.session_state.analysis_result = {
                            "preset": preset_name,
                            "preset_name": preset["name"],
                            "parameters": {
                                "brightness": preset["brightness"],
                                "contrast": preset["contrast"],
                                "temperature": preset["temperature"]
                            },
                            "ai_used": False
                        }

            # Результат анализа
            if st.session_state.analysis_result:
                result = st.session_state.analysis_result
                st.divider()

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Выбранный пресет**")
                    st.success(f"🎯 {result.get('preset_name', result.get('preset'))}")
                    st.write(f"AI использован: {'🤖 Да' if result.get('ai_used') else '👤 Ручной выбор'}")

                    if result.get("reasoning"):
                        st.write(f"Обоснование: {result['reasoning']}")

                with col2:
                    st.markdown("**Параметры**")
                    params = result.get("parameters", {})
                    st.metric("Яркость", f"{params.get('brightness', 0):+d}")
                    st.metric("Контраст", f"{params.get('contrast', 1.0):.2f}")
                    st.metric("Температура", f"{params.get('temperature', 5500)}K")

                # Анализ изображения
                if result.get("analysis"):
                    st.divider()
                    st.markdown("**Анализ изображения**")
                    analysis = result["analysis"]

                    cols = st.columns(4)
                    cols[0].metric("Яркость", f"{analysis.get('brightness', 0):.0f}")
                    cols[1].metric("Контраст", f"{analysis.get('contrast', 0):.0f}")
                    cols[2].metric("Цвет", analysis.get("color_cast", "unknown"))
                    cols[3].metric("Проблема", analysis.get("quality_issue", "none"))

    with tab3:
        st.subheader("⚡ Обработка изображения")

        if not st.session_state.original_image:
            st.info("Сначала загрузите изображение")
        else:
            # Ручные параметры
            st.markdown("**Параметры обработки**")
            col1, col2, col3, col4 = st.columns(4)

            brightness = col1.number_input("Яркость", value=st.session_state.analysis_result.get("parameters", {}).get("brightness", 10) if st.session_state.analysis_result else 10, step=5)
            contrast = col2.number_input("Контраст", value=st.session_state.analysis_result.get("parameters", {}).get("contrast", 1.15) if st.session_state.analysis_result else 1.15, step=0.05)
            temperature = col3.number_input("Температура", value=st.session_state.analysis_result.get("parameters", {}).get("temperature", 6000) if st.session_state.analysis_result else 6000, step=100)
            vertical_offset = col4.number_input("Смещение", value=0, step=5)

            # Категория товара
            category = st.selectbox(
                "Категория товара",
                ["vintage_clothing", "modern_clothing", "accessories", "shoes", "bags", "jewelry"],
                format_func=lambda x: {
                    "vintage_clothing": "Винтажная одежда",
                    "modern_clothing": "Современная одежда",
                    "accessories": "Аксессуары",
                    "shoes": "Обувь",
                    "bags": "Сумки",
                    "jewelry": "Украшения"
                }.get(x, x)
            )

            col1, col2 = st.columns(2)
            brand = col1.text_input("Бренд (опционально)")
            price = col2.text_input("Цена (опционально)")

            st.divider()

            # Кнопка обработки
            if st.button("🚀 Запустить полную обработку", type="primary", use_container_width=True):
                with st.spinner("Обработка..."):
                    try:
                        # Создаём пайплайн
                        pipeline = InstaAutoPipeline(
                            use_ai_preset=use_ai_preset,
                            use_comfyui_enhance=use_comfyui,
                            use_ai_description=use_ai_desc
                        )

                        # Запускаем обработку
                        result = pipeline.process(
                            image_path=st.session_state.original_image,
                            category=category,
                            brand=brand or None,
                            price=price or None,
                            target_size=(target_size[0], target_size[1]),
                            jpeg_quality=jpeg_quality
                        )

                        st.session_state.processed_result = result

                        if result.get("status") == "success":
                            st.success(f"✅ Обработка завершена за {result.get('processing_time', 0):.1f} сек")
                        else:
                            st.error(f"❌ Ошибка: {result.get('error')}")

                    except Exception as e:
                        st.error(f"❌ Ошибка: {str(e)}")

            # Показ результата
            if hasattr(st.session_state, 'processed_result'):
                result = st.session_state.processed_result
                if result.get("status") == "success":
                    st.divider()
                    st.markdown("**Результат обработки**")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("**До/После**")
                        # Показываем обработанное
                        if result.get("output_path") and os.path.exists(result["output_path"]):
                            processed_img = Image.open(result["output_path"])
                            st.image(processed_img, caption="Обработанное", use_container_width=True)

                    with col2:
                        st.markdown("**Шаги обработки**")
                        for step in result.get("steps", []):
                            st.write(f"• {step}")

                        st.metric("Время обработки", f"{result.get('processing_time', 0):.1f} сек")

    with tab4:
        st.subheader("📝 Результат и описание")

        if not hasattr(st.session_state, 'processed_result'):
            st.info("Сначала обработайте изображение")
        else:
            result = st.session_state.processed_result

            if result.get("status") == "success":
                # Генерация описания
                st.markdown("### AI-Сгенерированное описание")

                if result.get("description"):
                    desc = result["description"]

                    st.text_input("Название", value=desc.get("title", ""), disabled=True)
                    st.text_area("Описание", value=desc.get("description", ""), height=100, disabled=True)
                    st.text_input("Хештеги", value=desc.get("hashtags", ""), disabled=True)

                    cols = st.columns(3)
                    cols[0].text_input("Размер", value=desc.get("size", ""), disabled=True)
                    cols[1].text_input("Состояние", value=desc.get("condition", ""), disabled=True)
                    cols[2].text_input("Цвет", value=desc.get("color", ""), disabled=True)

                    # Готовый пост
                    st.divider()
                    st.markdown("### 📝 Готовый пост для Instagram")

                    if result.get("instagram_post"):
                        st.text_area("Пост", value=result["instagram_post"], height=300, disabled=True)

                        # Кнопка копирования
                        st.code(result["instagram_post"], language=None)

                # Скачивание
                st.divider()
                if result.get("output_path") and os.path.exists(result["output_path"]):
                    with open(result["output_path"], "rb") as f:
                        st.download_button(
                            "📥 Скачать обработанное изображение",
                            f,
                            file_name=os.path.basename(result["output_path"]),
                            mime="image/jpeg"
                        )

                    st.markdown(f"**Файл сохранён:** `{result['output_path']}`")
            else:
                st.error("Ошибка обработки")


if __name__ == "__main__":
    main()
