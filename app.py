"""
Streamlit Web Interface для Instagram Photo Processor
Запуск: streamlit run app.py
"""

import streamlit as st
import os
import tempfile
from pathlib import Path
from datetime import datetime

from instagram_processor import InstagramProcessor, INSTAGRAM_WIDTH, INSTAGRAM_HEIGHT


# Configuration
st.set_page_config(
    page_title="Instagram Photo Processor",
    page_icon="👗",
    layout="wide"
)


def init_processor() -> InstagramProcessor:
    """Инициализация процессора."""
    output_dir = os.getenv("OUTPUT_DIR", "./output/instagram")
    return InstagramProcessor(output_dir=output_dir)


def save_uploaded_file(uploaded_file) -> str:
    """Сохранение загруженного файла во временную директорию."""
    temp_dir = Path(tempfile.gettempdir()) / "instagram_uploads"
    temp_dir.mkdir(exist_ok=True)

    # Сохраняем с оригинальным именем
    temp_path = temp_dir / uploaded_file.name
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return str(temp_path)


def main():
    st.title("👗 Instagram Photo Processor")
    st.markdown("Обработка фото винтажной одежды для Instagram")

    # Sidebar с настройками
    st.sidebar.header("Настройки")

    # Выбор пресета
    preset = st.sidebar.selectbox(
        "Пресет",
        ["shop_vintage", "warm_vintage", "neutral", "minimal"],
        format_func=lambda x: {
            "shop_vintage": "Магазин (тёплый винтаж)",
            "warm_vintage": "Тёплый винтаж",
            "neutral": "Нейтральный",
            "minimal": "Минималистичный"
        }.get(x, x)
    )

    # Параметры качества
    st.sidebar.subheader("Параметры")
    jpeg_quality = st.sidebar.slider(
        "Качество JPEG",
        min_value=50,
        max_value=100,
        value=80,
        step=5,
        help="Более высокое качество = больший размер файла"
    )

    vertical_offset = st.sidebar.slider(
        "Вертикальное смещение кадрирования",
        min_value=-20,
        max_value=20,
        value=0,
        step=5,
        help="Сдвиг вверх/вниз для лучшего кадрирования одежды"
    )

    target_size = st.sidebar.selectbox(
        "Размер",
        [
            (1080, 1350, "Instagram (4:5)"),
            (2160, 2700, "Высокое (4:5)"),
            (1080, 1920, "Story (9:16)"),
        ],
        format_func=lambda x: x[2]
    )

    auto_fix_edges = st.sidebar.checkbox(
        "Авто-исправление краёв",
        value=True,
        help="Удаление тёмных полос по краям"
    )

    # Main area
    tab1, tab2 = st.tabs(["📤 Загрузка", "📊 Пакетная обработка"])

    with tab1:
        st.subheader("Загрузить фото")

        uploaded_file = st.file_uploader(
            "Выберите изображение",
            type=["jpg", "jpeg", "png", "tiff", "tif", "nef"],
            help="Поддерживаются: JPG, PNG, TIFF, NEF (RAW)"
        )

        if uploaded_file is not None:
            # Показываем оригинал
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Оригинал**")
                st.image(uploaded_file, use_container_width=True)

            # Сохраняем во временный файл
            temp_path = save_uploaded_file(uploaded_file)

            # Обработка
            if st.button("Обработать", type="primary"):
                with st.spinner("Обработка..."):
                    processor = init_processor()

                    result = processor.process_image(
                        image_path=temp_path,
                        preset=preset,
                        jpeg_quality=jpeg_quality,
                        target_size=(target_size[0], target_size[1]),
                        center_crop=True,
                        vertical_offset_percent=vertical_offset,
                        auto_fix_edges=auto_fix_edges
                    )

                    # Показываем результат
                    with col2:
                        st.markdown("**Обработанное**")
                        st.image(result["output_path"], use_container_width=True)

                    # Информация
                    st.success(f"Готово! Время: {result['processing_time']:.2f} сек")

                    # Метрики
                    col_a, col_b, col_c = st.columns(3)
                    col_a.metric("Размер", f"{result['file_size'] / 1024 / 1024:.2f} MB")
                    col_b.metric("Разрешение", f"{result['width']}x{result['height']}")
                    col_c.metric("Шаги", ", ".join(result["steps"]))

                    # Скачивание
                    with open(result["output_path"], "rb") as f:
                        st.download_button(
                            "Скачать",
                            f,
                            file_name=os.path.basename(result["output_path"]),
                            mime="image/jpeg"
                        )

                    # Удаляем временный файл
                    os.remove(temp_path)

    with tab2:
        st.subheader("Пакетная обработка")

        input_dir = st.text_input(
            "Папка с исходными фото",
            value="D:/input",
            help="Укажите путь к папке с изображениями"
        )

        if os.path.isdir(input_dir):
            # Список файлов
            image_extensions = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".nef"}
            files = [
                f for f in os.listdir(input_dir)
                if Path(f).suffix.lower() in image_extensions
            ]

            st.write(f"Найдено файлов: {len(files)}")

            if files:
                # Показать первые 10
                st.write("Файлы:", ", ".join(files[:10]))
                if len(files) > 10:
                    st.write(f"... и ещё {len(files) - 10}")

                if st.button("Обработать все", type="primary"):
                    processor = init_processor()

                    results = []
                    progress_bar = st.progress(0)

                    for i, filename in enumerate(files):
                        file_path = os.path.join(input_dir, filename)

                        try:
                            result = processor.process_image(
                                image_path=file_path,
                                preset=preset,
                                jpeg_quality=jpeg_quality,
                                target_size=(target_size[0], target_size[1]),
                                center_crop=True,
                                vertical_offset_percent=vertical_offset,
                                auto_fix_edges=auto_fix_edges
                            )
                            results.append({"file": filename, "status": "success", "result": result})
                        except Exception as e:
                            results.append({"file": filename, "status": "error", "error": str(e)})

                        progress_bar.progress((i + 1) / len(files))

                    # Итоги
                    success_count = sum(1 for r in results if r["status"] == "success")
                    st.success(f"Обработано: {success_count}/{len(files)}")

                    # Показать результаты
                    for r in results:
                        if r["status"] == "success":
                            st.write(f"✅ {r['file']} - {r['result']['file_size'] / 1024 / 1024:.2f} MB")
                        else:
                            st.write(f"❌ {r['file']} - {r['error']}")
        else:
            st.warning("Указанная папка не существует")

    # Информация в footer
    st.markdown("---")
    st.markdown(
        """
        **Поддерживаемые форматы:** JPG, PNG, TIFF, NEF (Nikon RAW)

        **Параметры обработки:**
        - Кадрирование 4:5 (1080x1350 или 2160x2700)
        - Коррекция экспозиции, теней, светов
        - Тёплая цветокоррекция (vintage style)
        - Удаление артефактов по краям
        """
    )


if __name__ == "__main__":
    main()
