"""
Streamlit Web Interface для Instagram Photo Processor
Запуск: streamlit run app.py
"""

import streamlit as st
import os
import tempfile
from pathlib import Path
from datetime import datetime
import io
from PIL import Image
import numpy as np

from instagram_processor import InstagramProcessor, INSTAGRAM_WIDTH, INSTAGRAM_HEIGHT


# Configuration
st.set_page_config(
    page_title="Instagram Photo Processor",
    page_icon="👗",
    layout="wide"
)


# Готовые шаблоны
PRESETS = {
    "shop_vintage": {
        "name": "Магазин (тёплый)",
        "brightness": 20,
        "contrast": 1.15,
        "temperature": 6000,
    },
    "warm_vintage": {
        "name": "Тёплый винтаж",
        "brightness": 15,
        "contrast": 1.1,
        "temperature": 6500,
    },
    "neutral": {
        "name": "Нейтральный",
        "brightness": 5,
        "contrast": 1.05,
        "temperature": 5500,
    },
    "minimal": {
        "name": "Минималистичный",
        "brightness": 0,
        "contrast": 1.0,
        "temperature": 5200,
    }
}


def init_processor() -> InstagramProcessor:
    """Инициализация процессора."""
    output_dir = os.getenv("OUTPUT_DIR", "./output/instagram")
    return InstagramProcessor(output_dir=output_dir)


def load_image_for_preview(image_path: str) -> Image.Image:
    """Загрузить изображение для превью (поддержка NEF)."""
    path = Path(image_path)
    ext = path.suffix.upper()

    if ext == ".NEF":
        try:
            import rawpy
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


def save_uploaded_file(uploaded_file) -> str:
    """Сохранение загруженного файла во временную директорию."""
    temp_dir = Path(tempfile.gettempdir()) / "instagram_uploads"
    temp_dir.mkdir(exist_ok=True)

    temp_path = temp_dir / uploaded_file.name
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return str(temp_path)


def apply_corrections(img: Image.Image, brightness: int, contrast: float, temperature: int, auto_fix_edges: bool = True, vertical_offset: float = 0.0, target_size: tuple = (1080, 1350)) -> Image.Image:
    """Применить коррекцию и кадрирование - тот же алгоритм что в процессоре."""
    result = img.copy()

    # Только center crop с вертикальным смещением (без auto_crop в превью)
    target_w, target_h = target_size
    result = center_crop_with_offset(result, (target_w, target_h), vertical_offset)

    # 1. Brightness
    if brightness != 0:
        img_array = np.array(result).astype(np.float32)
        img_array = img_array + brightness
        img_array = np.clip(img_array, 0, 255)
        result = Image.fromarray(img_array.astype(np.uint8))

    # 2. Contrast
    if contrast != 1.0:
        img_array = np.array(result).astype(np.float32)
        img_array = ((img_array - 128) * contrast) + 128
        img_array = np.clip(img_array, 0, 255)
        result = Image.fromarray(img_array.astype(np.uint8))

    # 3. Temperature
    if temperature != 5500:
        img_array = np.array(result).astype(np.float32)
        temp_adjust = (temperature - 5500) / 1000
        img_array[:, :, 0] += temp_adjust * 15  # Red
        img_array[:, :, 2] -= temp_adjust * 10  # Blue
        img_array = np.clip(img_array, 0, 255)
        result = Image.fromarray(img_array.astype(np.uint8))

    return result


def center_crop_with_offset(img: Image.Image, target_size: tuple, vertical_offset_percent: float = 0.0) -> Image.Image:
    """Центрировать и обрезать до нужного соотношения с вертикальным смещением."""
    target_w, target_h = target_size
    img_w, img_h = img.size

    # Масштабируем чтобы покрыть цель
    scale = max(target_w / img_w, target_h / img_h)
    new_w = int(img_w * scale)
    new_h = int(img_h * scale)
    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Центрируем с вертикальным смещением
    left = (new_w - target_w) // 2
    offset_pixels = int(target_h * (vertical_offset_percent / 100.0))
    top = (new_h - target_h) // 2 + offset_pixels

    # Ограничиваем границы
    top = max(0, min(top, new_h - target_h))

    return img.crop((left, top, left + target_w, top + target_h))


def auto_crop_to_content(img: Image.Image) -> Image.Image:
    """Обрезать фон по контуру объекта."""
    import cv2
    import numpy as np

    img_array = np.array(img)
    h, w = img_array.shape[:2]

    try:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (9, 9), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest = max(contours, key=cv2.contourArea)
            x, y, cw, ch = cv2.boundingRect(largest)
            margin_x = int(cw * 0.1)
            margin_y = int(ch * 0.1)

            new_x = max(0, x - margin_x)
            new_y = max(0, y - margin_y)
            new_w = min(w - new_x, cw + margin_x * 2)
            new_h = min(h - new_y, ch + margin_y * 2)

            if new_w > w * 0.25 and new_h > h * 0.25:
                img_array = img_array[new_y:new_y+new_h, new_x:new_x+new_w]
                return Image.fromarray(img_array)
    except Exception:
        pass

    # Fallback - анализ яркости
    try:
        gray = np.mean(img_array, axis=2)
        brightness_per_row = gray.mean(axis=1)
        brightness_per_col = gray.mean(axis=0)
        avg = brightness_per_row.mean()

        # Ищем верх/низ
        top = 0
        for i in range(h):
            if brightness_per_row[i] < avg * 0.7:
                top = i
                break

        bottom = h
        for i in range(h - 1, -1, -1):
            if brightness_per_row[i] < avg * 0.7:
                bottom = i + 1
                break

        # Ищем лево/право
        left = 0
        for i in range(w):
            if brightness_per_col[i] < avg * 0.7:
                left = i
                break

        right = w
        for i in range(w - 1, -1, -1):
            if brightness_per_col[i] < avg * 0.7:
                right = i + 1
                break

        # Применяем если разумно
        if (right - left) > w * 0.35 and (bottom - top) > h * 0.35:
            img_array = img_array[top:bottom, left:right]
    except Exception:
        pass

    return Image.fromarray(img_array)


def main():
    st.title("👗 Instagram Photo Processor")
    st.markdown("Обработка фото винтажной одежды для Instagram")

    # Sidebar с настройками
    st.sidebar.header("Настройки")

    # Выбор шаблона
    st.sidebar.subheader("Шаблон")
    preset = st.sidebar.selectbox(
        "Выберите шаблон",
        list(PRESETS.keys()),
        format_func=lambda x: PRESETS[x]["name"]
    )

    # Описание шаблона
    p = PRESETS[preset]
    st.sidebar.info(
        f"**{p['name']}**\n\n"
        f"Яркость: {p['brightness']:+d}\n"
        f"Контраст: {p['contrast']:.2f}\n"
        f"Температура: {p['temperature']}K"
    )

    # Параметры обработки (начинаем со значений шаблона)
    st.sidebar.subheader("Коррекция")

    brightness = st.sidebar.slider(
        "Яркость",
        min_value=-100,
        max_value=100,
        value=p["brightness"],
        help="Осветление (+) или затемнение (-)"
    )

    contrast = st.sidebar.slider(
        "Контраст",
        min_value=0.8,
        max_value=1.5,
        value=p["contrast"],
        step=0.05,
        help="Контраст (1.0 = без изменений)"
    )

    temperature = st.sidebar.slider(
        "Температура",
        min_value=4000,
        max_value=8000,
        value=p["temperature"],
        help="Тёплый / Холодный оттенок"
    )

    # Кадрирование
    st.sidebar.subheader("Кадрирование")

    vertical_offset = st.sidebar.slider(
        "Вертикальное смещение",
        min_value=-30,
        max_value=30,
        value=0,
        step=5
    )

    target_size = st.sidebar.selectbox(
        "Размер",
        [
            (1080, 1350, "Instagram (4:5)"),
            (2160, 2700, "Высокое (4:5)"),
            (1080, 1920, "Story (9:16)"),
        ],
        index=1,
        format_func=lambda x: x[2]
    )

    # Качество
    st.sidebar.subheader("Экспорт")

    jpeg_quality = st.sidebar.slider(
        "Качество JPEG",
        min_value=50,
        max_value=100,
        value=100,
        step=5
    )

    auto_fix_edges = st.sidebar.checkbox(
        "Авто-исправление краёв",
        value=True
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
            temp_path = save_uploaded_file(uploaded_file)

            with st.spinner("Загрузка изображения..."):
                orig_img = load_image_for_preview(temp_path)

            if orig_img is None:
                st.error("Не удалось загрузить изображение")
            else:
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Оригинал**")
                    st.image(orig_img, caption="Загруженное фото", use_container_width=True)

                with col2:
                    st.markdown("**Превью**")
                    preview_img = apply_corrections(orig_img, brightness, contrast, temperature, auto_fix_edges, vertical_offset, (target_size[0], target_size[1]))
                    st.image(preview_img, caption="С коррекцией", use_container_width=True)

                st.markdown("---")
                process_btn = st.button("💾 Обработать и сохранить", type="primary", use_container_width=True)

                if process_btn:
                    with st.spinner("Обработка..."):
                        processor = init_processor()

                        result = processor.process_image(
                            image_path=temp_path,
                            preset=preset,
                            jpeg_quality=jpeg_quality,
                            target_size=(target_size[0], target_size[1]),
                            center_crop=True,
                            vertical_offset_percent=vertical_offset,
                            auto_fix_edges=auto_fix_edges,
                            brightness=brightness,
                            contrast=contrast,
                            saturation=1.0,  # не используем
                            temperature=temperature
                        )

                        st.success(f"Готово! Время: {result['processing_time']:.2f} сек")

                        col_a, col_b, col_c = st.columns(3)
                        col_a.metric("Размер", f"{result['file_size'] / 1024 / 1024:.2f} MB")
                        col_b.metric("Разрешение", f"{result['size'][0]}x{result['size'][1]}")
                        col_c.metric("Шаги", ", ".join(result["steps"]))

                        st.image(result["output"], caption="Обработанное фото", use_container_width=True)

                        with open(result["output"], "rb") as f:
                            st.download_button(
                                "📥 Скачать",
                                f,
                                file_name=os.path.basename(result["output"]),
                                mime="image/jpeg"
                            )

                        os.remove(temp_path)

    with tab2:
        st.subheader("Пакетная обработка")

        input_dir = st.text_input("Папка с исходными фото", value="D:/input")

        if os.path.isdir(input_dir):
            image_extensions = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".nef"}
            files = [
                f for f in os.listdir(input_dir)
                if Path(f).suffix.lower() in image_extensions
            ]

            st.write(f"Найдено файлов: {len(files)}")

            if files:
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
                                auto_fix_edges=auto_fix_edges,
                                brightness=brightness,
                                contrast=contrast,
                                saturation=1.0,
                                temperature=temperature
                            )
                            results.append({"file": filename, "status": "success", "result": result})
                        except Exception as e:
                            results.append({"file": filename, "status": "error", "error": str(e)})

                        progress_bar.progress((i + 1) / len(files))

                    success_count = sum(1 for r in results if r["status"] == "success")
                    st.success(f"Обработано: {success_count}/{len(files)}")

                    for r in results:
                        if r["status"] == "success":
                            st.write(f"✅ {r['file']} - {r['result']['file_size'] / 1024 / 1024:.2f} MB")
                        else:
                            st.write(f"❌ {r['file']} - {r['error']}")
        else:
            st.warning("Указанная папка не существует")

    st.markdown("---")
    st.markdown(
        f"""
        **Шаблоны:**
        - 🏪 **Магазин**: ярко +20, контраст 1.15, тёплый 6000K
        - 🍂 **Тёплый винтаж**: ярко +15, контраст 1.1, тёплый 6500K
        - ⚪ **Нейтральный**: лёгкая коррекция, естественные цвета
        - 🔵 **Минималистичный**: без изменений, слегка холодный
        """
    )


if __name__ == "__main__":
    main()
