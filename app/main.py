import streamlit as st
import tempfile
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import torch
from CounTR import models_mae_cross
from classes import TrackingDatabase, VideoTracker


def main():
    st.set_page_config(layout="wide")
    st.title("Автоматизация анализа флотации")

    db = TrackingDatabase()

    tab1, tab2, tab3 = st.tabs(
        ["Основные метрики", "Покадровый просмотр", "Детальные графики"]
    )

    tracker_configs = {
        "ByteTrack + YOLOv11s": "bytetrack.yaml",
        "BoT-SORT + YOLOv11s": "bot-sort.yaml",
        "DeepSORT + YOLOv11s": None,
        "Мой трекер + CounTR": None,
    }

    uploaded_file = st.sidebar.file_uploader(
        "Выберите видео", type=["mp4", "avi", "mov"]
    )
    tracker_type = st.sidebar.selectbox("Выберите трекер", list(tracker_configs.keys()))

    if "tracker_results" not in st.session_state:
        st.session_state.tracker_results = None
        st.session_state.processed = False

    if uploaded_file and st.sidebar.button("Запустить оценку"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name

        with st.spinner("Обработка видео..."):
            if tracker_type == "Мой трекер + CounTR":
                model_path = "model/FSC147.pth"
                model = models_mae_cross.__dict__["mae_vit_base_patch16"](
                    norm_pix_loss="store_true"
                )
                checkpoint = torch.load(
                    model_path, map_location="cpu", weights_only=False
                )
                model.load_state_dict(checkpoint["model"], strict=False)
                tracker = VideoTracker(model_path, tracker_type)
                tracker.model = model
            else:
                tracker = VideoTracker("model/YOLOv11s.pt", tracker_type)
            metrics = tracker.process_video(video_path)

            print("Debug - metrics keys:", metrics.keys())

            analysis_data = {
                "filename": uploaded_file.name,
                "model_name": tracker_type,
                "avg_displacement": (
                    np.mean(metrics["displacement"]) if metrics["displacement"] else 0
                ),
                "tracking_coverage": (
                    np.mean(metrics["coverage"]) if metrics["coverage"] else 0
                ),
                "temporal_consistency": (
                    np.mean(metrics["temporal_consistency"])
                    if metrics["temporal_consistency"]
                    else 0
                ),
                "optical_flow": (
                    np.mean(metrics["optical_flow"]) if metrics["optical_flow"] else 0
                ),
                "avg_track_length": (
                    np.mean(list(metrics["track_lengths"].values()))
                    if metrics["track_lengths"]
                    else 0
                ),
                "max_active_tracks": (
                    max(metrics["max_active_tracks_history"])
                    if metrics["max_active_tracks_history"]
                    else 0
                ),
                "tracking_score": metrics["final_score"],
                "processing_time": metrics["processing_time"],
            }
            db.save_analysis(analysis_data)

            st.session_state.tracker_results = {
                "metrics": metrics,
                "processed_frames": tracker.processed_frames,
                "quality_analyzer": tracker.quality_analyzer,
            }
            st.session_state.processed = True
            os.unlink(video_path)

    if st.session_state.tracker_results and st.session_state.processed:
        metrics = st.session_state.tracker_results["metrics"]
        processed_frames = st.session_state.tracker_results["processed_frames"]
        quality_analyzer = st.session_state.tracker_results.get(
            "quality_analyzer", None
        )

        with tab1:
            st.success("Обработка завершена!")

            cols = st.columns(3)
            with cols[0]:
                avg_displacement = (
                    np.mean(metrics["displacement"]) if metrics["displacement"] else 0
                )
                st.metric("Среднее смещение", f"{avg_displacement:.2f} px")

                avg_coverage = (
                    np.mean(metrics["coverage"]) if metrics["coverage"] else 0
                )
                st.metric("Полнота обнаружения", f"{avg_coverage:.2%}")

            with cols[1]:
                avg_optical_flow = (
                    np.mean(metrics["optical_flow"]) if metrics["optical_flow"] else 0
                )
                st.metric("Средний оптический поток", f"{avg_optical_flow:.2f} px")

                avg_temp_consistency = (
                    np.mean(metrics["temporal_consistency"])
                    if metrics["temporal_consistency"]
                    else 0
                )
                st.metric("Темпоральная согласованность", f"{avg_temp_consistency:.2f}")

            with cols[2]:
                avg_track_length = (
                    np.mean(list(metrics["track_lengths"].values()))
                    if metrics["track_lengths"]
                    else 0
                )
                st.metric("Средняя длина трека", f"{avg_track_length:.2f} кадров")

                max_active_tracks = (
                    max(metrics["max_active_tracks_history"])
                    if metrics["max_active_tracks_history"]
                    else 0
                )
                st.metric("Макс. активных треков", max_active_tracks)

            st.metric(
                "Итоговая оценка",
                f"{metrics['final_score']:.4f}",
                help="Оценка от 0 до 1, где 1 - наилучшее качество трекинга",
            )

            if "bubbles_per_frame" in metrics and metrics["bubbles_per_frame"]:
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.subheader("Динамика обнаружения пузырей")
                    chart_data = pd.DataFrame(
                        {
                            "Кадр": range(len(metrics["bubbles_per_frame"])),
                            "Количество пузырей": metrics["bubbles_per_frame"],
                        }
                    )
                    st.line_chart(chart_data.set_index("Кадр"))

                with col2:
                    st.subheader("Статистика")
                    stats = {
                        "Всего обнаружено": sum(metrics["bubbles_per_frame"]),
                        "Максимум в кадре": max(metrics["bubbles_per_frame"]),
                        "Среднее значение": round(
                            np.mean(metrics["bubbles_per_frame"]), 1
                        ),
                        "Пустых кадров": metrics["bubbles_per_frame"].count(0),
                    }
                    for k, v in stats.items():
                        st.metric(k, v)

        with tab2:
            st.header("Покадровый просмотр результатов")

            frame_idx = st.slider("Выберите кадр", 0, len(processed_frames) - 1, 0)
            frame_data = processed_frames[frame_idx]

            st.image(
                frame_data["frame"],
                channels="BGR",
                caption=f"Кадр {frame_idx+1} из {len(processed_frames)}",
            )

            st.write(f"Количество пузырей: {frame_data['bubbles_count']}")

        with tab3:
            st.header("Детальные графики метрик")

            if quality_analyzer:
                quality_analyzer.plot_metrics()
            else:
                if (
                    "max_active_tracks_history" in metrics
                    and metrics["max_active_tracks_history"]
                ):
                    st.subheader("Активные треки по кадрам")
                    fig, ax = plt.subplots(figsize=(12, 4))
                    ax.plot(
                        metrics["max_active_tracks_history"],
                        "c-",
                        label="Активные треки",
                    )
                    ax.axhline(
                        y=metrics["max_active_tracks"],
                        color="r",
                        linestyle="--",
                        label=f'Максимум: {metrics["max_active_tracks"]}',
                    )
                    ax.set_xlabel("Номер кадра")
                    ax.set_ylabel("Количество треков")
                    ax.grid(True)
                    ax.legend()
                    st.pyplot(fig)
                    plt.close(fig)

                if "optical_flow" in metrics and metrics["optical_flow"]:
                    st.subheader("Оптический поток")
                    fig, ax = plt.subplots(figsize=(12, 4))
                    ax.plot(metrics["optical_flow"], "m-", label="Оптический поток")
                    ax.set_xlabel("Номер кадра")
                    ax.set_ylabel("Величина потока (пиксели)")
                    ax.grid(True)
                    ax.legend()
                    st.pyplot(fig)
                    plt.close(fig)

                if "track_lengths" in metrics and metrics["track_lengths"]:
                    st.subheader("Распределение длин треков")
                    fig, ax = plt.subplots(figsize=(12, 4))

                    lengths = metrics["track_lengths"]
                    if isinstance(lengths, dict):
                        lengths = list(lengths.values())
                    elif not isinstance(lengths, (list, np.ndarray)):
                        lengths = []

                    if lengths:
                        ax.hist(
                            lengths,
                            bins=20,
                            color="orange",
                            edgecolor="black",
                            alpha=0.7,
                        )

                        mean_length = np.mean(lengths)
                        ax.axvline(
                            mean_length,
                            color="r",
                            linestyle="--",
                            label=f"Среднее: {mean_length:.1f} кадров",
                        )

                        ax.set_xlabel("Длина трека (кадры)")
                        ax.set_ylabel("Количество треков")
                        ax.grid(True)
                        ax.legend()
                        st.pyplot(fig)
                        plt.close(fig)
                    else:
                        st.warning(
                            "Нет данных о длинах треков для построения гистограммы"
                        )

                if "displacement" in metrics and len(metrics["displacement"]) > 0:
                    st.subheader("Среднее смещение объектов между кадрами")
                    fig, ax = plt.subplots(figsize=(12, 4))
                    ax.plot(metrics["displacement"], "r-", label="Смещение (пиксели)")
                    ax.set_xlabel("Номер кадра")
                    ax.set_ylabel("Смещение")
                    ax.grid(True)
                    ax.legend()
                    st.pyplot(fig)
                    plt.close(fig)

                if "coverage" in metrics and len(metrics["coverage"]) > 0:
                    st.subheader("Полнота обнаружения")
                    fig, ax = plt.subplots(figsize=(12, 4))
                    ax.plot(metrics["coverage"], "g-", label="Процент совпадений")
                    ax.set_xlabel("Номер кадра")
                    ax.set_ylabel("Процент")
                    ax.grid(True)
                    ax.legend()
                    st.pyplot(fig)
                    plt.close(fig)

                if (
                    "temporal_consistency" in metrics
                    and len(metrics["temporal_consistency"]) > 0
                ):
                    st.subheader("Темпоральная согласованность (IoU)")
                    fig, ax = plt.subplots(figsize=(12, 4))
                    ax.plot(metrics["temporal_consistency"], "m-", label="Средний IoU")
                    ax.set_xlabel("Номер кадра")
                    ax.set_ylabel("IoU")
                    ax.grid(True)
                    ax.legend()
                    st.pyplot(fig)
                    plt.close(fig)

    elif not st.session_state.processed:
        st.info("Загрузите видеофайл и нажмите кнопку 'Запустить оценку'")

    st.header("История запусков")
    db.display_recent_analyses()


if __name__ == "__main__":
    main()
