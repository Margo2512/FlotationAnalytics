import streamlit as st
import pandas as pd
import sqlite3


class TrackingDatabase:
    def __init__(self, db_name="tracking_analysis.db"):
        self.db_name = db_name
        self.init_db()

    def init_db(self):
        with sqlite3.connect(self.db_name) as conn:
            c = conn.cursor()
            c.execute(
                """CREATE TABLE IF NOT EXISTS analyses
                         (id INTEGER PRIMARY KEY AUTOINCREMENT,
                          filename TEXT,
                          model_name TEXT,
                          avg_displacement REAL,
                          tracking_coverage REAL,
                          temporal_consistency REAL,
                          optical_flow REAL,
                          avg_track_length REAL,
                          max_active_tracks INTEGER,
                          tracking_score REAL,
                          processing_time REAL,
                          timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)"""
            )
            conn.commit()

    def save_analysis(self, data):
        with sqlite3.connect(self.db_name) as conn:
            c = conn.cursor()
            try:
                c.execute(
                    """INSERT INTO analyses 
                             (filename, model_name, avg_displacement, tracking_coverage, 
                              temporal_consistency, optical_flow, 
                              avg_track_length, max_active_tracks, tracking_score, processing_time)
                             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        str(data.get("filename", "")),
                        str(data.get("model_name", "")),
                        float(data.get("avg_displacement", 0)),
                        float(data.get("tracking_coverage", 0)),
                        float(data.get("temporal_consistency", 0)),
                        float(data.get("optical_flow", 0)),
                        float(data.get("avg_track_length", 0)),
                        int(data.get("max_active_tracks", 0)),
                        float(data.get("tracking_score", 0)),
                        float(data.get("processing_time", 0)),
                    ),
                )
                conn.commit()
            except Exception as e:
                st.error(f"Ошибка при сохранении в базу данных: {str(e)}")
                conn.rollback()

    def get_recent_analyses(self, limit=10):
        with sqlite3.connect(self.db_name) as conn:
            try:
                query = """SELECT 
                            id,
                            filename,
                            model_name,
                            avg_displacement,
                            tracking_coverage,
                            temporal_consistency,
                            optical_flow,
                            avg_track_length,
                            max_active_tracks,
                            tracking_score,
                            processing_time,
                            timestamp
                          FROM analyses 
                          ORDER BY timestamp DESC 
                          LIMIT ?"""
                df = pd.read_sql(query, conn, params=(limit,))

                numeric_cols = [
                    "avg_displacement",
                    "tracking_coverage",
                    "temporal_consistency",
                    "optical_flow",
                    "avg_track_length",
                    "max_active_tracks",
                    "tracking_score",
                    "processing_time",
                ]
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")

                return df
            except Exception as e:
                st.error(f"Ошибка при чтении из базы данных: {str(e)}")
                return pd.DataFrame()

    def display_recent_analyses(self):
        df = self.get_recent_analyses()

        if df.empty:
            st.warning("В базе данных пока нет записей")
            return

        display_df = df.drop(columns=["id"]).copy()

        display_df = display_df.rename(
            columns={
                "filename": "Файл",
                "model_name": "Модель",
                "avg_displacement": "Ср. смещение",
                "tracking_coverage": "Покрытие",
                "temporal_consistency": "Согласованность",
                "optical_flow": "Опт. поток",
                "avg_track_length": "Длина трека",
                "max_active_tracks": "Макс. треков",
                "tracking_score": "Оценка",
                "processing_time": "Время (сек)",
                "timestamp": "Время анализа",
            }
        )

        styled_df = display_df.style.format(
            {
                "Ср. смещение": "{:.2f}",
                "Покрытие": "{:.2%}",
                "Согласованность": "{:.2f}",
                "Опт. поток": "{:.2f}",
                "Длина трека": "{:.1f}",
                "Оценка": "{:.3f}",
                "Время (сек)": "{:.2f}",
            }
        )

        st.dataframe(
            styled_df,
            use_container_width=True,
            height=min(400, 35 * (len(display_df) + 35)),
        )
