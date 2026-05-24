from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from voice_access_pipeline_final import (
    DEFAULT_DEVICE,
    DEFAULT_WORK_DIR,
    ProjectPaths,
    enroll_new_allowed_speaker_operational,
    get_paths,
    load_experiment_config,
    load_speaker_index,
    predict_wav_file_operational,
)


FINAL_EXPERIMENT_NAME = "final_operational_model"
FINAL_CHECKPOINT_NAME = "final_operational_model.pt"


def save_enrollment_audio_to_speaker_dir(
    paths: ProjectPaths,
    speaker_name: str,
    uploaded_files=None,
    recorded_samples=None,
) -> Path:
    """Save uploaded and microphone-recorded .wav samples for one enrolled speaker."""
    speaker_name = speaker_name.strip()
    if not speaker_name:
        raise ValueError("Speaker name cannot be empty.")

    uploaded_files = uploaded_files or []
    recorded_samples = recorded_samples or []

    target_dir = paths.app_added_dir / speaker_name
    if target_dir.exists():
        for old_file in target_dir.glob("*.wav"):
            old_file.unlink()
    target_dir.mkdir(parents=True, exist_ok=True)

    kept = 0
    for i, sample_bytes in enumerate(recorded_samples, start=1):
        out_path = target_dir / f"recorded_sample_{i:03d}.wav"
        with open(out_path, "wb") as f:
            f.write(sample_bytes)
        kept += 1

    for uploaded in uploaded_files:
        if Path(uploaded.name).suffix.lower() != ".wav":
            continue
        out_path = target_dir / Path(uploaded.name).name
        with open(out_path, "wb") as f:
            f.write(uploaded.getbuffer())
        kept += 1

    if kept == 0:
        raise ValueError("No .wav files were provided for enrollment.")
    return target_dir


def save_uploaded_files_to_speaker_dir(paths: ProjectPaths, speaker_name: str, uploaded_files) -> Path:
    return save_enrollment_audio_to_speaker_dir(
        paths=paths,
        speaker_name=speaker_name,
        uploaded_files=uploaded_files,
        recorded_samples=[],
    )


def _resolve_audio_input(recorded_wav, uploaded_wav) -> tuple[Path | None, str | None]:
    if recorded_wav is None and uploaded_wav is None:
        return None, None

    source_name = "microphone" if recorded_wav is not None else "uploaded file"
    payload = recorded_wav if recorded_wav is not None else uploaded_wav

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(payload.getbuffer())
        return Path(tmp.name), source_name


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _load_final_artifacts(work_dir_str: str, device: str):
    root_paths = get_paths(Path(work_dir_str))

    selected_config = load_experiment_config(
        root_paths.work_dir,
        FINAL_EXPERIMENT_NAME,
    )
    selected_config.train.device = device

    selected_paths = get_paths(selected_config)
    selected_checkpoint = selected_paths.model_dir / FINAL_CHECKPOINT_NAME

    if not selected_checkpoint.exists():
        raise FileNotFoundError(
            f"Final checkpoint not found: {selected_checkpoint}\n"
            "Run the workflow notebook first or check FINAL_CHECKPOINT_NAME."
        )

    if not selected_paths.speaker_index_path.exists():
        raise FileNotFoundError(
            f"Speaker index not found: {selected_paths.speaker_index_path}\n"
            "Run build_speaker_index_for_experiment(final_config) in the workflow notebook first."
        )

    speaker_index_payload = load_speaker_index(selected_paths.speaker_index_path)
    if len(speaker_index_payload["prototypes"]) == 0:
        raise ValueError(
            "Speaker index exists, but it contains no prototypes. "
            "Rebuild the speaker index for the final model."
        )

    return root_paths, selected_config, selected_paths, selected_checkpoint, speaker_index_payload


def main():
    st.set_page_config(page_title="Voice Allow App", layout="wide")
    st.title("Voice Allow App")
    st.caption(
        "Operational app built around one final mode: "
        "prototype-based admission with incremental enrollment of new allowed speakers."
    )

    with st.sidebar:
        work_dir_str = st.text_input("WORK_DIR", str(DEFAULT_WORK_DIR))

        device_options = ["cpu", "cuda"] if DEFAULT_DEVICE == "cuda" else ["cpu"]
        device = st.selectbox("Device", device_options, index=0)

        try:
            (
                root_paths,
                selected_config,
                selected_paths,
                selected_checkpoint,
                speaker_index_payload,
            ) = _load_final_artifacts(work_dir_str, device=device)
        except Exception as e:
            st.error("Could not load the final operational model.")
            st.exception(e)
            st.stop()

        st.write("Operational mode: prototype")
        st.write("Experiment config:")
        st.code(FINAL_EXPERIMENT_NAME)
        st.write("Checkpoint:")
        st.code(str(selected_checkpoint))
        st.write(f"Data signature: {selected_paths.data_signature}")
        st.write(f"Speaker index size: {len(speaker_index_payload['prototypes'])}")
        st.write(f"Similarity threshold: {speaker_index_payload['similarity_threshold']:.4f}")

    tab1, tab2, tab3 = st.tabs(
        ["Access check", "Enroll new allowed speaker", "Project status"]
    )

    with tab1:
        st.subheader("Test one recording")
        st.info(
            "This app intentionally uses only the final prototype mode, "
            "so enrollment and inference stay logically consistent."
        )

        input_mode = st.radio("Audio source", ["Record", "Upload wav"], horizontal=True)

        recorded_wav = None
        uploaded_wav = None

        if input_mode == "Record":
            recorded_wav = st.audio_input("Record voice", sample_rate=16000, key="predict_mic")
            if recorded_wav is not None:
                st.audio(recorded_wav)
        else:
            uploaded_wav = st.file_uploader(
                "Upload a single .wav file",
                type=["wav"],
                key="predict_upload",
            )
            if uploaded_wav is not None:
                st.audio(uploaded_wav)

        if st.button("Run access check"):
            wav_path, source_name = _resolve_audio_input(recorded_wav, uploaded_wav)

            if wav_path is None:
                st.warning("Provide audio first.")
            else:
                try:
                    pred = predict_wav_file_operational(
                        wav_path=wav_path,
                        checkpoint_path=selected_checkpoint,
                        speaker_index_path=selected_paths.speaker_index_path,
                        device=device,
                    )

                    c1, c2, c3, c4, c5 = st.columns(5)
                    c1.metric("Class", pred["predicted_class_name"])
                    c2.metric("Top speaker", pred["best_speaker"] or "-")
                    c3.metric("Similarity", f"{pred['best_similarity']:.4f}")
                    c4.metric("Threshold", f"{pred['threshold']:.4f}")
                    c5.metric("Margin", f"{pred['decision_margin']:.4f}")

                    st.caption(
                        f"Input source: {source_name}; "
                        f"segments used: {pred['n_segments']}"
                    )

                    sim_df = pd.DataFrame(
                        [
                            {"speaker": k, "similarity": v}
                            for k, v in pred["all_similarities"].items()
                        ]
                    ).sort_values("similarity", ascending=False)

                    if not sim_df.empty:
                        st.dataframe(sim_df, use_container_width=True)

                    if pred["predicted_class_name"] == "allow":
                        st.success("ACCESS GRANTED")
                    else:
                        st.error("ACCESS DENIED")

                except Exception as e:
                    st.exception(e)

    with tab2:
        st.subheader("Add new allowed speaker")
        st.write(
            "Record or upload several .wav samples for one new person. "
            "The app refreshes the prototype index using the fixed final checkpoint."
        )

        if "enroll_recorded_samples" not in st.session_state:
            st.session_state.enroll_recorded_samples = []

        speaker_name = st.text_input("Speaker name", key="speaker_name")

        st.markdown("#### Record samples")
        recorded_enroll_wav = st.audio_input(
            "Record a sample for the new speaker",
            sample_rate=16000,
            key="enroll_mic",
        )

        c_add, c_clear = st.columns(2)

        with c_add:
            if st.button("Add recorded sample", disabled=(recorded_enroll_wav is None)):
                st.session_state.enroll_recorded_samples.append(
                    bytes(recorded_enroll_wav.getbuffer())
                )
                st.success(
                    "Recorded sample added. "
                    f"Total recorded samples: {len(st.session_state.enroll_recorded_samples)}"
                )

        with c_clear:
            if st.button("Clear recorded samples"):
                st.session_state.enroll_recorded_samples = []
                st.info("Recorded enrollment samples cleared.")

        n_recorded = len(st.session_state.enroll_recorded_samples)
        st.write(f"Recorded samples ready for enrollment: {n_recorded}")

        st.markdown("#### Or upload samples")
        uploaded_speaker_wavs = st.file_uploader(
            "Upload several .wav files for the new person",
            type=["wav"],
            accept_multiple_files=True,
            key="speaker_wavs",
        )

        n_uploaded = len(uploaded_speaker_wavs or [])
        st.write(f"Uploaded samples ready for enrollment: {n_uploaded}")

        has_any_samples = n_recorded > 0 or n_uploaded > 0

        if st.button("Enroll speaker", disabled=(not speaker_name or not has_any_samples)):
            try:
                speaker_dir = save_enrollment_audio_to_speaker_dir(
                    root_paths,
                    speaker_name,
                    uploaded_files=uploaded_speaker_wavs,
                    recorded_samples=st.session_state.enroll_recorded_samples,
                )

                selected_config.train.device = device

                result = enroll_new_allowed_speaker_operational(
                    base_config=selected_config,
                    checkpoint_path=selected_checkpoint,
                    new_speaker_dir=speaker_dir,
                    speaker_name=speaker_name,
                    overwrite_source=True,
                )

                st.session_state.enroll_recorded_samples = []

                st.success(f"Added speaker: {result['new_speaker_name']}")
                st.json(result)
                st.rerun()

            except Exception as e:
                st.exception(e)

    with tab3:
        st.subheader("Project status")

        st.write("Final experiment config:")
        st.code(FINAL_EXPERIMENT_NAME)

        st.write("Final checkpoint:")
        st.code(str(selected_checkpoint))

        if selected_paths.allow_speakers_path.exists():
            st.write("Allowed speakers:")
            with open(selected_paths.allow_speakers_path, "r", encoding="utf-8") as f:
                st.json(json.load(f))

        dup_df = _safe_read_csv(selected_paths.duplicate_meta_path)
        st.write(f"Duplicates removed during preprocessing: {len(dup_df)}")

        if not dup_df.empty:
            st.dataframe(dup_df.head(20), use_container_width=True)

        st.write("Data cache dir:")
        st.code(str(selected_paths.data_cache_dir))

        st.write("Speaker index path:")
        st.code(str(selected_paths.speaker_index_path))


if __name__ == "__main__":
    main()
