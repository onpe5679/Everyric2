import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Literal

from everyric2.alignment.base import (
    AlignmentError,
    BaseAlignmentEngine,
    EngineNotAvailableError,
    TranscriptionResult,
    WordTimestamp,
)
from everyric2.audio.loader import AudioData, AudioLoader
from everyric2.config.settings import AlignmentSettings
from everyric2.inference.prompt import LyricLine, SyncResult


class MFAEngine(BaseAlignmentEngine):
    MFA_MODELS = {
        "en": {"acoustic": "english_mfa", "dictionary": "english_us_arpa"},
        "ko": {"acoustic": "korean_mfa", "dictionary": "korean_mfa"},
        "ja": {"acoustic": "japanese_mfa", "dictionary": "japanese_mfa"},
    }

    def __init__(self, config: AlignmentSettings | None = None):
        super().__init__(config)
        self._mfa_bin = None
        self._model_dir = None
        self._temp_workdir = None
        self.audio_loader = AudioLoader()
        self._last_matcher = None
        self._last_match_stats = None
        self._mfa_stage = ""

    def _resolve_mfa_bin(self) -> str | None:
        if self._mfa_bin:
            return self._mfa_bin
        # env override
        override = os.getenv("EVERYRIC_ALIGNMENT__MFA_BIN")
        if override and Path(override).exists():
            self._mfa_bin = override
            return self._mfa_bin
        # common conda path
        default_conda = Path.home() / ".conda/envs/mfaenv/bin/mfa"
        if default_conda.exists():
            self._mfa_bin = str(default_conda)
            return self._mfa_bin
        found = shutil.which("mfa")
        if found:
            self._mfa_bin = found
        return self._mfa_bin

    def _mfa_env(self) -> dict | None:
        # ensure PATH has mfa bin
        mfa_bin = self._resolve_mfa_bin()
        if not mfa_bin:
            return None
        env = os.environ.copy()
        bin_dir = str(Path(mfa_bin).parent)
        env_path = env.get("PATH", "")
        if bin_dir not in env_path:
            env["PATH"] = f"{bin_dir}:{env_path}" if env_path else bin_dir
        return env

    def is_available(self) -> bool:
        return self._resolve_mfa_bin() is not None

    def _get_cpu_count(self) -> int:
        try:
            return os.cpu_count() or 4
        except Exception:
            return 4

    def _check_models_installed(self, language: str) -> tuple[bool, list[str]]:
        if language not in self.MFA_MODELS:
            return False, [f"Language '{language}' not supported"]

        missing = []
        models = self.MFA_MODELS[language]

        try:
            mfa_bin = self._resolve_mfa_bin()
            if not mfa_bin:
                return False, ["MFA binary not found"]

            result = subprocess.run(
                [mfa_bin, "model", "list", "acoustic"],
                capture_output=True,
                text=True,
                timeout=30,
                env=self._mfa_env(),
            )
            if models["acoustic"] not in result.stdout:
                missing.append(f"acoustic model: {models['acoustic']}")

            result = subprocess.run(
                [mfa_bin, "model", "list", "dictionary"],
                capture_output=True,
                text=True,
                timeout=30,
                env=self._mfa_env(),
            )
            if models["dictionary"] not in result.stdout:
                missing.append(f"dictionary: {models['dictionary']}")

        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False, ["MFA command failed"]

        return len(missing) == 0, missing

    def download_models(self, language: str) -> None:
        if language not in self.MFA_MODELS:
            raise AlignmentError(f"Language '{language}' not supported by MFA")

        models = self.MFA_MODELS[language]

        subprocess.run(
            ["mfa", "model", "download", "acoustic", models["acoustic"]],
            check=True,
            timeout=300,
        )
        subprocess.run(
            ["mfa", "model", "download", "dictionary", models["dictionary"]],
            check=True,
            timeout=300,
        )

    def transcribe(
        self,
        audio: AudioData,
        language: str | None = None,
    ) -> TranscriptionResult:
        raise NotImplementedError(
            "MFA does not support transcription. Use WhisperX for transcription."
        )

    def align(
        self,
        audio: AudioData,
        lyrics: list[LyricLine],
        language: str | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[SyncResult]:
        if not self.is_available():
            raise EngineNotAvailableError(
                "MFA not installed. Install with: conda install -c conda-forge montreal-forced-aligner"
            )

        resolved_lang = self._resolve_language(language)

        models_ok, missing = self._check_models_installed(resolved_lang)
        if not models_ok:
            raise AlignmentError(
                f"MFA models not installed for {resolved_lang}: {missing}. "
                f"Run: mfa model download acoustic {self.MFA_MODELS[resolved_lang]['acoustic']}"
            )

        if progress_callback:
            progress_callback(1, 4)

        with tempfile.TemporaryDirectory() as tmpdir:
            corpus_dir = Path(tmpdir) / "corpus"
            output_dir = Path(tmpdir) / "output"
            corpus_dir.mkdir()
            output_dir.mkdir()

            prepared = self.audio_loader.prepare_for_alignment(
                audio,
                target_sr=16000,
                normalize=True,
            )

            audio_path = corpus_dir / "audio.wav"
            prepared.to_file(audio_path)

            transcript_path = corpus_dir / "audio.lab"
            transcript_text = "\n".join(line.text for line in lyrics)
            transcript_path.write_text(transcript_text, encoding="utf-8")

            if progress_callback:
                progress_callback(2, 4)

            mfa_bin = self._resolve_mfa_bin()
            if not mfa_bin:
                raise AlignmentError("MFA binary not found")

            models = self.MFA_MODELS[resolved_lang]
            num_jobs = (
                self.config.mfa_num_jobs if self.config.mfa_num_jobs > 0 else self._get_cpu_count()
            )
            cmd = [
                mfa_bin,
                "align",
                str(corpus_dir),
                models["dictionary"],
                models["acoustic"],
                str(output_dir),
                "--clean",
                "--overwrite",
                "--beam",
                str(self.config.mfa_beam),
                "--retry_beam",
                str(self.config.mfa_retry_beam),
                "--num_jobs",
                str(num_jobs),
            ]
            if getattr(self.config, "mfa_single_speaker", True):
                cmd.append("--single_speaker")

            mfa_stages = [
                ("Generating MFCCs", "mfcc"),
                ("Generating final features", "features"),
                ("first-pass alignment", "align1"),
                ("Generating alignments", "align2"),
                ("Exporting", "export"),
            ]

            try:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=self._mfa_env(),
                )

                stderr_lines = []
                current_stage = ""
                while True:
                    line = process.stderr.readline()
                    if not line and process.poll() is not None:
                        break
                    if line:
                        stderr_lines.append(line)
                        for stage_text, stage_name in mfa_stages:
                            if stage_text in line and stage_name != current_stage:
                                current_stage = stage_name
                                if progress_callback:
                                    self._mfa_stage = stage_name
                                    progress_callback(2, 4)
                                break

                returncode = process.wait(timeout=600)
                if returncode != 0:
                    raise AlignmentError(f"MFA alignment failed: {''.join(stderr_lines)}")
            except subprocess.TimeoutExpired:
                process.kill()
                raise AlignmentError("MFA alignment timed out (>10 minutes)")

            if progress_callback:
                progress_callback(3, 4)

            textgrid_path = output_dir / "audio.TextGrid"
            if not textgrid_path.exists():
                textgrid_path = output_dir / "corpus" / "audio.TextGrid"

            if not textgrid_path.exists():
                for path in output_dir.rglob("*.TextGrid"):
                    textgrid_path = path
                    break

            if not textgrid_path.exists():
                raise AlignmentError("MFA output TextGrid not found")

            results = self._parse_textgrid(textgrid_path, lyrics)

            if progress_callback:
                progress_callback(4, 4)

            return results

    def _parse_textgrid(
        self,
        textgrid_path: Path,
        lyrics: list[LyricLine],
    ) -> list[SyncResult]:
        try:
            from praatio import textgrid
        except ImportError:
            raise AlignmentError("praatio not installed. Install with: pip install praatio")

        tg = textgrid.openTextgrid(str(textgrid_path), includeEmptyIntervals=False)

        word_tier = None
        for tier_name in ["words", "word", "Words", "Word"]:
            if tier_name in tg.tierNames:
                word_tier = tg.getTier(tier_name)
                break

        if word_tier is None and len(tg.tierNames) > 0:
            word_tier = tg.getTier(tg.tierNames[0])

        if word_tier is None:
            raise AlignmentError("No word tier found in TextGrid")

        words = []
        for entry in word_tier.entries:
            if entry.label and entry.label.strip():
                words.append(
                    WordTimestamp(
                        word=entry.label,
                        start=entry.start,
                        end=entry.end,
                    )
                )

        from everyric2.alignment.matcher import LyricsMatcher

        matcher = LyricsMatcher()
        results = matcher.match_lyrics_to_words(lyrics, words, "")
        self._last_matcher = matcher
        self._last_match_stats = matcher.last_match_stats
        return results

    def get_last_transcription_data(self):
        if getattr(self, "_last_matcher", None) is None:
            return None, None, None
        return (
            self._last_matcher.last_transcription_words,
            self._last_match_stats,
            "mfa",
        )

    def get_transcription_sets(self):
        words, stats, engine = self.get_last_transcription_data()
        if words:
            return [(words, stats, engine)]
        return []

    @staticmethod
    def get_engine_type() -> Literal["whisperx", "mfa", "hybrid", "qwen"]:
        return "mfa"
