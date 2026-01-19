import asyncio
import logging
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


async def process_job(job_id: str) -> None:
    from everyric2.server.db.connection import get_session
    from everyric2.server.db.repository import JobRepository, SyncRepository, hash_lyrics

    async with get_session() as session:
        job_repo = JobRepository(session)
        job = await job_repo.get_by_id(job_id)

        if not job:
            logger.error(f"Job not found: {job_id}")
            return

        await job_repo.update_status(job_id, "processing", progress=10)

    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None, _sync_process, job.video_id, job.lyrics, job.language
        )

        async with get_session() as session:
            job_repo = JobRepository(session)
            sync_repo = SyncRepository(session)

            lyrics_hash_value = hash_lyrics(job.lyrics)
            sync_result = await sync_repo.create(
                video_id=job.video_id,
                lyrics_hash=lyrics_hash_value,
                timestamps=result["timestamps"],
                language=result.get("language"),
                engine="ctc",
                quality_score=result.get("quality_score"),
            )

            await job_repo.update_status(
                job_id, "completed", progress=100, result_id=sync_result.id
            )
            logger.info(f"Job completed: {job_id}")

    except Exception as e:
        logger.exception(f"Job failed: {job_id}")
        async with get_session() as session:
            job_repo = JobRepository(session)
            await job_repo.update_status(job_id, "failed", error=str(e))


def _sync_process(video_id: str, lyrics: str, language: str | None) -> dict:
    from everyric2.alignment.factory import EngineFactory
    from everyric2.audio.downloader import YouTubeDownloader
    from everyric2.audio.loader import AudioLoader
    from everyric2.config.settings import get_settings
    from everyric2.inference.prompt import LyricLine

    settings = get_settings()
    downloader = YouTubeDownloader()
    loader = AudioLoader()

    youtube_url = f"https://www.youtube.com/watch?v={video_id}"
    dl_result = downloader.download(youtube_url)

    try:
        audio = loader.load(dl_result.audio_path)
        lyric_lines = LyricLine.from_text(lyrics)

        engine = EngineFactory.get_engine("ctc", settings.alignment)
        if not engine.is_available():
            raise RuntimeError("CTC engine not available")

        results = engine.align(audio, lyric_lines, language=language or "auto")

        timestamps = []
        for r in results:
            seg = {
                "text": r.text,
                "start": r.start_time,
                "end": r.end_time,
            }
            if r.confidence is not None:
                seg["confidence"] = r.confidence
            if r.word_segments:
                seg["words"] = [
                    {"word": w.word, "start": w.start, "end": w.end} for w in r.word_segments
                ]
            timestamps.append(seg)

        avg_confidence = None
        confidences = [t.get("confidence") for t in timestamps if t.get("confidence") is not None]
        if confidences:
            avg_confidence = sum(confidences) / len(confidences)

        detected_lang = language
        if hasattr(engine, "_current_lang"):
            detected_lang = engine._current_lang

        return {
            "timestamps": timestamps,
            "language": detected_lang,
            "quality_score": avg_confidence,
        }
    finally:
        if dl_result.audio_path.exists():
            dl_result.audio_path.unlink()
