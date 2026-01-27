import asyncio
import hashlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def compute_audio_hash(file_path: Path) -> str:
    md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5.update(chunk)
    return md5.hexdigest()


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
        lyrics_hash_value = hash_lyrics(job.lyrics)

        download_result = await asyncio.get_event_loop().run_in_executor(
            None, _download_and_hash, job.video_id
        )
        audio_hash = download_result["audio_hash"]
        audio_path = download_result["audio_path"]

        async with get_session() as session:
            sync_repo = SyncRepository(session)
            existing = await sync_repo.get_by_audio_and_lyrics_hash(audio_hash, lyrics_hash_value)
            if existing:
                job_repo = JobRepository(session)
                await job_repo.update_status(
                    job_id, "completed", progress=100, result_id=existing.id
                )
                logger.info(f"Job {job_id} reused existing sync (audio_hash match)")
                Path(audio_path).unlink(missing_ok=True)
                return

        result = await asyncio.get_event_loop().run_in_executor(
            None, _run_alignment, audio_path, job.lyrics, job.language
        )

        async with get_session() as session:
            job_repo = JobRepository(session)
            sync_repo = SyncRepository(session)

            sync_result = await sync_repo.create(
                video_id=job.video_id,
                lyrics_hash=lyrics_hash_value,
                timestamps=result["timestamps"],
                language=result.get("language"),
                engine="ctc",
                quality_score=result.get("quality_score"),
                audio_hash=audio_hash,
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


def _download_and_hash(video_id: str) -> dict:
    from everyric2.audio.downloader import YouTubeDownloader

    downloader = YouTubeDownloader()
    youtube_url = f"https://www.youtube.com/watch?v={video_id}"
    dl_result = downloader.download(youtube_url)
    audio_hash = compute_audio_hash(dl_result.audio_path)

    return {
        "audio_path": str(dl_result.audio_path),
        "audio_hash": audio_hash,
    }


def _run_alignment(audio_path: str, lyrics: str, language: str | None) -> dict:
    from everyric2.alignment.factory import EngineFactory
    from everyric2.audio.loader import AudioLoader
    from everyric2.config.settings import get_settings
    from everyric2.inference.prompt import LyricLine

    settings = get_settings()
    loader = AudioLoader()
    audio_path_obj = Path(audio_path)

    try:
        audio = loader.load(audio_path_obj)
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
                    {"word": w.word, "start": w.start, "end": w.end, "confidence": w.confidence}
                    for w in r.word_segments
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
        audio_path_obj.unlink(missing_ok=True)
