"""Multi-variant subtitle output generator.

Reconstructed to match everyric2.cli usage:
    gen = MultiOutputGenerator(format, mode)
    outputs = gen.generate_all_variants(results, translation_result, output_dir,
                                        base_name, line_results=line_results)
    for out in outputs: out.variant / out.path
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from everyric2.inference.prompt import SyncResult
from everyric2.output.formatters import BaseFormatter, FormatterFactory


@dataclass
class OutputVariant:
    variant: str  # "original" | "translated" | "pronunciation" | "full"
    path: Path


class MultiOutputGenerator:
    def __init__(self, format: str, mode: str = "line"):
        self.format = format
        self.mode = mode

    def _formatter(self, *, translation: bool = False,
                   pronunciation: bool = False) -> BaseFormatter:
        f = FormatterFactory.get_formatter(self.format)
        f.include_translation = translation
        f.include_pronunciation = pronunciation
        return f

    def _ext(self) -> str:
        return FormatterFactory.get_formatter(self.format).get_extension()

    def generate_all_variants(
        self,
        results: list[SyncResult],
        translation_result=None,
        output_dir: str | Path = ".",
        base_name: str = "output",
        line_results: list[SyncResult] | None = None,
    ) -> list[OutputVariant]:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ext = self._ext()
        variants: list[OutputVariant] = []

        # original
        p = out_dir / f"{base_name}.{ext}"
        p.write_text(self._formatter().format(results), encoding="utf-8")
        variants.append(OutputVariant("original", p))

        has_tr = any(getattr(r, "translation", None) for r in results)
        has_pr = any(getattr(r, "pronunciation", None) for r in results)

        if has_tr:
            tr_results = [
                SyncResult(
                    text=(r.translation or r.text),
                    start_time=r.start_time,
                    end_time=r.end_time,
                    line_number=r.line_number,
                )
                for r in results
            ]
            p = out_dir / f"{base_name}_translated.{ext}"
            p.write_text(self._formatter().format(tr_results), encoding="utf-8")
            variants.append(OutputVariant("translated", p))

        if has_pr:
            p = out_dir / f"{base_name}_pronunciation.{ext}"
            p.write_text(
                self._formatter(pronunciation=True).format(results), encoding="utf-8"
            )
            variants.append(OutputVariant("pronunciation", p))

        if has_tr or has_pr:
            p = out_dir / f"{base_name}_full.{ext}"
            p.write_text(
                self._formatter(translation=True, pronunciation=True).format(results),
                encoding="utf-8",
            )
            variants.append(OutputVariant("full", p))

        return variants
