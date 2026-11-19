"""Medición de tiempos por fase para el RAG (búsqueda e indexación).

    crono = Cronometro("RAG búsqueda")
    with crono.fase("embedding"):
        ...
    crono.imprimir("hibrida/rrf")
    ->  ⏱ [RAG búsqueda] embedding 640 ms · milvus 158 ms · total 806 ms (hibrida/rrf)
"""
import time
from contextlib import contextmanager


class Cronometro:
    def __init__(self, nombre: str):
        self.nombre = nombre
        self.fases: dict[str, float] = {}
        self._inicio = time.perf_counter()

    @contextmanager
    def fase(self, etiqueta: str):
        t = time.perf_counter()
        try:
            yield
        finally:
            self.fases[etiqueta] = self.fases.get(etiqueta, 0.0) + time.perf_counter() - t

    @property
    def total(self) -> float:
        return time.perf_counter() - self._inicio

    def resumen(self) -> dict[str, float]:
        """Segundos por fase y total (para devolverlos en una respuesta JSON)."""
        return {**{k: round(v, 3) for k, v in self.fases.items()}, "total": round(self.total, 3)}

    def imprimir(self, detalle: str = "") -> None:
        fases = " · ".join(f"{k} {v * 1000:.0f} ms" for k, v in self.fases.items())
        extra = f" ({detalle})" if detalle else ""
        print(f"   ⏱ [{self.nombre}] {fases}{' · ' if fases else ''}total {self.total * 1000:.0f} ms{extra}", flush=True)