from sentence_transformers import SentenceTransformer
import threading
import time

import torch

import os
# Forzar a HuggingFace a no buscar actualizaciones en internet para ir más rápido
os.environ["HF_HUB_OFFLINE"] = "1"


class RecipeEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-m3"):
        # Guardamos el nombre y el dispositivo, pero NO cargamos el modelo aún
        self.model_name = model_name
        self.model = None
        self._bloqueo = threading.Lock()    # la precarga del arranque y una petición no lo cargan dos veces
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"RecipeEmbedder inicializado (El modelo {model_name} se cargará bajo demanda).")

    def _cargar_modelo(self):
        """Carga el modelo solo si no está en memoria (en fp16 si hay GPU: la mitad de VRAM)."""
        if self.model is not None:
            return
        with self._bloqueo:
            if self.model is not None:          # otro hilo lo cargó mientras se esperaba
                return
            t0 = time.perf_counter()
            print(f"\n⏳ Cargando modelo de embeddings ({self.model_name})...", flush=True)
            model_kwargs = {"torch_dtype": torch.float16} if self.device == "cuda" else {}
            self.model = SentenceTransformer(self.model_name, device=self.device,
                                             local_files_only=True, model_kwargs=model_kwargs)
            print(f"✅ Modelo de embeddings cargado en {self.device.upper()} "
                  f"({time.perf_counter() - t0:.1f} s)\n", flush=True)

    def precargar(self):
        """Carga el modelo y calcula un embedding de prueba (el primero siempre es más lento)."""
        self._cargar_modelo()
        self.model.encode("precarga", normalize_embeddings=True)

    def get_embedding(self, text: str) -> list[float]:
        """Convierte un texto (receta o ingredientes) en un vector denso."""
        self._cargar_modelo()
        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def get_embeddings(self, texts: list[str], batch_size: int = 16) -> list[list[float]]:
        """Vectoriza varios textos de una vez (mucho más rápido que uno a uno al indexar)."""
        self._cargar_modelo()
        embeddings = self.model.encode(texts, normalize_embeddings=True, batch_size=batch_size)
        return embeddings.tolist()


# Instancia única compartida por todo el backend (main, rag_node, indexer):
# el modelo solo se carga una vez en memoria.
embedder = RecipeEmbedder()