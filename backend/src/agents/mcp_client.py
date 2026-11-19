"""Cliente MCP de CookingAura.

Una única sesión MCP compartida por todo el backend: el servidor de herramientas (mcp_tool.py) se arranca
una vez al iniciar FastAPI y todas las peticiones (nodo `mcp` del grafo y agente del chat) reutilizan
sus herramientas. Así se evita lanzar un subproceso de Python en cada mensaje.

- `gestor_mcp.iniciar()` / `gestor_mcp.detener()` se llaman desde el `lifespan` de FastAPI.
- `herramientas_mcp()` da las herramientas de la sesión compartida; si no está activa (scripts sueltos,
  arranque fallido...), abre una sesión temporal solo para esa petición.
- Una tarea supervisora es la dueña de la sesión: la abre, la cierra y la vuelve a abrir si el servidor
  se cae (abrir y cerrar en la misma tarea es un requisito de anyio, que usa el SDK de MCP).
"""
import asyncio
import os
import sys
from contextlib import AsyncExitStack, asynccontextmanager

RUTA_SERVIDOR = os.path.join(os.path.dirname(__file__), "mcp_tool.py")
ESPERA_REINTENTO = 30          # segundos entre intentos si el servidor no arranca


async def _abrir_sesion(pila: AsyncExitStack) -> list:
    """Arranca el servidor MCP como subproceso dentro de `pila` y devuelve sus herramientas de LangChain."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from langchain_mcp_adapters.tools import load_mcp_tools

    parametros = StdioServerParameters(
        command=sys.executable,
        args=[RUTA_SERVIDOR],
        env=dict(os.environ),   # sin esto el subproceso no ve MILVUS_HOST ni el resto del .env
    )
    read, write = await pila.enter_async_context(stdio_client(parametros))
    sesion = await pila.enter_async_context(ClientSession(read, write))
    await sesion.initialize()
    return await load_mcp_tools(sesion)


class ErrorArranqueMCP(RuntimeError):
    """El servidor de herramientas MCP no ha podido arrancar."""


def _error_arranque(e: Exception) -> ErrorArranqueMCP:
    # "Connection closed" casi siempre significa que mcp_tool.py se ha cerrado al arrancar (p. ej. un ImportError);
    # su traza aparece justo encima en los logs del backend.
    return ErrorArranqueMCP("El servidor de herramientas MCP (src/agents/mcp_tool.py) no ha arrancado. "
                        "Revisa la traza que aparece justo encima en los logs del backend. "
                        f"Para verlo directamente: python src/agents/mcp_tool.py ({type(e).__name__}: {e})")


class GestorMCP:
    """Mantiene viva una sesión MCP compartida y la reabre si se cae."""

    def __init__(self):
        self.herramientas: list = []
        self.activo = False
        self._tarea: asyncio.Task | None = None
        self._listo = asyncio.Event()
        self._cambio = asyncio.Event()       # pide cerrar la sesión actual (reinicio o parada)
        self._parar = asyncio.Event()

    async def _supervisar(self):
        while not self._parar.is_set():
            try:
                async with AsyncExitStack() as pila:
                    self.herramientas = await _abrir_sesion(pila)
                    self.activo = True
                    self._listo.set()
                    nombres = ", ".join(sorted(h.name for h in self.herramientas))
                    print(f"✅ MCP: sesión compartida lista ({len(self.herramientas)} herramientas: {nombres})",
                          flush=True)
                    await self._cambio.wait()        # hasta que se pida reiniciar o parar
                    self._cambio.clear()
            except Exception as e:
                print(f"⚠️  {_error_arranque(e)}\n   Se reintentará en {ESPERA_REINTENTO} s; mientras tanto, "
                      "cada petición abrirá su propia sesión.", flush=True)
                self._listo.set()
                try:
                    await asyncio.wait_for(self._parar.wait(), timeout=ESPERA_REINTENTO)
                except asyncio.TimeoutError:
                    pass
            finally:
                self.activo = False
                self.herramientas = []

    async def iniciar(self, espera: float = 60):
        """Arranca la sesión compartida (desde el lifespan de FastAPI)."""
        self._tarea = asyncio.create_task(self._supervisar(), name="mcp-supervisor")
        try:
            await asyncio.wait_for(self._listo.wait(), timeout=espera)
        except asyncio.TimeoutError:
            print("⚠️  MCP: la sesión compartida tarda en arrancar; se seguirá intentando en segundo plano.",
                  flush=True)

    async def detener(self):
        self._parar.set()
        self._cambio.set()
        if self._tarea:
            try:
                await asyncio.wait_for(self._tarea, timeout=10)
            except (asyncio.TimeoutError, Exception):
                self._tarea.cancel()

    def notificar_error(self, error: Exception) -> bool:
        """Si un fallo indica que el servidor se ha caído, pide reabrir la sesión. Devuelve True en ese caso."""
        texto = f"{type(error).__name__}: {error}"
        if not any(s in texto for s in ("Connection closed", "ClosedResource", "BrokenResource", "EndOfStream")):
            return False
        if self.activo:
            print(f"♻️  MCP: la sesión compartida se ha caído ({texto}); se reabre.", flush=True)
            self.activo = False
            self._cambio.set()
        return self._tarea is not None and not self._parar.is_set()

    async def esperar_activo(self, espera: float = 15) -> bool:
        """Espera a que la sesión compartida vuelva a estar disponible."""
        limite = asyncio.get_running_loop().time() + espera
        while asyncio.get_running_loop().time() < limite:
            if self.activo and self.herramientas:
                return True
            await asyncio.sleep(0.2)
        return False


gestor_mcp = GestorMCP()


@asynccontextmanager
async def herramientas_mcp():
    """Herramientas MCP para una petición: las de la sesión compartida o, si no está activa, una temporal."""
    if gestor_mcp.activo and gestor_mcp.herramientas:
        try:
            yield gestor_mcp.herramientas
        except Exception as e:
            gestor_mcp.notificar_error(e)
            raise
        return

    pila = AsyncExitStack()
    try:
        herramientas = await _abrir_sesion(pila)
    except Exception as e:
        try:
            await pila.aclose()
        except Exception:
            pass
        raise _error_arranque(e) from e
    try:
        yield herramientas
    finally:
        await pila.aclose()


async def setup_mcp_tools():
    """Compatibilidad con el código anterior: abre una sesión temporal. Devuelve (herramientas, pila_a_cerrar)."""
    pila = AsyncExitStack()
    try:
        return await _abrir_sesion(pila), pila
    except Exception as e:
        try:
            await pila.aclose()
        except Exception:
            pass
        raise _error_arranque(e) from e


def texto_resultado_mcp(resultado) -> str:
    """Extrae el texto de lo que devuelve una herramienta MCP, sea cual sea su forma."""
    if isinstance(resultado, str):
        return resultado.strip()
    if isinstance(resultado, tuple) and resultado:
        return texto_resultado_mcp(resultado[0])
    if isinstance(resultado, list):
        partes = []
        for item in resultado:
            if hasattr(item, "text"):
                partes.append(item.text)
            elif isinstance(item, dict) and "text" in item:
                partes.append(str(item["text"]))
            elif isinstance(item, str):
                partes.append(item)
        return " ".join(partes).strip()
    if hasattr(resultado, "content"):
        return texto_resultado_mcp(resultado.content)
    return str(resultado).strip()