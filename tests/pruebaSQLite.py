import sqlite3
import os

DB_NAME = os.path.join(os.path.dirname(__file__), "..", "data", "ingredients.db")

def comprobar_base_de_datos():
    if not os.path.exists(DB_NAME):
        print(f"❌ ERROR: El archivo '{DB_NAME}' no existe en esta ruta.")
        print(f"Ruta actual desde donde ejecutas: {os.getcwd()}")
        return

    try:
        conexion = sqlite3.connect(DB_NAME)
        cursor = conexion.cursor()

        # 1. Comprobar qué tablas existen realmente
        print(f"Conectado a '{DB_NAME}' con éxito.\n")
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tablas = cursor.fetchall()
        
        print("Tablas encontradas en la base de datos:")
        nombres_tablas = [t[0] for t in tablas]
        for nombre in nombres_tablas:
            print(f" 📂 {nombre}")
        print("-" * 50)

        # 2. Consultar la tabla de ingredientes (ajusta el nombre si es en inglés)
        tabla_objetivo = "ingredientes" if "ingredientes" in nombres_tablas else "ingredients"
        
        if tabla_objetivo not in nombres_tablas:
            print(f"❌ ERROR: No se encontró ninguna tabla llamada 'ingredientes' o 'ingredients'.")
            return

        print(f"Consultando los registros de la tabla '{tabla_objetivo}'...\n")
        cursor.execute(f"SELECT * FROM {tabla_objetivo}")
        filas = cursor.fetchall()

        if not filas:
            print(f"⚠️ La tabla '{tabla_objetivo}' está VACÍA.")
            print("Conclusión: El 'commit' de tu backend no está funcionando o estás escribiendo en otra base de datos.")
        else:
            # Extraer nombres de las columnas para formatear la salida
            columnas = [descripcion[0] for descripcion in cursor.description]
            print(f"✅ Se encontraron {len(filas)} registros:\n")
            print(f"{columnas}")
            for fila in filas:
                print(fila)
                
            print("\nConclusión: Si ves los datos aquí, el fallo no está al guardar, sino en el endpoint GET /ingredientes de tu backend que no los está leyendo o mapeando bien.")

    except sqlite3.Error as e:
        print(f"❌ Error de SQLite: {e}")
    finally:
        if 'conexion' in locals() and conexion:
            conexion.close()

def eliminar_base_de_datos():
    """Elimina la base de datos SQLite si existe."""
    if os.path.exists(DB_NAME):
        os.remove(DB_NAME)
        print(f"✅ Base de datos '{DB_NAME}' eliminada con éxito.")
    else:
        print(f"⚠️ No se encontró la base de datos '{DB_NAME}' para eliminar.")

def eliminar_tabla(tabla_nombre):
    """Elimina una tabla específica de la base de datos SQLite."""
    if not os.path.exists(DB_NAME):
        print(f"❌ ERROR: El archivo '{DB_NAME}' no existe en esta ruta.")
        return

    try:
        conexion = sqlite3.connect(DB_NAME)
        cursor = conexion.cursor()
        
        cursor.execute(f"DROP TABLE IF EXISTS {tabla_nombre}")
        conexion.commit()
        print(f"✅ Tabla '{tabla_nombre}' eliminada con éxito de la base de datos '{DB_NAME}'.")
    except sqlite3.Error as e:
        print(f"❌ Error al eliminar la tabla '{tabla_nombre}': {e}")
    finally:
        if 'conexion' in locals() and conexion:
            conexion.close()

def insertar_multiples_ingredientes(tabla_nombre):
    if not os.path.exists(DB_NAME):
        print(f"❌ ERROR: El archivo '{DB_NAME}' no existe en esta ruta.")
        return
    
    try:
        conexion = sqlite3.connect(DB_NAME)
        cursor = conexion.cursor()
        
        # Usamos f y comillas triples (""") para el string multilínea y la variable tabla_nombre
        consulta = f"""INSERT INTO {tabla_nombre} (name, quantity, category, unit) VALUES
            ('Aceite de girasol', 1, 'Aceites y Aliños', 'botella'),
            ('Aceite de oliva', 1, 'Aceites y Aliños', 'botella'),
            ('Sal', 1, 'Aceites y Aliños', 'paquete'),
            ('Vinagre de Módena', 1, 'Aceites y Aliños', 'botella'),
            ('Agua', 1, 'Bebidas', 'garrafa'),
            ('Aquarius', 1, 'Bebidas', 'botella'),
            ('Café', 1, 'Bebidas', 'paquete'),
            ('Gazpacho', 1, 'Bebidas', 'brik'),
            ('Infusiones para dormir (Tila/Valeriana)', 1, 'Bebidas', 'caja'),
            ('Nestea', 1, 'Bebidas', 'botella'),
            ('Vino blanco', 1, 'Bebidas', 'botella'),
            ('Alitas', 1, 'Carnes y Aves', 'bandeja'),
            ('Bacon', 1, 'Carnes y Aves', 'paquete'),
            ('Carne (Filetes)', 1, 'Carnes y Aves', 'bandeja'),
            ('Hamburguesas', 1, 'Carnes y Aves', 'paquete'),
            ('Pollo', 1, 'Carnes y Aves', 'bandeja'),
            ('Croquetas', 1, 'Congelados y Platos Preparados', 'bolsa'),
            ('Ensalada hecha', 1, 'Congelados y Platos Preparados', 'ud'),
            ('Flautas', 1, 'Congelados y Platos Preparados', 'paquete'),
            ('Kebab', 1, 'Congelados y Platos Preparados', 'paquete'),
            ('Pizza', 1, 'Congelados y Platos Preparados', 'ud'),
            ('Revueltos', 1, 'Congelados y Platos Preparados', 'bolsa'),
            ('Rollitos', 1, 'Congelados y Platos Preparados', 'paquete'),
            ('San Jacobo', 1, 'Congelados y Platos Preparados', 'paquete'),
            ('Verduras congeladas', 1, 'Congelados y Platos Preparados', 'bolsa'),
            ('Aceitunas', 1, 'Conservas y Encurtidos', 'bote'),
            ('Botes de conserva (variados)', 1, 'Conservas y Encurtidos', 'ud'),
            ('Caldo de pollo', 1, 'Conservas y Encurtidos', 'brik'),
            ('Latillas', 1, 'Conservas y Encurtidos', 'pack'),
            ('Paté', 1, 'Conservas y Encurtidos', 'lata'),
            ('Arroz', 1, 'Despensa (Pastas y Legumbres)', 'paquete'),
            ('Cous cous', 1, 'Despensa (Pastas y Legumbres)', 'paquete'),
            ('Espaguetis', 1, 'Despensa (Pastas y Legumbres)', 'paquete'),
            ('Fideos', 1, 'Despensa (Pastas y Legumbres)', 'paquete'),
            ('Lentejas', 1, 'Despensa (Pastas y Legumbres)', 'paquete'),
            ('Macarrones', 1, 'Despensa (Pastas y Legumbres)', 'paquete'),
            ('Noodles', 1, 'Despensa (Pastas y Legumbres)', 'paquete'),
            ('Barritas', 1, 'Dulces y Snacks', 'caja'),
            ('Colines', 1, 'Dulces y Snacks', 'bolsa'),
            ('Frutos secos', 1, 'Dulces y Snacks', 'bolsa'),
            ('Nachos', 1, 'Dulces y Snacks', 'bolsa'),
            ('Tortitas de maíz', 1, 'Dulces y Snacks', 'paquete'),
            ('Fiambre', 1, 'Embutidos', 'paquete'),
            ('Fuet', 1, 'Embutidos', 'ud'),
            ('Jamón', 1, 'Embutidos', 'paquete'),
            ('Pavo', 1, 'Embutidos', 'paquete'),
            ('Aguacate', 1, 'Frutas y Verduras', 'ud'),
            ('Brócoli', 1, 'Frutas y Verduras', 'ud'),
            ('Calabacín', 1, 'Frutas y Verduras', 'ud'),
            ('Cebolla', 1, 'Frutas y Verduras', 'malla'),
            ('Fruta', 1, 'Frutas y Verduras', 'kg'),
            ('Guisantes', 1, 'Frutas y Verduras', 'bolsa'),
            ('Lechuga', 1, 'Frutas y Verduras', 'ud'),
            ('Maíz', 1, 'Frutas y Verduras', 'lata'),
            ('Patatas', 1, 'Frutas y Verduras', 'malla'),
            ('Pepino', 1, 'Frutas y Verduras', 'ud'),
            ('Pimientos', 1, 'Frutas y Verduras', 'bandeja'),
            ('Pimientos del piquillo', 1, 'Frutas y Verduras', 'bote'),
            ('Puerros', 1, 'Frutas y Verduras', 'manojo'),
            ('Setas', 1, 'Frutas y Verduras', 'bandeja'),
            ('Tomates', 1, 'Frutas y Verduras', 'kg'),
            ('Zanahorias', 1, 'Frutas y Verduras', 'bolsa'),
            ('Huevos', 1, 'Lácteos y Huevos', 'docena'),
            ('Leche', 1, 'Lácteos y Huevos', 'brik'),
            ('Queso (variedad estándar)', 1, 'Lácteos y Huevos', 'cuña'),
            ('Queso feta', 1, 'Lácteos y Huevos', 'paquete'),
            ('Queso light', 1, 'Lácteos y Huevos', 'paquete'),
            ('Queso para gratinar', 1, 'Lácteos y Huevos', 'bolsa'),
            ('Queso para untar', 1, 'Lácteos y Huevos', 'tarrina'),
            ('Tranchetes', 1, 'Lácteos y Huevos', 'paquete'),
            ('Yogures', 1, 'Lácteos y Huevos', 'pack'),
            ('Bollería', 1, 'Panadería', 'paquete'),
            ('Pan de hamburguesa', 1, 'Panadería', 'bolsa'),
            ('Pan de molde', 1, 'Panadería', 'paquete'),
            ('Pan de molde de cereales', 1, 'Panadería', 'paquete'),
            ('Pan pita', 1, 'Panadería', 'paquete'),
            ('Tostadas', 1, 'Panadería', 'bolsa'),
            ('Almejas', 1, 'Pescados y Mariscos', 'malla'),
            ('Atún', 1, 'Pescados y Mariscos', 'pack'),
            ('Caballa', 1, 'Pescados y Mariscos', 'lata'),
            ('Gambas', 1, 'Pescados y Mariscos', 'caja'),
            ('Mejillones', 1, 'Pescados y Mariscos', 'lata'),
            ('Merluza', 1, 'Pescados y Mariscos', 'bandeja'),
            ('Pescado congelado', 1, 'Pescados y Mariscos', 'bolsa'),
            ('Salmón ahumado', 1, 'Pescados y Mariscos', 'sobre'),
            ('Ajo en polvo', 1, 'Salsas y Especias', 'bote'),
            ('Guacamole', 1, 'Salsas y Especias', 'tarrina'),
            ('Hierbabuena', 1, 'Salsas y Especias', 'manojo'),
            ('Ketchup', 1, 'Salsas y Especias', 'bote'),
            ('Mayonesa', 1, 'Salsas y Especias', 'bote'),
            ('Perejil', 1, 'Salsas y Especias', 'manojo'),
            ('Salsa agridulce', 1, 'Salsas y Especias', 'bote'),
            ('Salsa de soja', 1, 'Salsas y Especias', 'bote'),
            ('Salsa de yogur', 1, 'Salsas y Especias', 'bote'),
            ('Salsa teriyaki', 1, 'Salsas y Especias', 'bote'),
            ('Tomate frito', 1, 'Salsas y Especias', 'brik')"""
        
        cursor.execute(consulta)
        conexion.commit()
        print(f"✅ Datos insertados con éxito en la tabla '{tabla_nombre}' de la base de datos '{DB_NAME}'.")
    
    except sqlite3.Error as e:
        print(f"❌ Error al insertar datos en la tabla '{tabla_nombre}': {e}")
    
    finally:
        if 'conexion' in locals() and conexion:
            conexion.close()

if __name__ == "__main__":
    comprobar_base_de_datos()
    #eliminar_base_de_datos()
    #eliminar_tabla("ingredients")
    #insertar_multiples_ingredientes("ingredients")
    pass