import os
import json
import warnings
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import FileResponse
from google.oauth2 import service_account
from google.cloud import bigquery
import httpx
import re

warnings.filterwarnings("ignore", category=UserWarning)

app = FastAPI(title="API Logística NVOCC")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ID_PROYECTO_REAL = "nvocc-496816"

if "GOOGLE_APPLICATION_CREDENTIALS_JSON" in os.environ:
    print("Entorno de producción. Cargando credenciales de BigQuery...")
    try:
        info_claves = json.loads(os.environ["GOOGLE_APPLICATION_CREDENTIALS_JSON"])
        credenciales_gcp = service_account.Credentials.from_service_account_info(info_claves)
    except Exception as e:
        print(f"Error crítico al parsear GOOGLE_APPLICATION_CREDENTIALS_JSON: {e}")
        credenciales_gcp = None
else:
    print("Entorno local. Buscando archivo JSON físico...")
    ruta_base = os.path.dirname(__file__)
    ruta_json = os.path.join(ruta_base, "nvocc-496816-09270102ee36.json")
    if os.path.exists(ruta_json):
        credenciales_gcp = service_account.Credentials.from_service_account_file(ruta_json)
    else:
        credenciales_gcp = None
        print("ALERTA: No se encontró el archivo de credenciales local.")

os.environ["GOOGLE_CLOUD_PROJECT"] = ID_PROYECTO_REAL

CACHE_DATOS = {"df_actual": None}

PARSER_LOCAL_ACTIVO = True

PAISES = [
    "argentina",
    "brasil",
    "chile",
    "uruguay",
    "paraguay",
    "bolivia",
    "estados unidos",
    "china",
    "españa",
    "italia",
    "japon",
    "francia",
    "alemania",
]

def limpiar_dataframe_para_json(df: pd.DataFrame) -> list:
    if df is None or df.empty:
        return []
    df = df.replace([np.inf, -np.inf], 0)
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.strftime('%Y-%m-%d %H:%M:%S').fillna("")
    registros = df.to_dict(orient="records")
    for reg in registros:
        for clave, valor in reg.items():
            if pd.isna(valor) or valor is None:
                reg[clave] = ""
            elif isinstance(valor, float) and (np.isnan(valor) or np.isinf(valor)):
                reg[clave] = 0.0
    return registros

def traer_datos(query):
    if not credenciales_gcp:
        print("Error: No hay credenciales de GCP configuradas.")
        return None
    try:
        client_bq = bigquery.Client(project=ID_PROYECTO_REAL, credentials=credenciales_gcp)
        return client_bq.query(query).to_dataframe()
    except Exception as e:
        print(f"Error en consulta BigQuery: {e}")
        return None

async def generar_con_ia(instruccion, pedido):
    api_key_directa = os.getenv("GEMINI_API_KEY")
    if not api_key_directa:
        return "ERROR_IA: La API Key de Gemini no está configurada."
    
    url = f"https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash:generateContent?key={api_key_directa}"
    
    payload = {
        "contents": [{
            "parts": [{
                "text": f"System Instructions:\n{instruccion}\n\nUser Request:\n{pedido}"
            }]
        }],
        "generationConfig": {
            "temperature": 0.0
        }
    }
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=payload)
        
        if response.status_code == 200:
            res_json = response.json()
            return res_json["candidates"][0]["content"]["parts"][0]["text"]
        elif response.status_code == 429:
            return "ERROR_IA: Créditos/Cuota agotados en Google AI Studio. Revisa la facturación."
        else:
            return f"ERROR_IA: Servidor respondió con código {response.status_code}"
            
    except Exception as e:
        return f"ERROR_IA: {e}"

def interpretar_localmente(texto):
    texto = texto.lower()
    filtros = []

    for pais in PAISES:
        if pais in texto:
            filtros.append(f"UPPER(F.NOMBRE)='{pais.upper()}'")

    if "impo" in texto:
        filtros.append("LEFT(A.UNID_NEGOCIO,1)='I'")

    if "expo" in texto:
        filtros.append("LEFT(A.UNID_NEGOCIO,1)='E'")

    if "maritimo" in texto:
        filtros.append("SUBSTRING(A.UNID_NEGOCIO,2,1)='M'")

    if "aereo" in texto:
        filtros.append("SUBSTRING(A.UNID_NEGOCIO,2,1)='A'")
    
    cliente_sql = detectar_cliente(texto)
    if cliente_sql:
        filtros.append(cliente_sql)
    
    return filtros

def detectar_cliente(texto):
    m = re.search(r'cliente\s+(.+)', texto, re.IGNORECASE)
    if not m:
        return None

    cliente = m.group(1).strip()
    return f"UPPER(C.ENTNOMBRE) LIKE '%{cliente.upper()}%'"

class ConsultaIA(BaseModel):
    pedido: str

@app.get("/cargar-inicial")
def cargar_inicial():
    query = f"""
    SELECT 
        A.ID_OPERACION AS OPERACION, A.DOC_TRANSPORTE AS DOCUMENTO, IFNULL(B.NOMBRE, '') AS DESPACHANTE,
        A.PESO, A.BULTOS, A.FLETE, A.FECHA, C.ENTNOMBRE AS CLIENTE, D.NOMBRE AS EMBARQUE, 
        E.NOMBRE AS DESTINO, IFNULL(G.ENTNOMBRE, '') AS TRANSPORTISTA,
        CASE WHEN LEFT(A.UNID_NEGOCIO, 1) = 'I' THEN 'IMPO' ELSE 'EXPO' END AS TIPO,
        CASE WHEN SUBSTRING(A.UNID_NEGOCIO, 2, 1) = 'M' THEN 'MARITIMO' WHEN SUBSTRING(A.UNID_NEGOCIO, 2, 1) = 'A' THEN 'AEREO' ELSE 'TERRESTRE' END AS MODO,
        F.NOMBRE AS ORIGEN
    FROM `{ID_PROYECTO_REAL}.NVOCC.OPERACIONES_ADUANA` A
    INNER JOIN `{ID_PROYECTO_REAL}.NVOCC.ENTIDADES` C ON C.ENT = A.COD_ENT
    INNER JOIN `{ID_PROYECTO_REAL}.NVOCC.PUERTOS` D ON D.NUMERO = A.COD_ORIGEN
    INNER JOIN `{ID_PROYECTO_REAL}.NVOCC.PUERTOS` E ON E.NUMERO = A.COD_DESTINO
    INNER JOIN `{ID_PROYECTO_REAL}.NVOCC.PAISES` F ON F.PAIS = A.COD_PAIS
    LEFT JOIN `{ID_PROYECTO_REAL}.NVOCC.DESPACHANTES` B ON B.CUIL = A.CUIL_DESPACHANTE
    LEFT JOIN `{ID_PROYECTO_REAL}.NVOCC.ENTIDADES` G ON G.ENT = A.CUIT_TRANSPORTISTA
    LIMIT 100;
    """
    df = traer_datos(query)
    if df is not None:
        CACHE_DATOS["df_actual"] = df.copy()
        return limpiar_dataframe_para_json(df)
    raise HTTPException(status_code=500, detail="Error al cargar datos desde BigQuery")

@app.post("/consultar-ia")
@app.post("/api/consultar-ia")
async def consultar_ia(data: ConsultaIA):
    pedido_min = data.pedido.lower()
    df = CACHE_DATOS["df_actual"]
    
    # --------------------------------------------------------------------------
    # ESCUDO LOCAL 1: INTERCEPTACIÓN DE LISTADO DE CLIENTES Y TRANSPORTISTAS (SIN IA)
    # --------------------------------------------------------------------------
    if "cliente" in pedido_min and ("transportista" in pedido_min or "transp" in pedido_min):
        if df is None or df.empty:
            raise HTTPException(status_code=400, detail="Primero debes cargar datos iniciales o realizar una búsqueda.")
        
        df_res = df.copy()
        if "CLIENTE" in df_res.columns and "TRANSPORTISTA" in df_res.columns:
            df_res["CLIENTE"] = df_res["CLIENTE"].fillna("").astype(str).str.strip()
            df_res["TRANSPORTISTA"] = df_res["TRANSPORTISTA"].fillna("").astype(str).str.strip()
            df_res = df_res[(df_res["CLIENTE"] != "") & (df_res["TRANSPORTISTA"] != "")]
            df_res = df_res[["CLIENTE", "TRANSPORTISTA"]].drop_duplicates()
            
            print("--- PROCESAMIENTO 100% LOCAL: Forzando estructura estricta de 2 columnas ---")
            CACHE_DATOS["df_actual"] = df_res.copy()
            return limpiar_dataframe_para_json(df_res)

    # --------------------------------------------------------------------------
    # ESCUDO LOCAL 3: PRIORIDAD - FILTRAR CONTACTOS VÁLIDOS EN PANTALLA (SIN IA)
    # --------------------------------------------------------------------------
    if df is not None and "TELEFONO" in df.columns and "CORREO" in df.columns:
        if any(w in pedido_min for w in ["valido", "válido", "con", "distinto", "filtrar", "excluir", "limpiar", "solo", "sólo"]):
            print("--- PROCESAMIENTO 100% LOCAL: Filtrando contactos válidos en pantalla ---")
            df_res = df.copy()
            
            df_res["TELEFONO"] = df_res["TELEFONO"].astype(str).str.strip()
            df_res["CORREO"] = df_res["CORREO"].astype(str).str.strip()
            
            df_res = df_res[
                (df_res["TELEFONO"].str.lower() != "sin teléfono") & 
                (df_res["TELEFONO"].str.lower() != "sin telefono") & 
                (df_res["CORREO"].str.lower() != "sin correo") &
                (df_res["TELEFONO"] != "") &
                (df_res["CORREO"] != "")
            ]
            
            CACHE_DATOS["df_actual"] = df_res.copy()
            return limpiar_dataframe_para_json(df_res)

    # --------------------------------------------------------------------------
    # ESCUDO LOCAL 2: BUSCAR CONTACTOS NUEVOS EN BIGQUERY (SIN IA)
    # --------------------------------------------------------------------------
    if "contacto" in pedido_min or "telefono" in pedido_min or "correo" in pedido_min or "email" in pedido_min:
        if df is None or df.empty:
            raise HTTPException(status_code=400, detail="Primero debes cargar o filtrar una lista de operaciones para extraer los contactos.")
        
        clientes_pantalla = df['CLIENTE'].dropna().unique().tolist()
        clientes_limpios = []
        
        for c in clientes_pantalla:
            c_str = str(c).replace("...", "").strip().upper()
            for char in [".", ",", "-", "'", "  "]:
                c_str = c_str.replace(char, "")
            c_str = c_str.strip()
            if c_str:
                clientes_limpios.append(c_str)
        
        if not clientes_limpios:
            raise HTTPException(status_code=400, detail="No hay clientes válidos en pantalla para consultar contactos.")

        lista_sql = ", ".join([f"'{c}'" for c in clientes_limpios])

        query_cont = r"""
        SELECT 
            ENTNOMBRE AS CLIENTE, 
            IFNULL(ENTTELEFON, 'Sin teléfono') AS TELEFONO,
            IFNULL(ENTEMAIL, 'Sin correo') AS CORREO, 
            IFNULL(ENTDIRECCI, 'Sin dirección') AS DIRECCION
        FROM `{0}.NVOCC.ENTIDADES` 
        WHERE (
            UPPER(SUBSTR(COLLATE(ENTNOMBRE, ''), 1, 30)) || 
            REGEXP_REPLACE(REGEXP_REPLACE(UPPER(SUBSTR(COLLATE(ENTNOMBRE, ''), 31)), r'[.,\-\x27]', ''), r'\s+', '')
        ) IN ({1})
        """.format(ID_PROYECTO_REAL, lista_sql)
        
        print("--- PROCESAMIENTO MIXTO: Extrayendo contactos con regla de 30 caracteres intactos ---")
        df_contactos_db = traer_datos(query_cont)
        
        if df_contactos_db is not None and not df_contactos_db.empty:
            CACHE_DATOS["df_actual"] = df_contactos_db.copy()
            return limpiar_dataframe_para_json(df_contactos_db)
        else:
            df_vacio = pd.DataFrame(columns=['CLIENTE', 'TELEFONO', 'CORREO', 'DIRECCION'])
            for c in df['CLIENTE'].dropna().unique().tolist():
                df_vacio.loc[len(df_vacio)] = [c, "Sin teléfono", "Sin correo", "Sin dirección"]
            CACHE_DATOS["df_actual"] = df_vacio.copy()
            return limpiar_dataframe_para_json(df_vacio)
    
    if df is not None:
        decision = "B"
    else:
        decision = "A"
    
    sql_base = f"""
    SELECT 
        A.ID_OPERACION AS OPERACION, A.DOC_TRANSPORTE AS DOCUMENTO, IFNULL(B.NOMBRE, '') AS DESPACHANTE,
        A.PESO, A.BULTOS, A.FLETE, A.FECHA, C.ENTNOMBRE AS CLIENTE, D.NOMBRE AS EMBARQUE, 
        E.NOMBRE AS DESTINO, IFNULL(G.ENTNOMBRE, '') AS TRANSPORTISTA,
        CASE WHEN LEFT(A.UNID_NEGOCIO, 1) = 'I' THEN 'IMPO' ELSE 'EXPO' END AS TIPO,
        CASE WHEN SUBSTRING(A.UNID_NEGOCIO, 2, 1) = 'M' THEN 'MARITIMO' WHEN SUBSTRING(A.UNID_NEGOCIO, 2, 1) = 'A' THEN 'AEREO' ELSE 'TERRESTRE' END AS MODO,
        F.NOMBRE AS ORIGEN
    FROM `{ID_PROYECTO_REAL}.NVOCC.OPERACIONES_ADUANA` A
    INNER JOIN `{ID_PROYECTO_REAL}.NVOCC.ENTIDADES` C ON C.ENT = A.COD_ENT
    INNER JOIN `{ID_PROYECTO_REAL}.NVOCC.PUERTOS` D ON D.NUMERO = A.COD_ORIGEN
    INNER JOIN `{ID_PROYECTO_REAL}.NVOCC.PUERTOS` E ON E.NUMERO = A.COD_DESTINO
    INNER JOIN `{ID_PROYECTO_REAL}.NVOCC.PAISES` F ON F.PAIS = A.COD_PAIS
    LEFT JOIN `{ID_PROYECTO_REAL}.NVOCC.DESPACHANTES` B ON B.CUIL = A.CUIL_DESPACHANTE
    LEFT JOIN `{ID_PROYECTO_REAL}.NVOCC.ENTIDADES` G ON G.ENT = A.CUIT_TRANSPORTISTA
    """

    if PARSER_LOCAL_ACTIVO:
        filtros_locales = interpretar_localmente(data.pedido)
        if filtros_locales:
            where_sql = " AND ".join(filtros_locales)
            sql_final = f"""
            {sql_base}
            WHERE 1=1
            AND {where_sql}
            LIMIT 100
            """
            print(f"--- PARSER HARDCODED SQL: {sql_final} ---")
            df_sql = traer_datos(sql_final)
            if df_sql is not None:
                CACHE_DATOS["df_actual"] = df_sql.copy()
                return limpiar_dataframe_para_json(df_sql)

    decision_ia = await generar_con_ia("Responde solo una letra. A: Nueva búsqueda SQL. B: Filtrar lista actual.", data.pedido)
    if "ERROR_IA" in decision_ia:
        raise HTTPException(status_code=402, detail=decision_ia)

    if "B" in decision_ia.upper() and df is not None:
        ruta_base = os.path.dirname(__file__)
        ruta_pandas = os.path.join(ruta_base, "reglas_pandas.txt")
        instruccion_pandas = "Eres un motor de filtrado para Pandas. Responde únicamente la condición de df.query()."
        if os.path.exists(ruta_pandas):
            with open(ruta_pandas, "r", encoding="utf-8") as f:
                instruccion_pandas = f.read()

        raw_pandas = await generar_con_ia(instruccion_pandas, data.pedido)
        if "ERROR_IA" in raw_pandas:
            raise HTTPException(status_code=402, detail=raw_pandas)
            
        filtro_query = raw_pandas.strip().replace("```python", "").replace("```", "").replace('"', '').replace("'", "")
        filtro_query = filtro_query.split('\n')[0].strip()
        
        if "!= and" in filtro_query:
            filtro_query = filtro_query.replace("!= and", '!= "" and')
        if filtro_query.endswith("and") or filtro_query.endswith("and "):
            filtro_query = filtro_query.rpartition("and")[0].strip()
        
        print(f"--- FILTRO LOCAL EVALUADO EN PANDAS: {filtro_query} ---")
        
        try:
            df_res = df.copy()
            for col in df_res.columns:
                if df_res[col].dtype == object:
                    df_res[col] = df_res[col].fillna("").astype(str)
            
            if filtro_query.upper().startswith("UNIQUE:"):
                columnas_raw = filtro_query.split(":")[1].strip().upper()
                lista_columnas = [c.strip() for c in columnas_raw.split(",")]
                columnas_validas = [c for c in lista_columnas if c in df_res.columns]
                
                if columnas_validas:
                    for col in columnas_validas:
                        df_res = df_res[df_res[col].str.strip() != ""]
                    df_res = df_res[columnas_validas].drop_duplicates()
                    CACHE_DATOS["df_actual"] = df_res.copy()
                    return limpiar_dataframe_para_json(df_res)
                else:
                    raise HTTPException(status_code=400, detail=f"Columnas únicas inválidas: {columnas_raw}")

            if filtro_query:
                df_res = df_res.query(filtro_query, engine='python')
                
            CACHE_DATOS["df_actual"] = df_res.copy()
            return limpiar_dataframe_para_json(df_res)
            
        except Exception as err_pandas:
            print(f"Fallo crítico en procesamiento Pandas: {err_pandas}")
            raise HTTPException(status_code=400, detail=f"Error en filtro local: {filtro_query}")
    else:
        ruta_base = os.path.dirname(__file__)
        ruta_reglas = os.path.join(ruta_base, "reglas_sql.txt")
        contexto_sql = "ERES UN COMPILADOR ESTRICTO DE GOOGLE BIGQUERY SQL. Responde únicamente código SQL que inicie con AND."
        if os.path.exists(ruta_reglas):
            with open(ruta_reglas, "r", encoding="utf-8") as f:
                contexto_sql = f.read()

        raw_sql = await generar_con_ia(contexto_sql, data.pedido)
        if "ERROR_IA" in raw_sql:
            raise HTTPException(status_code=402, detail=raw_sql)

        filtros_sql = raw_sql.strip().replace("```sql", "").replace("```", "").replace(";", "")
        filtros_sql = filtros_sql.split('\n')[0].strip()

        if filtros_sql:
            if filtros_sql.upper().startswith("WHERE "): 
                filtros_sql = filtros_sql[6:].strip()
            
            if filtros_sql.upper().startswith("AND "):
                sql_final = f"{sql_base} WHERE 1=1 {filtros_sql} LIMIT 100;"
            else:
                sql_final = f"{sql_base} WHERE 1=1 AND {filtros_sql} LIMIT 100;"
        else:
            sql_final = f"{sql_base} LIMIT 100;"

        print(f"--- CONSULTA EJECUTADA EN BASE DE DATOS: {sql_final} ---")

        df_sql = traer_datos(sql_final)
        if df_sql is not None:
            CACHE_DATOS["df_actual"] = df_sql.copy()
            return limpiar_dataframe_para_json(df_sql)
        raise HTTPException(status_code=500, detail="Error al consultar BigQuery.")

@app.get("/")
def leer_index():
    ruta_html = os.path.join(os.path.dirname(__file__), "index.html")
    return FileResponse(ruta_html)

# --------------------------------------------------------------------------
# NUEVO ENDPOINT: ENVÍO DE EMAIL NATIVO
# --------------------------------------------------------------------------
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

class PedidoEnvioMail(BaseModel):
    correo_destino: str

@app.post("/enviar-contactos-email")
async def enviar_contactos_email(data: PedidoEnvioMail):
    df = CACHE_DATOS["df_actual"]
    
    if df is None or df.empty:
        raise HTTPException(status_code=400, detail="No hay datos en pantalla para enviar.")
        
    if "TELEFONO" not in df.columns or "CORREO" not in df.columns:
        raise HTTPException(status_code=400, detail="La grilla actual debe ser de contactos para poder enviarla.")

    SMTP_SERVER = "smtp.gmail.com"
    SMTP_PORT = 587
    SMTP_USER = "nvoccsystem@gmail.com"  # <- Acordate de completar esto
    SMTP_PASS = "czey xohd jsqz cnli"  # <- Acordate de completar esto

    try:
        html_tabla = """
        <html>
        <body style="font-family: sans-serif; color: #333;">
            <h2>Listado de Contactos NVOCC Seleccionados</h2>
            <table border="1" cellpadding="8" cellspacing="0" style="border-collapse: collapse; width: 100%;">
                <tr style="background-color: #f2f2f2;">
                    <th>Cliente</th>
                    <th>Teléfono</th>
                    <th>Correo</th>
                    <th>Dirección</th>
                </tr>
        """
        
        for _, fila in df.iterrows():
            html_tabla += f"""
            <tr>
                <td>{str(fila.get('CLIENTE', ''))}</td>
                <td>{str(fila.get('TELEFONO', ''))}</td>
                <td>{str(fila.get('CORREO', ''))}</td>
                <td>{str(fila.get('DIRECCION', ''))}</td>
            </tr>
            """
        html_tabla += "</table></body></html>"

        msg = MIMEMultipart()
        msg['From'] = SMTP_USER
        msg['To'] = data.correo_destino
        msg['Subject'] = "Sistema Logístico NVOCC - Exportación de Contactos"
        msg.attach(MIMEText(html_tabla, 'html'))

        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SMTP_USER, SMTP_PASS)
        server.sendmail(SMTP_USER, data.correo_destino, msg.as_string())
        server.quit()

        print(f"--- CORREO ENVIADO CON EXITO A: {data.correo_destino} ---")
        return {"status": "ok", "message": f"Correo enviado con éxito a {data.correo_destino}"}

    except Exception as e:
        print(f"Fallo al enviar correo electrónico: {e}")
        raise HTTPException(status_code=500, detail=f"Error al enviar el email: {str(e)}")

# Modelo para el envío masivo con plantilla
class PedidoEnvioMasivo(BaseModel):
    asunto: str
    plantilla: str

@app.post("/enviar-masivo-plantilla")
async def enviar_masivo_plantilla(data: PedidoEnvioMasivo):
    df = CACHE_DATOS["df_actual"]
    
    if df is None or df.empty or "CORREO" not in df.columns:
        raise HTTPException(status_code=400, detail="No hay una lista de contactos válida en pantalla.")

    SMTP_SERVER = "smtp.gmail.com"
    SMTP_PORT = 587
    SMTP_USER = "nvoccsystem@gmail.com"  # <- Acordate de completar esto
    SMTP_PASS = "czey xohd jsqz cnli"  # <- Acordate de completar esto

    enviados = 0
    errores = 0

    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SMTP_USER, SMTP_PASS)

        for _, fila in df.iterrows():
            raw_correo = str(fila.get('CORREO', '')).strip()
            nombre_cliente = str(fila.get('CLIENTE', ''))

            if not raw_correo or "sin correo" in raw_correo.lower():
                continue

            # CORRECCIÓN: Reemplazamos puntos y comas por comas ordinarias y separamos en una lista limpia
            lista_destinatarios = [
                m.strip().lower() 
                for m in raw_correo.replace(";", ",").split(",") 
                if "@" in m and m.strip() != ""
            ]

            if not lista_destinatarios:
                continue

            cuerpo_personalizado = data.plantilla.replace("{{CLIENTE}}", nombre_cliente)
            
            msg = MIMEMultipart()
            msg['From'] = SMTP_USER
            # El encabezado visual 'To' sí requiere un string separado por comas
            msg['To'] = ", ".join(lista_destinatarios)
            msg['Subject'] = data.asunto
            msg.attach(MIMEText(cuerpo_personalizado, 'plain'))

            try:
                # CORRECCIÓN CLAVE: sendmail necesita recibir la lista separada para procesar envíos múltiples
                server.sendmail(SMTP_USER, lista_destinatarios, msg.as_string())
                enviados += 1
            except Exception as e:
                print(f"Error enviando a {lista_destinatarios}: {e}")
                errores += 1
        
        server.quit()
        return {"status": "ok", "message": f"Proceso terminado. Enviados: {enviados}, Fallidos: {errores}"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en el servidor de correo: {str(e)}")

@app.get("/nvocc-v2.png")
def obtener_logo():
    ruta_logo = os.path.join(os.path.dirname(__file__), "nvocc-v2.png")
    if os.path.exists(ruta_logo):
        return FileResponse(ruta_logo)
    raise HTTPException(status_code=404, detail="Logo no encontrado")

if __name__ == "__main__":
    import uvicorn
    puerto = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=puerto, loop="asyncio", workers=1)