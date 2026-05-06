# PostureOS web

Interfaz React para revisar imagenes frontales y laterales contra el backend FastAPI local del proyecto.

La pantalla principal permite comprobar la camara, validar encuadre e iniciar una sesion de seguimiento. La captura periodica de fotogramas queda preparada a nivel de interfaz para conectarla al worker/backend en la siguiente iteracion.

## Desarrollo local

Desde la raiz del repositorio:

```bash
python -m pip install -r requirements-api.txt
```

Para Windows:
```bash
$env:PYTHONPATH = "src"
uvicorn ergonomics.api:app --host 0.0.0.0 --port 8000
```

Para Linux and macOS:
```bash
PYTHONPATH=src uvicorn ergonomics.api:app --host 0.0.0.0 --port 8000
```

En otra terminal:

```bash
cd apps/web
npm install
npm run dev
```

La web usa `VITE_API_BASE_URL` para decidir a que backend local conectarse. Si no existe, usa `http://localhost:8000`.

El acceso a camara funciona en `localhost` durante desarrollo. Si se accede desde otro dispositivo de la red, el navegador normalmente exigira HTTPS.

## Instalacion local

El modo principal del proyecto es local-first:

- la camara se procesa en el ordenador del usuario
- la base de datos SQLite queda en ese ordenador
- las imagenes no se guardan
- el usuario puede borrar sus analisis desde su propio perfil
- no hace falta Vercel ni servidor corporativo

## Backend y privacidad

El backend ya expone endpoints de autenticacion y persistencia:

- `POST /api/auth/register`
- `POST /api/auth/login`
- `GET /api/auth/me`
- `POST /api/auth/logout`
- `GET /api/analyses`
- `DELETE /api/analyses`
- `GET /api/summary`
- `POST /api/analyze/front`
- `POST /api/analyze/lateral`

Variables utiles en la instalacion local:

```bash
ERGONOMICS_DB_PATH=/ruta/local/postureos.sqlite3
ERGONOMICS_SEED_DEFAULT_USERS=true
```

Si no defines `ERGONOMICS_SECRET_KEY`, el backend genera una clave local en un archivo `.key` junto a la base de datos. La base de datos guarda usuarios, sesiones y resultados cifrados.

Usuarios iniciales en modo local:

- `admin` / `admin`, rol tecnico
- `Pablo` / `1234`, usuario normal

La pantalla de login de la web sigue siendo visual hasta conectar estos endpoints desde React.
