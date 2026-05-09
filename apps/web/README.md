# PostureOS Web

Interfaz React/Vite para usar el backend local de análisis postural.

La pantalla principal permite validar encuadre, iniciar una sesión de seguimiento y capturar vistas frontales o combinadas frontal+lateral. El historial y las estadísticas se guardan por usuario en la base de datos local del backend.

## Desarrollo Local

Desde la raíz del repositorio:

```bash
python -m pip install -r requirements-api.txt
PYTHONPATH=src uvicorn ergonomics.api:app --host 0.0.0.0 --port 8000
```

En otra terminal:

```bash
cd apps/web
npm install
npm run dev
```

La variable `VITE_API_BASE_URL` permite cambiar el backend. Si no existe, la web usa `http://localhost:8000`.

## Scripts

```bash
npm run dev        # servidor Vite
npm run typecheck  # TypeScript sin emitir archivos
npm run build      # build de producción
npm test           # Playwright
```

Playwright espera la web en `http://localhost:5173`, salvo que se defina `PLAYWRIGHT_BASE_URL`.

## Backend

Endpoints usados por la web:

- `POST /api/auth/login`
- `GET /api/auth/me`
- `POST /api/auth/logout`
- `GET /api/analyses`
- `DELETE /api/analyses`
- `GET /api/summary`
- `POST /api/analyze/front`
- `POST /api/analyze/lateral`
- `POST /api/analyze/combined`
- `POST /api/dev/analyze-image` solo para usuarios con rol `dev`

Usuarios iniciales en modo local:

- `admin` / `admin`, rol técnico
- `Pablo` / `1234`, usuario normal

## Privacidad

El modo principal es local-first:

- la cámara se procesa en el ordenador del usuario
- la base de datos SQLite queda en ese ordenador
- las imágenes de análisis normal son temporales
- los resultados se guardan cifrados
- cada usuario puede borrar sus análisis

La herramienta de depuración visual del rol técnico sí guarda original y overlay anotado en `data/app/dev_captures/` para revisar el pipeline.

## Pruebas LAN

1. Arranca el backend con `--host 0.0.0.0`.
2. Arranca el frontend con `npm run dev`.
3. Abre `http://IP_DEL_HOST:5173` desde otro equipo.
4. En el campo de API local usa `http://IP_DEL_HOST:8000`.

Algunos navegadores solo permiten cámara en `localhost` o contextos seguros. Para la demo final de cámara, usa `http://localhost:5173` en el mismo equipo.
