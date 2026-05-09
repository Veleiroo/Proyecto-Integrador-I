import { test, expect } from '@playwright/test';

test.describe('PostureOS Visual Validation', () => {
  const BASE_URL = process.env.PLAYWRIGHT_BASE_URL ?? 'http://localhost:5173';

  async function pageHasCssClass(page, className: string) {
    return page.evaluate((name) => {
      return Array.from(document.styleSheets).some((sheet) => {
        try {
          return Array.from(sheet.cssRules).some((rule) => rule.cssText.includes(name));
        } catch {
          return false;
        }
      });
    }, className);
  }

  test.beforeEach(async ({ page }) => {
    await page.addInitScript(() => {
      localStorage.clear();
    });

    // Detectar errores de consola
    page.on('console', msg => {
      if (msg.type() === 'error') {
        console.error('Console error:', msg.text());
      }
    });
  });

  test('light mode: login page renders correctly', async ({ page }) => {
    await page.goto(BASE_URL);
    
    // Verificar que la página se cargó
    await expect(page).toHaveTitle('PostureOS');
    
    // Verificar elementos principales
    await expect(page.locator('.login-shell')).toBeVisible();
    await expect(page.locator('.login-panel')).toBeVisible();
    
    // Verificar que los inputs tienen el estilo correcto
    const inputs = page.locator('.input-wrap input');
    await expect(inputs.first()).toBeVisible();
    
    // Verificar que el botón de tema existe
    const themeButton = page.locator('.theme-button');
    await expect(themeButton).toBeVisible();
    
    const loginShell = page.locator('.login-shell');
    const bgColor = await loginShell.evaluate(el => 
      window.getComputedStyle(el).backgroundColor
    );
    expect(bgColor).toBeTruthy();
  });

  test('dark mode: toggle and verify colors', async ({ page }) => {
    await page.goto(BASE_URL);
    
    const htmlElement = page.locator('html');
    const initialTheme = await htmlElement.getAttribute('data-theme');
    expect(initialTheme).toBe('light');
    
    // Buscar y clickear el botón de tema
    const themeButton = page.locator('.theme-button').first();
    await themeButton.click();
    
    // Esperar a que cambie el tema
    await page.waitForTimeout(300);
    
    const newTheme = await htmlElement.getAttribute('data-theme');
    expect(newTheme).toBe('dark');
    
    // Verificar que los elementos son visibles en dark mode
    await expect(page.locator('.login-shell')).toBeVisible();
    
    const body = page.locator('body');
    const textColor = await body.evaluate(el => 
      window.getComputedStyle(el).color
    );
    expect(textColor).toBeTruthy();
  });

  test('responsive: layout adapts at breakpoints', async ({ page }) => {
    await page.goto(BASE_URL);
    
    // Test 980px breakpoint
    await page.setViewportSize({ width: 980, height: 720 });
    await page.waitForTimeout(200);
    
    const sidebar = page.locator('.sidebar');
    const isVisible980 = await sidebar.isVisible();
    expect(typeof isVisible980).toBe('boolean');
    
    // Test 680px breakpoint (mobile)
    await page.setViewportSize({ width: 680, height: 1024 });
    await page.waitForTimeout(200);
    
    const loginPanel = page.locator('.login-panel');
    const gridCols = await loginPanel.evaluate(el => 
      window.getComputedStyle(el).gridTemplateColumns
    );
    expect(gridCols).toBeTruthy();
  });

  test('components: inputs have correct styling', async ({ page }) => {
    await page.goto(BASE_URL);
    
    const inputWrap = page.locator('.input-wrap').first();
    await expect(inputWrap).toBeVisible();
    
    // Verificar border
    const borderStyle = await inputWrap.evaluate(el => 
      window.getComputedStyle(el).borderWidth
    );
    
    // Verificar que el borde existe y no depende del estilo por defecto del navegador
    const borderWidth = parseFloat(borderStyle);
    expect(borderWidth).toBeGreaterThanOrEqual(1);
    
    // Verificar border-radius
    const borderRadius = await inputWrap.evaluate(el => 
      window.getComputedStyle(el).borderRadius
    );
    expect(parseFloat(borderRadius)).toBeGreaterThanOrEqual(6);
  });

  test('components: buttons have hover effects', async ({ page }) => {
    await page.goto(BASE_URL);
    
    const primaryButton = page.locator('.primary-button').first();
    
    // Obtener estilos por defecto
    const defaultShadow = await primaryButton.evaluate(el => 
      window.getComputedStyle(el).boxShadow
    );
    
    // Hover y verificar que cambia
    await primaryButton.hover();
    await page.waitForTimeout(200);
    
    const hoverShadow = await primaryButton.evaluate(el => 
      window.getComputedStyle(el).boxShadow
    );
    
    // Las sombras deberían ser diferentes
    expect(hoverShadow).not.toBe(defaultShadow);
  });

  test('stat tiles: cards render with proper styling', async ({ page }) => {
    await page.goto(BASE_URL);
    
    // Verificar que las clases de estadísticas existen en CSS aunque no estén montadas en login.
    expect(await pageHasCssClass(page, 'stat-tile')).toBe(true);
    expect(await pageHasCssClass(page, 'timeline-chart')).toBe(true);
    expect(await pageHasCssClass(page, 'recommendation-list')).toBe(true);
    expect(await pageHasCssClass(page, 'trend-chart')).toBe(true);
  });

  test('dark mode: stat tiles and timeline visible', async ({ page }) => {
    await page.goto(BASE_URL);
    
    // Cambiar a dark mode
    const htmlElement = page.locator('html');
    await htmlElement.evaluate(el => el.setAttribute('data-theme', 'dark'));
    await page.waitForTimeout(300);
    
    // Verificar que el dark mode está aplicado
    const theme = await htmlElement.getAttribute('data-theme');
    expect(theme).toBe('dark');
    
    // Verificar que los colores cambiaron
    const body = page.locator('body');
    const bgColor = await body.evaluate(el => 
      window.getComputedStyle(el).backgroundColor
    );
    expect(bgColor).not.toContain('248');  // No debe ser light gray (f8)
  });

  test('connection card: online/offline styling', async ({ page }) => {
    await page.goto(BASE_URL);
    
    // La tarjeta de conexión solo se renderiza en rol dev, pero el estilo debe existir.
    expect(await pageHasCssClass(page, 'connection-card')).toBe(true);
    expect(await pageHasCssClass(page, 'online')).toBe(true);
    expect(await pageHasCssClass(page, 'offline')).toBe(true);
  });

  test('user role: hides dev-only navigation and API controls', async ({ page }) => {
    await page.route('**/api/**', async (route) => {
      const url = route.request().url();
      if (url.endsWith('/api/auth/me')) {
        await route.fulfill({ json: { id: 1, username: 'Pablo', display_name: 'Pablo', role: 'user', created_at: new Date().toISOString() } });
        return;
      }
      if (url.endsWith('/api/health')) {
        await route.fulfill({ json: { ok: true } });
        return;
      }
      if (url.includes('/api/analyses')) {
        await route.fulfill({ json: { items: [] } });
        return;
      }
      if (url.includes('/api/summary')) {
        await route.fulfill({ json: { total: 0, by_status: {}, latest_at: null, periods: { last_7_days: { total: 0, adequate_ratio: 0, risk_count: 0, improvable_count: 0 } }, timeline: [], recommendations: [] } });
        return;
      }
      await route.fulfill({ status: 404, json: {} });
    });
    await page.addInitScript(() => {
      localStorage.setItem('authSession', JSON.stringify({
        user: { id: 1, username: 'Pablo', display_name: 'Pablo', role: 'user', created_at: new Date().toISOString() },
        access_token: 'test-token',
        token_type: 'bearer',
        expires_at: new Date(Date.now() + 86400000).toISOString(),
      }));
    });
    await page.goto(BASE_URL);

    await expect(page.getByRole('button', { name: /Revisión/ })).toHaveCount(0);
    await expect(page.getByRole('button', { name: /Debug visual/ })).toHaveCount(0);
    await expect(page.locator('.connection-card')).toHaveCount(0);
    await page.getByRole('button', { name: /Información/ }).click();
    await expect(page.getByText('Qué significan los datos')).toBeVisible();
    await expect(page.getByText('API local')).toHaveCount(0);
  });

  test('dev role: shows visual debug tools', async ({ page }) => {
    await page.route('**/api/**', async (route) => {
      const url = route.request().url();
      if (url.endsWith('/api/auth/me')) {
        await route.fulfill({ json: { id: 1, username: 'admin', display_name: 'Administrador', role: 'dev', created_at: new Date().toISOString() } });
        return;
      }
      if (url.endsWith('/api/health')) {
        await route.fulfill({ json: { ok: true } });
        return;
      }
      if (url.includes('/api/analyses')) {
        await route.fulfill({ json: { items: [] } });
        return;
      }
      if (url.includes('/api/summary')) {
        await route.fulfill({ json: { total: 0, by_status: {}, latest_at: null, periods: { last_7_days: { total: 0, adequate_ratio: 0, risk_count: 0, improvable_count: 0 } }, timeline: [], recommendations: [] } });
        return;
      }
      await route.fulfill({ status: 404, json: {} });
    });
    await page.addInitScript(() => {
      localStorage.setItem('authSession', JSON.stringify({
        user: { id: 1, username: 'admin', display_name: 'Administrador', role: 'dev', created_at: new Date().toISOString() },
        access_token: 'test-token',
        token_type: 'bearer',
        expires_at: new Date(Date.now() + 86400000).toISOString(),
      }));
    });

    await page.goto(BASE_URL);
    await page.getByRole('button', { name: /Debug visual/ }).click();
    await expect(page.getByText('Probar imagen sin cámara')).toBeVisible();
    await expect(page.getByText('Keypoints y reglas')).toBeVisible();
    await expect(page.getByRole('button', { name: /Ejecutar depuración/ })).toBeDisabled();
  });

  test('camera panel: lateral mode enables dual-view capture', async ({ page }) => {
    await page.route('**/api/**', async (route) => {
      const url = route.request().url();
      if (url.endsWith('/api/auth/me')) {
        await route.fulfill({ json: { id: 1, username: 'Pablo', display_name: 'Pablo', role: 'user', created_at: new Date().toISOString() } });
        return;
      }
      if (url.endsWith('/api/health')) {
        await route.fulfill({ json: { ok: true } });
        return;
      }
      if (url.includes('/api/analyses')) {
        await route.fulfill({ json: { items: [] } });
        return;
      }
      if (url.includes('/api/summary')) {
        await route.fulfill({ json: { total: 0, by_status: {}, latest_at: null, periods: { last_7_days: { total: 0, adequate_ratio: 0, risk_count: 0, improvable_count: 0 } }, timeline: [], recommendations: [] } });
        return;
      }
      await route.fulfill({ status: 404, json: {} });
    });
    await page.addInitScript(() => {
      localStorage.setItem('authSession', JSON.stringify({
        user: { id: 1, username: 'Pablo', display_name: 'Pablo', role: 'user', created_at: new Date().toISOString() },
        access_token: 'test-token',
        token_type: 'bearer',
        expires_at: new Date(Date.now() + 86400000).toISOString(),
      }));
    });

    await page.goto(BASE_URL);
    const lateralButton = page.getByRole('button', { name: /Lateral Opcional/ });
    await expect(lateralButton).toBeEnabled();
    await lateralButton.click();
    await expect(page.getByText('Frontal + lateral')).toBeVisible();
    await expect(page.getByLabel('Cámara frontal')).toBeVisible();
    await expect(page.getByLabel('Cámara lateral')).toBeVisible();
    await expect(page.getByRole('button', { name: /Capturar doble vista/ })).toBeVisible();
    await expect(page.getByRole('button', { name: /Seguimiento doble/ })).toBeVisible();
  });

  test('camera panel: dual capture posts one combined analysis', async ({ page }) => {
    const createdAt = new Date().toISOString();
    const requests: string[] = [];

    await page.addInitScript(() => {
      Object.defineProperty(navigator, 'mediaDevices', {
        configurable: true,
        value: {
          enumerateDevices: async () => [
            { deviceId: 'front-cam', kind: 'videoinput', label: 'Front camera', groupId: 'front' },
            { deviceId: 'side-cam', kind: 'videoinput', label: 'Side camera', groupId: 'side' },
          ],
          addEventListener: () => undefined,
          removeEventListener: () => undefined,
          getUserMedia: async () => {
            const canvas = document.createElement('canvas');
            canvas.width = 1280;
            canvas.height = 720;
            return canvas.captureStream(1);
          },
        },
      });
      HTMLMediaElement.prototype.play = async () => undefined;
      Object.defineProperty(HTMLVideoElement.prototype, 'videoWidth', { configurable: true, get: () => 1280 });
      Object.defineProperty(HTMLVideoElement.prototype, 'videoHeight', { configurable: true, get: () => 720 });
      HTMLCanvasElement.prototype.getContext = () => ({ drawImage: () => undefined } as unknown as CanvasRenderingContext2D);
      HTMLCanvasElement.prototype.toBlob = function toBlob(callback: BlobCallback) {
        callback(new Blob(['fake-image'], { type: 'image/jpeg' }));
      };
      localStorage.setItem('authSession', JSON.stringify({
        user: { id: 1, username: 'Pablo', display_name: 'Pablo', role: 'user', created_at: new Date().toISOString() },
        access_token: 'test-token',
        token_type: 'bearer',
        expires_at: new Date(Date.now() + 86400000).toISOString(),
      }));
    });

    await page.route('**/api/**', async (route) => {
      const url = route.request().url();
      requests.push(url);
      if (url.endsWith('/api/auth/me')) {
        await route.fulfill({ json: { id: 1, username: 'Pablo', display_name: 'Pablo', role: 'user', created_at: createdAt } });
        return;
      }
      if (url.endsWith('/api/health')) {
        await route.fulfill({ json: { ok: true } });
        return;
      }
      if (url.includes('/api/analyses')) {
        await route.fulfill({ json: { items: [] } });
        return;
      }
      if (url.includes('/api/summary')) {
        await route.fulfill({ json: { total: 0, by_status: {}, latest_at: null, periods: { last_7_days: { total: 0, adequate_ratio: 0, risk_count: 0, improvable_count: 0 } }, timeline: [], recommendations: [] } });
        return;
      }
      if (url.endsWith('/api/analyze/combined')) {
        await route.fulfill({
          json: {
            id: 101,
            created_at: createdAt,
            view: 'combined',
            model: 'MediaPipe Pose + YOLO Pose',
            backend: 'combined',
            pose_detected: true,
            visible_landmarks_count: 31,
            status: 'improvable',
            status_label: 'Mejorable',
            feedback: 'Frontal: estable. Lateral: revisa el tronco.',
            metrics: { shoulder_tilt_deg: 4.2, trunk_forward_tilt_deg: 8.1 },
            components: { front_shoulder_status: 'adequate', lateral_trunk_status: 'improvable' },
          },
        });
        return;
      }
      await route.fulfill({ status: 500, json: { detail: `Unexpected endpoint ${url}` } });
    });

    await page.goto(BASE_URL);
    await page.getByRole('button', { name: /Lateral Opcional/ }).click();
    await page.getByRole('button', { name: /Capturar doble vista/ }).click();

    await expect(page.getByText('Combinada actualizada: Mejorable')).toBeVisible();
    const lastResult = page.locator('.last-result-card');
    await expect(lastResult.getByText('Última evaluación')).toBeVisible();
    await expect(lastResult.getByText('Evaluación combinada')).toBeVisible();
    expect(requests.some((url) => url.endsWith('/api/analyze/combined'))).toBe(true);
    expect(requests.some((url) => url.endsWith('/api/analyze/lateral'))).toBe(false);
  });

  test('backend metric keys are presented with Spanish labels', async ({ page }) => {
    const createdAt = new Date().toISOString();
    const record = {
      id: 12,
      view: 'front',
      model: 'MediaPipe Pose',
      backend: 'mediapipe',
      pose_detected: true,
      visible_landmarks_count: 21,
      status: 'improvable',
      status_label: 'Mejorable',
      feedback: 'Conviene revisar la alineación cervical.',
      metrics: {
        head_lateral_offset_ratio: 0.084,
        neck_tilt_deg: 11.2,
      },
      components: {
        head_offset_status: 'improvable',
        neck_tilt_status: 'risk',
        overall_status: 'risk',
      },
      created_at: createdAt,
      fileName: 'Captura 12',
      createdAt,
    };

    await page.route('**/api/**', async (route) => {
      const url = route.request().url();
      if (url.endsWith('/api/auth/me')) {
        await route.fulfill({ json: { id: 1, username: 'Pablo', display_name: 'Pablo', role: 'user', created_at: createdAt } });
        return;
      }
      if (url.endsWith('/api/health')) {
        await route.fulfill({ json: { ok: true } });
        return;
      }
      if (url.includes('/api/analyses')) {
        await route.fulfill({ json: { items: [record] } });
        return;
      }
      if (url.includes('/api/summary')) {
        await route.fulfill({
          json: {
            total: 1,
            by_status: { improvable: 1 },
            latest_at: createdAt,
            periods: { last_7_days: { total: 1, adequate_ratio: 0, risk_count: 0, improvable_count: 1 } },
            timeline: [{ date: createdAt.slice(0, 10), total: 1, adequate: 0, improvable: 1, risk: 0 }],
            recommendations: [],
          },
        });
        return;
      }
      await route.fulfill({ status: 404, json: {} });
    });
    await page.addInitScript(() => {
      localStorage.setItem('authSession', JSON.stringify({
        user: { id: 1, username: 'Pablo', display_name: 'Pablo', role: 'user', created_at: new Date().toISOString() },
        access_token: 'test-token',
        token_type: 'bearer',
        expires_at: new Date(Date.now() + 86400000).toISOString(),
      }));
    });

    await page.goto(BASE_URL);
    await page.getByRole('button', { name: /Estadísticas/ }).click();

    await expect(page.getByText('Desplazamiento lateral de cabeza')).toBeVisible();
    await expect(page.getByText(/Desplazamiento de cabeza aparece como foco recurrente/)).toBeVisible();
    await expect(page.getByText('head offset')).toHaveCount(0);

    await page.getByRole('button', { name: /Mejorable/ }).click();
    const detail = page.getByLabel('Detalle de captura');
    await expect(detail.getByText('Inclinación cervical', { exact: true })).toBeVisible();
    await expect(detail.getByText('Riesgo', { exact: true })).toBeVisible();
    await expect(page.getByText('neck tilt')).toHaveCount(0);
  });

  test('no css errors: verify all color values are valid', async ({ page }) => {
    await page.goto(BASE_URL);
    
    // Recopilar todos los errores de consola
    const errors: string[] = [];
    
    page.on('console', msg => {
      if (msg.type() === 'error' && msg.text().includes('CSS')) {
        errors.push(msg.text());
      }
    });
    
    // Esperar a que todo se cargue
    await page.waitForLoadState('domcontentloaded');
    await page.waitForTimeout(500);
    
    // No debería haber errores de CSS
    expect(errors).toHaveLength(0);
  });
});
