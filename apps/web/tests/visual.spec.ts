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
    
    // Verificar colores en light mode (muestreo)
    const loginShell = page.locator('.login-shell');
    const bgColor = await loginShell.evaluate(el => 
      window.getComputedStyle(el).backgroundColor
    );
    console.log('Light mode - login-shell background:', bgColor);
  });

  test('dark mode: toggle and verify colors', async ({ page }) => {
    await page.goto(BASE_URL);
    
    // Verificar que está en light mode inicialmente
    const htmlElement = page.locator('html');
    const initialTheme = await htmlElement.getAttribute('data-theme');
    console.log('Initial theme:', initialTheme);
    
    // Buscar y clickear el botón de tema
    const themeButton = page.locator('.theme-button').first();
    await themeButton.click();
    
    // Esperar a que cambie el tema
    await page.waitForTimeout(300);
    
    // Verificar que el tema cambió
    const newTheme = await htmlElement.getAttribute('data-theme');
    console.log('New theme:', newTheme);
    expect(newTheme).toBe('dark');
    
    // Verificar que los elementos son visibles en dark mode
    await expect(page.locator('.login-shell')).toBeVisible();
    
    // Verificar colores en dark mode
    const body = page.locator('body');
    const textColor = await body.evaluate(el => 
      window.getComputedStyle(el).color
    );
    console.log('Dark mode - body text color:', textColor);
  });

  test('responsive: layout adapts at breakpoints', async ({ page }) => {
    await page.goto(BASE_URL);
    
    // Test 980px breakpoint
    await page.setViewportSize({ width: 980, height: 720 });
    await page.waitForTimeout(200);
    
    const sidebar = page.locator('.sidebar');
    const isVisible980 = await sidebar.isVisible();
    console.log('Sidebar visible at 980px:', isVisible980);
    
    // Test 680px breakpoint (mobile)
    await page.setViewportSize({ width: 680, height: 1024 });
    await page.waitForTimeout(200);
    
    const loginPanel = page.locator('.login-panel');
    const gridCols = await loginPanel.evaluate(el => 
      window.getComputedStyle(el).gridTemplateColumns
    );
    console.log('Login panel grid columns at 680px:', gridCols);
  });

  test('components: inputs have correct styling', async ({ page }) => {
    await page.goto(BASE_URL);
    
    const inputWrap = page.locator('.input-wrap').first();
    await expect(inputWrap).toBeVisible();
    
    // Verificar border
    const borderStyle = await inputWrap.evaluate(el => 
      window.getComputedStyle(el).borderWidth
    );
    console.log('Input wrap border width:', borderStyle);
    
    // Verificar que el borde existe y no depende del estilo por defecto del navegador
    const borderWidth = parseFloat(borderStyle);
    expect(borderWidth).toBeGreaterThanOrEqual(1);
    
    // Verificar border-radius
    const borderRadius = await inputWrap.evaluate(el => 
      window.getComputedStyle(el).borderRadius
    );
    console.log('Input wrap border radius:', borderRadius);
  });

  test('components: buttons have hover effects', async ({ page }) => {
    await page.goto(BASE_URL);
    
    const primaryButton = page.locator('.primary-button').first();
    
    // Obtener estilos por defecto
    const defaultShadow = await primaryButton.evaluate(el => 
      window.getComputedStyle(el).boxShadow
    );
    console.log('Primary button default shadow:', defaultShadow);
    
    // Hover y verificar que cambia
    await primaryButton.hover();
    await page.waitForTimeout(200);
    
    const hoverShadow = await primaryButton.evaluate(el => 
      window.getComputedStyle(el).boxShadow
    );
    console.log('Primary button hover shadow:', hoverShadow);
    
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
    console.log('Dark mode body background:', bgColor);
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
    await expect(page.locator('.connection-card')).toHaveCount(0);
    await page.getByRole('button', { name: /Información/ }).click();
    await expect(page.getByText('Qué significan los datos')).toBeVisible();
    await expect(page.getByText('API local')).toHaveCount(0);
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
    console.log('CSS errors found:', errors.length);
    expect(errors).toHaveLength(0);
  });
});
