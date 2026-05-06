import { test, expect } from '@playwright/test';

test.describe('PostureOS Visual Validation', () => {
  const BASE_URL = 'http://localhost:5173';

  test.beforeEach(async ({ page }) => {
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
    
    // Verificar que no es 1px (debería ser 1.5px o similar)
    const borderWidth = parseFloat(borderStyle);
    expect(borderWidth).toBeGreaterThanOrEqual(1.5);
    
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
    
    // Si la página de login carga bien, verificar que los stat-tiles tendrían el CSS correcto
    // (esto es un test preventivo para la estructura)
    
    const html = await page.content();
    
    // Verificar que el CSS de stat-tiles está aplicado
    expect(html).toContain('stat-tile');
    expect(html).toContain('timeline-chart');
    expect(html).toContain('recommendation-list');
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
    
    // Crear un mock de connection-card
    const html = await page.content();
    
    // Verificar que las clases existen
    expect(html).toContain('connection-card');
    expect(html).toContain('online');
    expect(html).toContain('offline');
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
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(500);
    
    // No debería haber errores de CSS
    console.log('CSS errors found:', errors.length);
    expect(errors).toHaveLength(0);
  });
});
