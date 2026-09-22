import fs from 'node:fs';
import path from 'node:path';

const src = path.resolve('dist/server/prerendered-routes');
const dest = path.resolve('dist/client');

if (fs.existsSync(src)) {
  fs.cpSync(src, dest, { recursive: true });
  console.log('[copy-static] Successfully copied prerendered static HTML routes into dist/client');

  try {
    const files = fs.readdirSync(dest).filter((f) => f.endsWith('.html'));
    for (const file of files) {
      const filePath = path.join(dest, file);
      const content = fs.readFileSync(filePath, 'utf8');
      const sanitized = content.replace(/url\([a-zA-Z]:\/[^)]+\/fonts\/([^)]+)\)/g, 'url(/_next/static/fonts/$1)');
      if (sanitized !== content) {
        fs.writeFileSync(filePath, sanitized, 'utf8');
        console.log(`[copy-static] Sanitized local font paths in ${file}`);
      }
    }
  } catch (err) {
    console.warn('[copy-static] Path cleanup warning:', err);
  }
} else {
  console.warn('[copy-static] Warning: prerendered-routes directory not found at', src);
}

